import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, PILToTensor
from PIL import Image
import torch.nn.functional as F
import re
import numpy as np
import math
import random

from transform import world2image, cam2world, world2cam
from .augment import Augmentation

class SatImageAlign:
    def __init__(self):
        pass

    @torch.no_grad()
    def __call__(self,
                 image: Image.Image,
                 K_sat: torch.Tensor,
                 R_sat2world: torch.Tensor,
                 t_sat2world: torch.Tensor,
                 rotate: float,
                 t_sat2world_new: torch.Tensor):
        
        w, h = image.size
        
        # Part 1: Calculate the new satellite pose (R, t). This logic is unchanged.
        cos_r, sin_r = math.cos(rotate), math.sin(rotate)
        R_2d = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]], dtype=torch.float32)
        
        R_expand = torch.eye(3, dtype=torch.float32)
        R_expand[:2, :2] = R_2d
        R_sat2world_new = R_sat2world @ R_expand

        t_center = t_sat2world_new.clone()
        t_center[-1] = 0

        pix_init = world2image(t_center[None, None, :],
                               K_sat[None],
                               R_sat2world[None],
                               t_sat2world[None]).reshape(2)  # (2)

        cw, ch = (w-1)/2, (h-1)/2
        trans_params = -R_2d @ torch.tensor([cw,ch]) + pix_init
        affine_params = [R_2d[0,0].item(), R_2d[0,1].item(), trans_params[0].item(),
                         R_2d[1,0].item(), R_2d[1,1].item(), trans_params[1].item()]

        # Part 3: Apply the transformation using Pillow's optimized C backend.
        image_new = image.transform(
            image.size,
            Image.AFFINE,
            data=affine_params,
            resample=Image.BILINEAR
        )

        return image_new, R_sat2world_new, t_sat2world_new


def pfm2tensor(file_path, size):
    """Convert a PFM file to a torch.Tensor."""

    with open(file_path, 'rb') as f:
        header = f.readline().decode().rstrip()
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('utf-8'))
        if not dim_match:
            raise ValueError("Malformed PFM header.")
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().decode().strip())
        endian = '<' if scale < 0 else '>'
        scale = abs(scale)

        data = np.fromfile(f, endian + 'f')
        shape = (height, width)
        depth = np.reshape(data, shape) # shape (height, width)
        depth = torch.from_numpy(depth)

    depth = F.interpolate(depth[None,None], (size, size), mode='bilinear', align_corners=False)
    depth = depth.squeeze()
        
    return depth


class LocDataset(Dataset):
    def __init__(self, pairs, left_image_size, sat_image_size, aug=False,
                 max_init_offset=20, max_aug_offset=50, max_aug_rotate=180, seed=None):

        super().__init__()
        self.pairs = pairs
        self.left_transform = Compose([
                                    Resize(left_image_size),
                                    ToTensor()])
        self.sat_transform = Compose([
                                Resize(sat_image_size),
                                ToTensor()])

        self.left_image_size = left_image_size
        self.sat_image_size = sat_image_size

        self.sat_align = SatImageAlign()
        self.aug = Augmentation(seed) if aug else None

        self.max_init_offset = max_init_offset
        self.max_aug_offset = max_aug_offset
        self.max_aug_rotate = max_aug_rotate
    
    def __len__(self):
        return len(self.pairs)

    def deep_copy_dict(self, data):
        
        new_data = {}
        for key, val in data.items():
            if isinstance(val, str):
                new_data[key] = val
            elif isinstance(val, float):
                new_data[key] = val
            elif isinstance(val, torch.Tensor):
                new_data[key] = val.clone()
            else:
                NotImplementedError(f'Unsupported data type: {type(val)}')
        return new_data
    
    def __getitem__(self, index):
        
        pair = self.pairs[index]
        pair = self.deep_copy_dict(pair)

        left_image_path = pair['left_image_path']
        left_depth_path = pair['left_depth_path']
        #right_image_path = pair['right_image_path']
        sat_image_path = pair['sat_image_path']
        R_sat2world = pair['R_sat2world']
        t_sat2world = pair['t_sat2world']
        K_sat = pair['K_sat']
        # R_left2world = pair['R_left2world']
        t_left2world = pair['t_left2world']
        x_offset_ratio = pair['x_offset_ratio']
        y_offset_ratio = pair['y_offset_ratio']
        x_offset = x_offset_ratio * self.max_init_offset
        y_offset = y_offset_ratio * self.max_init_offset
        t_left2world_init = t_left2world.clone()
        t_left2world_init[0:2] += torch.tensor([x_offset, y_offset])

        with Image.open(left_image_path).convert('RGB') as image:
            left_image_scale = self.left_image_size / max(image.size)

            left_image = self.aug(image) if self.aug else image
            left_image = self.left_transform(left_image)
        
        with Image.open(sat_image_path).convert('RGB') as image:
            sat_image_scale = self.sat_image_size / max(image.size)

            if self.aug:
                random_offset = (torch.rand(2) * 2 - 1) * self.max_aug_offset
                random_rotate = (random.random() * 2 - 1) * self.max_aug_rotate / 180 * np.pi
                t_sat2world_aug = t_sat2world.clone()
                t_sat2world_aug[0:2] += random_offset
                sat_image, R_sat2world, t_sat2world = self.sat_align(image,
                                                                     K_sat,
                                                                     R_sat2world,
                                                                     t_sat2world,
                                                                     random_rotate,
                                                                     t_sat2world_aug)
                sat_image = self.aug(sat_image)
            else:
                sat_image = image

            # # FG2
            # t_left2world_align = t_left2world_init
            # direc = torch.tensor([[0,0,0],[0,0,1]],dtype=torch.float).reshape(1,2,3)
            # direc = cam2world(direc, R_left2world[None], t_left2world[None])
            # direc = world2cam(direc, R_sat2world[None], t_sat2world[None])
            # direc = direc[0,1] - direc[0,0]
            # rotate = torch.arctan2(direc[0], -direc[1]).item()

            sat_image = self.sat_transform(sat_image)
        

        left_depth = pfm2tensor(left_depth_path, size=self.left_image_size)

        pair['left_image'] = left_image
        pair['sat_image'] = sat_image
        pair['left_depth'] = left_depth
        
        pair['R_sat2world'] = R_sat2world
        pair['t_sat2world'] = t_sat2world
        pair['t_left2world_init'] = t_left2world_init

        pair['K_left'] = pair['K_left'] * left_image_scale
        pair['K_left'][..., -1, -1] = 1.0
        pair['K_sat'] = pair['K_sat'] * sat_image_scale
        pair['K_sat'][..., -1, -1] = 1.0

        pair['max_init_offset'] = torch.tensor(self.max_init_offset)

        return pair

    def collate_fn(self, pairs):
        
        collated_pairs = {}

        for key in pairs[0].keys():
            collated_data = [pair[key] for pair in pairs]
            if isinstance(collated_data[0], torch.Tensor):
                collated_pairs[key] = torch.stack(collated_data, dim=0)
            else:
                collated_pairs[key] = collated_data

        data = {}
        label_keys = ['t_left2world']
        data['input'] = {k: collated_pairs[k] for k in collated_pairs if k not in label_keys}
        data['label'] = {k: collated_pairs[k] for k in collated_pairs if k in label_keys}
        
        return data
        

def read_data(dataset_name):
    dataset_dir = os.path.join('/data/cm_data/', dataset_name)
    keys = ['train', 'val', 'test']
    data_dict = {}
    for key in keys:
        data_file = os.path.join(dataset_dir, f'{key}_data.pth')
        data = torch.load(data_file, weights_only=False, map_location='cpu')

        # update image paths
        for i in range(len(data)):
            data[i]['left_image_path'] = os.path.join(dataset_dir, data[i]['left_image_path'])
            data[i]['sat_image_path'] = os.path.join(dataset_dir, data[i]['sat_image_path'])
            data[i]['left_depth_path'] = os.path.join(dataset_dir, data[i]['left_depth_path'])

        data_dict[key] = data

    return data_dict
