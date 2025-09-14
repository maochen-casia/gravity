import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.join(code_dir, 'dinov3-main')
WEIGHTS_PATH = '/data/zxk_data/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
WEIGHTS_SAT_PATH = '/data/zxk_data/dinov3/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
if REPO_DIR not in sys.path:
    sys.path.append(REPO_DIR)

import torch
import torch.nn as nn

class DINOv3(nn.Module):

    def __init__(self, device, freeze=True, sat=False):

        super().__init__()

        mean = torch.tensor((0.430, 0.411, 0.296)) if sat else torch.tensor((0.485, 0.456, 0.406))
        std = torch.tensor((0.213, 0.156, 0.143)) if sat else torch.tensor((0.229, 0.224, 0.225))
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

        weight_path = WEIGHTS_SAT_PATH if sat else WEIGHTS_PATH
        self.pretrained = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights=weight_path)
        self.pretrained.eval()

        self.device = device
        self.to(device)

        self.freeze = freeze
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        #self.to(self.dtype)

        self.intermediate_layer_idx = [4, 11, 17, 23]
        self.embed_dim = 1024
        self.scale = 16

    def train(self, mode=True):
        if self.freeze:
            return
        super().train(mode)

    def pre_process(self, images):
        images = images.to(self.device)
        images = images.to(self.dtype)
        images = (images - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return images
    
    def get_intermediate_layers(self, images, return_class_token=True, reshape=False):
        images = self.pre_process(images)
        if self.freeze:
            with torch.no_grad(), torch.autocast('cuda', dtype=self.dtype):
                features =  self.pretrained.get_intermediate_layers(images, n=self.intermediate_layer_idx, 
                                                                    return_class_token=return_class_token,
                                                                    reshape=reshape)
        else:
            features = self.pretrained.get_intermediate_layers(images, n=self.intermediate_layer_idx, 
                                                               return_class_token=return_class_token,
                                                               reshape=reshape)
        features = [(f[0].float(), f[1].float()) for f in features]
        return features

    def forward(self, images):
        images = self.pre_process(images)

        if self.freeze:
            with torch.no_grad(), torch.autocast('cuda', dtype=self.dtype):
                features = self.pretrained.forward_features(images)['x_norm_patchtokens']
        else:
            features = self.pretrained.forward_features(images)['x_norm_patchtokens']
        # h = w = int(features.shape[1] ** 0.5)
        # features = features.permute(0,2,1).unflatten(2, (h, w)) # 16 for 224, 37 for 518
        return features.float()

