import torch
import numpy as np
import os
from omegaconf import OmegaConf
from data.build_data_loaders import build_data_loaders
from utils.build_utils import build_checkpoint_logger, build_trainer_evaluator
from utils.random import set_seed
from models.build_model import build_model, compile_and_warmup_model
from visualize.utils import visualize_tensors_with_arrow
from models.S3Loc.transform import cam2world, world2image

def get_arrow(K_left, R_left2world, t_left2world,
              K_sat, R_sat2world, t_sat2world):
    
    left_point_cam = torch.tensor([0, 0, 0], dtype=torch.float32).reshape(1, 3)
    left_point_world = cam2world(left_point_cam, R_left2world, t_left2world)
    left_point_image = world2image(left_point_world, K_sat, R_sat2world, t_sat2world)

    end_point_cam = torch.tensor([0, 0, 10], dtype=torch.float32).reshape(1, 3)
    end_point_world = cam2world(end_point_cam, R_left2world, t_left2world)
    end_point_image = world2image(end_point_world, K_sat, R_sat2world, t_sat2world)

    start_point = left_point_image.squeeze().numpy().astype(np.int32)
    end_point = end_point_image.squeeze().numpy().astype(np.int32)

    return start_point, end_point

def main():

    config_path = './configs/lunarqformer_config.yaml'
    config = OmegaConf.load(config_path)
    
    checkpoint, logger = build_checkpoint_logger(config)
    config = checkpoint.config
    config.data.batch_size = 1  # For visualization, use batch size of 1

    set_seed(config.seed)

    data_loaders = build_data_loaders(config)
    train_dataloader = data_loaders['train']

    for i, data in enumerate(train_dataloader):

        if i % 10 != 0:
            continue

        input_data = data['input']
        left_image = input_data['left_image']
        sat_image = input_data['sat_image']
        K_left = input_data['K_left']
        R_left2world = input_data['R_left2world']
        t_left2world = input_data['t_left2world']
        K_sat = input_data['K_sat']
        R_sat2world = input_data['R_sat2world']
        t_sat2world = input_data['t_sat2world']

        left_point, end_point = get_arrow(K_left, R_left2world, t_left2world,
                                          K_sat, R_sat2world, t_sat2world)

        left_image_path = input_data['left_image_path'][0]

        basename = os.path.basename(left_image_path)
        save_path = os.path.join(f'./figures/{basename}')
        visualize_tensors_with_arrow(left_image, sat_image, left_point, end_point,
                                    save_path=save_path)
        print(f'Saved visualization to {save_path}')
        

if __name__ == '__main__':
    main()

