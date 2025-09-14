# import json
# from utils.build import build_utils
# from utils.checkpoint import Checkpoint
# from utils.random import set_seed
# from utils.logger import Logger
# from models.build import build_model

import os, sys
code_dir = os.path.dirname(os.path.abspath(__file__))
if code_dir not in sys.path:
    sys.path.append(code_dir)

from omegaconf import OmegaConf
from utils.build_utils import build_checkpoint_logger, build_trainer_evaluator, build_data_loaders
from utils.random import set_seed
from models.build_model import build_model
from visualize.visualize import visualize_model_output
#torch.autograd.set_detect_anomaly(True)

def main():

    config_path = './configs/gravity_config.yaml'
    config = OmegaConf.load(config_path)
    
    checkpoint, logger = build_checkpoint_logger(config)
    config = checkpoint.config
    config.data.batch_size = 1  # for visualization
    print(config)

    data_loaders = build_data_loaders(config.data)

    set_seed(config.seed)

    model = build_model(config.model)

    trainer, val_evaluator, test_evaluator = build_trainer_evaluator(config, model, data_loaders)
    checkpoint.set_trainer(trainer)

    num_epochs = trainer.num_epochs
    start_epoch = checkpoint.start_epoch

    print('Start visualization.')
    for data in data_loaders['test']:
        
        visualize_model_output(model, data)

if __name__ == '__main__':
    main()

