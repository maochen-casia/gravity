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
#torch.autograd.set_detect_anomaly(True)

def main():

    config_path = './configs/gravity_config.yaml'
    config = OmegaConf.load(config_path)
    
    checkpoint, logger = build_checkpoint_logger(config)
    config = checkpoint.config
    print(config)

    data_loaders = build_data_loaders(config.data, config.seed)

    set_seed(config.seed)

    model = build_model(config.model)
    #model = compile_and_warmup_model(model, config.data.batch_size)

    trainer, val_evaluator, test_evaluator = build_trainer_evaluator(config, model, data_loaders)
    checkpoint.set_trainer(trainer)

    num_epochs = trainer.num_epochs
    start_epoch = checkpoint.start_epoch

    print('Start training.')
    for epoch in range(start_epoch, num_epochs+1):

        # train
        trainer.train()

        epoch_info = f'[Epoch {epoch}/{num_epochs}]'

        # val
        val_metrics, val_info = val_evaluator.evaluate()
        logger.info(epoch_info + ' Validation ' + val_info)
        checkpoint.step(val_metrics['loss'])

        # test
        test_metrics, test_info = test_evaluator.evaluate()
        logger.info(epoch_info + ' Test ' + test_info)

    model.load_state_dict(checkpoint.best_val_param)
    test_metrics, test_info = test_evaluator.evaluate()
    logger.info(test_info)

if __name__ == '__main__':
    main()

