from .trainer import Trainer
from .evaluator import Evaluator
from .checkpoint import Checkpoint
from .logger import Logger
from .dataset import LocDataset, read_data
from torch.utils.data import DataLoader

def build_checkpoint_logger(config, need_logger=True):
    checkpoint = Checkpoint(config)
    logger = Logger(checkpoint.save_dir) if need_logger else None
    print('Checkpoint and logger built successfully.')
    return checkpoint, logger

def build_trainer_evaluator(config, model, data_loaders):

    trainer = Trainer(config.trainer,
                      model,
                      data_loaders['train'])
    
    val_evaluator = Evaluator(config.evaluator,
                              model,
                              data_loaders['val'])
    
    test_evaluator = Evaluator(config.evaluator,
                               model,
                               data_loaders['test'])
    
    print('Trainer and evaluators built successfully.')

    return trainer, val_evaluator, test_evaluator

def build_data_loaders(config, seed):

    dataset_name = config.dataset
    batch_size = config.batch_size
    left_image_size = config.left_image_size
    sat_image_size = config.sat_image_size
    aug = config.aug
    max_train_init_offset = config.max_train_init_offset
    max_aug_offset = config.max_aug_offset
    max_aug_rotate = config.max_aug_rotate
    max_test_init_offset = config.max_test_init_offset

    data_dict = read_data(dataset_name)

    train_dataset = LocDataset(data_dict['train'],
                               left_image_size,
                               sat_image_size,
                               aug=aug,
                               max_init_offset=max_train_init_offset,
                               max_aug_offset=max_aug_offset,
                               max_aug_rotate=max_aug_rotate,
                               seed=seed)
    val_dataset = LocDataset(data_dict['val'],
                             left_image_size,
                             sat_image_size,
                             aug=False,
                             max_init_offset=max_test_init_offset)
    test_dataset = LocDataset(data_dict['test'],
                              left_image_size,
                              sat_image_size,
                              aug=False,
                              max_init_offset=max_test_init_offset)

    num_workers = 16
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, 
                                   collate_fn=train_dataset.collate_fn, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, 
                                 collate_fn=val_dataset.collate_fn, pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers, 
                                  collate_fn=test_dataset.collate_fn, pin_memory=True)

    data_loaders = {'train': train_data_loader,
                    'val': val_data_loader,
                    'test': test_data_loader}
    
    print('Data loaders built successfully.')

    return data_loaders

    
