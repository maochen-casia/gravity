import random
import numpy as np
import torch

def random_from_range(range: tuple) -> float:
    """
        Randomly sample a number from range.
        Args:
            range: tuple, representing (min, max).
        Returns:
            sampled number.
    """
    min_, max_ = range
    r = random.random() * (max_ - min_) + min_
    return r

def set_seed(seed, deterministic=True):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
