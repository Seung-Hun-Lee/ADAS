from __future__ import absolute_import
from __future__ import division
import torch
import numpy as np
import random
from config import cfg
from trainer import ADAS_Trainer


if __name__ == '__main__':
    # Enable CUDNN Benchmarking optimization
    #torch.backends.cudnn.benchmark = True
    random_seed = cfg.RANDOM_SEED  #304
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    model = ADAS_Trainer()
    model.train()
