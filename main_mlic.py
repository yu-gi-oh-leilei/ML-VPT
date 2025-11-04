import math
import os, sys
import random
import time
import json

import _init_paths
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # specify your GPU ids here
from config import parser_args
from config_opt import get_config

def get_args():
    args = parser_args()
    return args

def main():
    args = get_args()
    args, config = get_config(args)
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(config.DDP.gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.DDP.gpus
    
    
    from utils.misc import init_distributed_and_seed
    from utils.util import show_args, init_logeger
    from main_worker import main_worker
    # init distributed and seed
    init_distributed_and_seed(config)
    
    # init logeger and show config
    logger = init_logeger(config)
    show_args(config, logger)

    return main_worker(args, config, logger)

if __name__ == '__main__':
    main()