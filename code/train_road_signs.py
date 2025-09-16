import argparse
import os
import torch
import warnings
warnings.filterwarnings('ignore')


from nanodet.util import mkdir, Logger, cfg, load_config
from nanodet.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    args = parser.parse_args()

    load_config(cfg, args.config)
    
    # Create save directory
    mkdir(cfg.save_dir)
    
    # Setup logger
    logger = Logger(cfg.save_dir, use_tensorboard=True)
    
    # Set random seed
    if args.seed is not None:
        logger.log('Set random seed to {}'.format(args.seed))
        torch.manual_seed(args.seed)
    
    # Initialize trainer
    trainer = Trainer(cfg, logger)
    
    # Start training
    trainer.run()

if __name__ == '__main__':
    main()