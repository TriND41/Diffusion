import os

import torch
import torch.distributed as distributed
import pandas as pd
import logging

logger = logging.getLogger(__name__)

DDP_HOST = 'localhost'
DDP_PORT = '12355'
DDP_BACKEND = 'nccl'
DDP_METHOD = 'env://'

def sample_distributed_data(dataset: pd.DataFrame, rank: int, world_size: int = 1) -> pd.DataFrame:
    if world_size == 1 or world_size == 0:
        return dataset
    
    num_samples = len(dataset)
    num_splits = num_samples // world_size
    remain_part = num_samples % world_size

    if remain_part == 0:
        start = rank * num_splits
        dataset = dataset[start: start + num_splits]
    else:
        if rank < remain_part:
            bonus_step = 1
            start_translation = rank
        else:
            bonus_step = 0
            start_translation = 0
        start = rank * num_splits + start_translation
        dataset = dataset[start: start + num_splits + bonus_step]

    return dataset

def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = DDP_HOST
    os.environ['MASTER_PORT'] = DDP_PORT
    torch.cuda.set_device(rank)
    distributed.init_process_group(backend=DDP_BACKEND, init_method=DDP_METHOD, rank=rank, world_size=world_size)
    logger.info(f"Initialized Thread at {rank + 1}/{world_size}")

def cleanup() -> None:
    distributed.destroy_process_group()