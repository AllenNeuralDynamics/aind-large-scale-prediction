import os
import logging
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data,
    estimate_output_volume,
    get_suggested_cpu_count,
    recover_global_position,
    unpad_global_coords,
)

from utils import Mock3DDataset


# TODO: change Mock3DDataset() to intantiate a zarr_data_loader

def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def setup_logger():
    logger = logging.getLogger("my_logger")  # You can name it anything
    logger.setLevel(logging.INFO)  # Set the log level

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and attach to handler
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(ch)

    # Example usage
    logger.info("Logger initialized")
    return logger

def main():
    DATASET_PATH = "s3://aind-open-data/SmartSPIM_709392_2024-01-29_18-33-39_stitched_2024-02-04_12-45-58/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr"

    multiscale = "3"
    target_size_mb = 1024
    n_workers = 1
    batch_size = 8
    prediction_chunksize = (128, 128, 128)
    overlap_prediction_chunksize = (30, 30, 30)
    super_chunksize = (512, 512, 512)

    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    logger = setup_logger()

    # dataset = Mock3DDataset()

    lazy_data = concatenate_lazy_data(
        dataset_paths=[DATASET_PATH],
        multiscales=[multiscale],
        concat_axis=-4,
    )

    _, dataset = create_data_loader(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=None,
        pin_memory=False,
        drop_last=False,
        override_suggested_cpus=True,
        locked_array=True,
    )

    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
        )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=n_workers
        )
    
    seen_indices = []
    for batch_idx, (volumes, labels, indices) in enumerate(dataloader):
        volumes = volumes.to(device)
        labels = labels.to(device)
        seen_indices.extend(indices.tolist())
        print(f"[Rank {rank}] Batch {batch_idx} | Volume shape: {volumes.shape} | Label shape: {labels.shape} | Indices: {indices}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print(f"Run this code as `$ torchrun --nproc-per-node=4 aind-large-scale-prediction/tests/multigpu_aind.py`")
    
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    main()
