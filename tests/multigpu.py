import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np

from utils import Mock3DDataset

def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    dataset = Mock3DDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2)

    seen_indices = []
    for batch_idx, (volumes, labels, indices) in enumerate(dataloader):
        volumes = volumes.to(device)
        labels = labels.to(device)
        seen_indices.extend(indices.tolist())
        print(f"[Rank {rank}] Batch {batch_idx} | Volume shape: {volumes.shape} | Indices: {indices}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print(f"Run this code as `$ torchrun --nproc-per-node=4 aind-large-scale-prediction/tests/multigpu.py`")
    main()
