import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np

# Simulated large 3D dataset
class Mock3DDataset(Dataset):
    def __init__(self, size=100, shape=(64, 64, 64)):
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate a large 3D volume
        volume = np.random.rand(*self.shape).astype(np.float32)
        label = np.random.randint(0, 2)
        return torch.tensor(volume), torch.tensor(label)

# Setup process for distributed training
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# Main training/test loop
def run_ddp(rank, world_size):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    dataset = Mock3DDataset(size=100)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2)

    device = torch.device(f"cuda:{rank}")
    for batch_idx, (volumes, labels) in enumerate(dataloader):
        volumes = volumes.to(device)
        labels = labels.to(device)
        print(f"[Rank {rank}] Batch {batch_idx}: Volume shape: {volumes.shape}, Label shape: {labels.shape}")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")
    mp.spawn(run_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
