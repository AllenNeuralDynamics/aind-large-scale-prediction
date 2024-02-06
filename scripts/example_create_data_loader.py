"""
This is an example script in how
a dataloader could be created
"""

import logging
import multiprocessing
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import get_suggested_cpu_count


def create_logger(output_log_path: str) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/large_scale.log"  # _{CURR_DATE_TIME}

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def main():
    """
    Main function
    """
    device = None

    pin_memory = True
    if device is not None:
        pin_memory = False
        multiprocessing.set_start_method("spawn", force=True)

    dataset_path = "s3://aind-open-data/SmartSPIM_709392_2024-01-29_18-33-39_stitched_2024-02-04_12-45-58/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr"
    # "s3://aind-open-data/HCR_681417-Easy-GFP_2023-11-10_13-45-01_fused_2024-01-09_13-16-14/channel_561.zarr"
    #

    multiscale = "3"
    target_size_mb = 512
    n_workers = 10
    batch_size = 1
    prediction_chunksize = (128, 128, 128)
    super_chunksize = None
    logger = create_logger(output_log_path=".")

    suggested_cpus = get_suggested_cpu_count()
    logger.info(
        f"Suggested number of CPUs: {suggested_cpus} - Provided n count workers {n_workers}"
    )

    if n_workers > suggested_cpus:
        n_workers = suggested_cpus
        logger.info(
            f"Changing the # Workers to the suggested number of CPUs: {suggested_cpus}"
        )

    start_time = time.time()
    zarr_data_loader, zarr_dataset = create_data_loader(
        dataset_path=dataset_path,
        multiscale=multiscale,
        target_size_mb=target_size_mb,
        prediction_chunksize=prediction_chunksize,
        n_workers=n_workers,
        batch_size=batch_size,
        dtype=np.float32,  # Allowed data type to process with pytorch cuda
        super_chunksize=super_chunksize,
        lazy_callback_fn=None,  # partial_lazy_deskewing,
        logger=logger,
        device=device,
        pin_memory=pin_memory,
        drop_last=False,
        override_suggested_cpus=False,
    )
    end_time = time.time()

    logger.info(f"Time creating data loader: {end_time - start_time}")

    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(
        f"Number of batches: {total_batches} - Samples per iteration: {samples_per_iter}"
    )
    logger.info(f"Defined super chunk size: {zarr_dataset.super_chunksize}")

    start_time = time.time()

    # for idx in range(len(zarr_dataset.super_chunk_slices)):
    #     logger.info(
    #         f"Super chunk slices: {zarr_dataset.super_chunk_slices[idx]}"
    #     )
    #     logger.info(
    #         f"Internal slices in super chunk: {zarr_dataset.internal_slices[idx]}"
    #     )

    for i, sample in enumerate(zarr_data_loader):
        logger.info(
            f"Batch {i}: {sample.batch_tensor.shape} - Pinned?: {sample.batch_tensor.is_pinned()} - dtype: {sample.batch_tensor.dtype} - device: {sample.batch_tensor.device}"
        )

        logger.info(
            f"Super chunk: {sample.batch_super_chunk} - super chunk slice: {sample.batch_internal_slice}"
        )

        if i == 10:
            break

        numpy_arr = sample.batch_tensor[0, ...].numpy()
        logger.info(f"BLock shape: {numpy_arr.shape}")
        max_z_sample = np.max(numpy_arr, axis=0)
        vmin, vmax = np.percentile(max_z_sample, (0.1, 98))

        plt.imshow(max_z_sample, vmin=vmin, vmax=vmax)
        plt.show()

    end_time = time.time()

    logger.info(f"Time going through data loader: {end_time - start_time}")


if __name__ == "__main__":
    main()
