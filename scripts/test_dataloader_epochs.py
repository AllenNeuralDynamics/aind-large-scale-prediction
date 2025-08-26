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
import zarr
from torch.utils.data import DataLoader

from aind_large_scale_prediction.generator.dataset import create_data_loader
from aind_large_scale_prediction.generator.utils import (
    concatenate_lazy_data,
    estimate_output_volume,
    get_suggested_cpu_count,
    recover_global_position,
    unpad_global_coords,
)


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
            logging.FileHandler(LOGS_FILE, "w"),
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

    # dataset_path = "s3://aind-open-data/HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49/channel_405.zarr"
    # nuclear_channel = "s3://aind-open-data/HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49/channel_3.zarr"

    dataset_path = "s3://aind-msma-morphology-data/test_data/protein_project/s1_stack_dbl(1).zarr"
    # "s3://aind-open-data/SmartSPIM_709392_2024-01-29_18-33-39_stitched_2024-02-04_12-45-58/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr"
    # "s3://aind-open-data/HCR_681417-Easy-GFP_2023-11-10_13-45-01_fused_2024-01-09_13-16-14/channel_561.zarr"
    # exaspim_test = "s3://aind-open-data/exaSPIM_653158_2023-06-01_20-41-38_fusion_2023-06-12_11-58-05/fused.zarr"

    multiscale = "3"
    target_size_mb = 1024
    n_workers = 0
    batch_size = 1
    epochs = 4
    prediction_chunksize = (128, 128, 128)
    overlap_prediction_chunksize = (0, 0, 0)
    super_chunksize = (512, 512, 512)
    logger = create_logger(output_log_path=".")

    lazy_data = concatenate_lazy_data(
        dataset_paths=[dataset_path],
        multiscales=[multiscale],
        concat_axis=-4,
        zarr_version="3.0",
    )

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
        device=device,
        pin_memory=pin_memory,
        drop_last=False,
        override_suggested_cpus=False,
        locked_array=True,
    )
    end_time = time.time()

    logger.info(f"Array shape: {zarr_dataset.lazy_data.shape}")
    logger.info(f"Prediction chunksize: {prediction_chunksize}")

    output_volume_shape = estimate_output_volume(
        image_shape=zarr_dataset.lazy_data.shape,
        chunk_shape=prediction_chunksize,
        overlap_per_axis=overlap_prediction_chunksize,
    )

    logger.info(
        f"Initial shape: {zarr_dataset.lazy_data.shape} - Estimated output volume shape: {output_volume_shape}"
    )

    logger.info(f"Time creating data loader: {end_time - start_time}")

    total_batches = np.prod(zarr_dataset.lazy_data.shape) / (
        np.prod(zarr_dataset.prediction_chunksize) * batch_size
    )
    samples_per_iter = n_workers * batch_size
    logger.info(
        f"Number of batches: {total_batches} - Samples per iteration: {samples_per_iter}"
    )
    logger.info(f"Defined super chunk size: {zarr_dataset.super_chunksize}")
    logger.info(f"Super chunk slices: {zarr_dataset.super_chunk_slices}")

    start_time = time.time()
    samples_per_epoch = []
    for epoch in range(epochs):
        zarr_data_loader.dataset.init_index()

        for i, sample in enumerate(zarr_data_loader):

            shape = [batch_size]
            for ix in range(0, len(prediction_chunksize)):
                shape.append(
                    sample.batch_internal_slice[0][ix].stop
                    - sample.batch_internal_slice[0][ix].start
                )

            if sum(shape) != sum(sample.batch_tensor.shape):
                raise ValueError(
                    f"Loaded tensor shape {sample.batch_tensor.shape} is not in the same location as slice shape: {shape} - slice: {sample.batch_internal_slice}"
                )

            (
                global_coord_pos,
                global_coord_positions_start,
                global_coord_positions_end,
            ) = recover_global_position(
                super_chunk_slice=sample.batch_super_chunk[0],
                internal_slices=sample.batch_internal_slice,
            )
            logger.info(
                f"Batch {i}: {sample.batch_tensor.shape} Super chunk: {sample.batch_super_chunk} - intern slice: {sample.batch_internal_slice} - global intern slice: {sample.batch_internal_slice_global}- global pos: {global_coord_pos}"
            )

            data_block = sample.batch_tensor[0, ...].numpy()
            unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
                global_coord_pos=global_coord_pos,
                block_shape=data_block.shape,
                overlap_prediction_chunksize=overlap_prediction_chunksize,
                dataset_shape=zarr_dataset.lazy_data.shape,
            )

            non_overlap_area = data_block[unpadded_local_slice]

            logger.info(
                f"Block shape: {data_block.shape} - nonoverlap area: {non_overlap_area.shape}"
            )

        samples_per_epoch.append(i)
        print("Final I: ", i)

    print("Samples per epoch ", samples_per_epoch)
    end_time = time.time()

    logger.info(f"Time going through data loader: {end_time - start_time}")


if __name__ == "__main__":
    main()
