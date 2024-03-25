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
from distributed import Client, LocalCluster, wait

from aind_large_scale_prediction.generator.dataset import (
    create_data_loader,
    create_overlapped_slices,
)
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

    dataset_path = "s3://aind-open-data/HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49/channel_405.zarr"
    nuclear_channel = "s3://aind-open-data/HCR_BL6-000_2023-06-1_00-00-00_fused_2024-02-09_13-28-49/channel_3.zarr"
    # "s3://aind-open-data/SmartSPIM_709392_2024-01-29_18-33-39_stitched_2024-02-04_12-45-58/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr"
    # "s3://aind-open-data/HCR_681417-Easy-GFP_2023-11-10_13-45-01_fused_2024-01-09_13-16-14/channel_561.zarr"

    start_time = time.time()
    multiscale = "2"
    target_size_mb = 4096
    n_workers = 10
    batch_size = 1
    prediction_chunksize = (2, 128, 128, 128)
    overlap_prediction_chunksize = (0, 30, 30, 30)
    super_chunksize = None  # (2, 512, 512, 512)
    logger = create_logger(output_log_path=".")

    lazy_data = concatenate_lazy_data(
        dataset_paths=[dataset_path, nuclear_channel],
        multiscale=multiscale,
        concat_axis=-4,
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

    da_workers = 10

    (
        lazy_data,
        super_chunksize,
        super_chunk_slices,
        local_slices,
        global_slices,
    ) = create_overlapped_slices(
        lazy_data=lazy_data,
        target_size_mb=target_size_mb,
        super_chunksize=super_chunksize,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
    )

    output_zarr_path = "./test_dataset.zarr"
    output_zarr = zarr.open(
        output_zarr_path,
        "w",
        shape=lazy_data.shape,  # output_volume_shape,
        chunks=tuple(prediction_chunksize),  # prediction_chunksize_overlap,
        dtype=np.uint16,
    )
    # print(lazy_data.shape, prediction_chunksize)
    # exit()

    flat_global_slices = [
        item for sublist in global_slices for item in sublist
    ]

    print(flat_global_slices, len(flat_global_slices))

    curr_slcs_len = len(flat_global_slices)

    blocks_per_worker = curr_slcs_len // da_workers

    if curr_slcs_len <= da_workers:
        blocks_per_worker = curr_slcs_len

    print("Blocks per worker: ", blocks_per_worker)

    start_slice = 0
    end_slice = blocks_per_worker

    # Dividing internal chunks
    divided_slcs = []

    for idx_worker in range(da_workers):
        divided_slcs.append(flat_global_slices[start_slice:end_slice])

        if idx_worker + 1 == da_workers - 1:
            start_slice = end_slice
            end_slice = curr_slcs_len
        else:
            start_slice = end_slice
            end_slice += blocks_per_worker

    def execute_worker(
        global_slices,
    ):
        # Retrieve the shared variable from the workers
        # shared_lazy_data = client.run(lambda: shared_lazy_data)
        # shared_output_zarr = client.run(lambda: shared_output_zarr)

        # idx_super_chunk, local_slices, global_slices = data_to_process

        for idx, glb_slc in enumerate(global_slices):

            chunked_data = lazy_data[glb_slc].compute()

            unpadded_global_slice, unpadded_local_slice = unpad_global_coords(
                global_coord_pos=global_slices[idx],
                block_shape=chunked_data.shape,
                overlap_prediction_chunksize=overlap_prediction_chunksize,
                dataset_shape=lazy_data.shape,
            )

            print(
                f"PID {os.getpid()}: global slice: {global_slices[idx]} - {chunked_data.shape} - Glboal slc: {unpadded_global_slice} - local_slc: {unpadded_local_slice}"
            )

            output_zarr[unpadded_global_slice] = chunked_data[
                unpadded_local_slice
            ]

    cluster = LocalCluster(
        n_workers=da_workers,
        threads_per_worker=1,
        processes=True,
        memory_limit="auto",
    )

    client = Client(cluster)

    # Submit jobs to sum the pairs and multiply by the shared variable on each worker
    futures = [
        client.submit(execute_worker, curr_slcs) for curr_slcs in divided_slcs
    ]

    # Optionally, wait for all futures to complete
    wait(futures)

    # Gather the results
    results = client.gather(futures)

    # Optionally, gather the divided values
    gathered_values = client.gather(futures)

    client.shutdown()

    end_time = time.time()

    print("Total time: ", end_time - start_time)


if __name__ == "__main__":
    main()
