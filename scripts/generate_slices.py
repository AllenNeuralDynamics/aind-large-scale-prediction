"""
Generates slices script example
"""

import numpy as np

from aind_large_scale_prediction.generator.dataset import (
    reshape_dataset_to_prediction_chunks,
)
from aind_large_scale_prediction.generator.zarr_slice_generator import (
    BlockedZarrArrayIterator,
    _closer_to_target_chunksize,
)
from aind_large_scale_prediction.io import ImageReaderFactory
from aind_large_scale_prediction.io.utils import extract_data


def main():
    """
    Main generate slice example
    """
    dataset_path = "s3://aind-open-data/SmartSPIM_709392_2024-01-29_18-33-39_stitched_2024-02-04_12-45-58/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr"
    multiscale = "3"
    prediction_chunksize = (128, 128, 128)

    dataset_reader = ImageReaderFactory().create(
        data_path=dataset_path,
        parse_path=False,
        multiscale=multiscale,
    )

    print(f"Dataset read from: {dataset_reader.data_path}")
    dataset_lazy_data = extract_data(dataset_reader.as_dask_array())

    overlap_prediction_chunksize = [30, 30, 30]

    overlapped_prediction_chunksize = np.array(prediction_chunksize) + (
        np.array(overlap_prediction_chunksize) * 2
    )

    dataset_lazy_data = reshape_dataset_to_prediction_chunks(
        lazy_data=dataset_lazy_data,
        prediction_chunksize=overlapped_prediction_chunksize,
    )

    print(f"Dataset shape: {dataset_lazy_data.shape}")
    print(f"Dataset chunksize: {dataset_lazy_data.chunksize}")

    target_size_mb = 512
    chunk_size_megabytes = (dataset_lazy_data.blocks[0, 0, 0].nbytes) / (
        1024 * 1024
    )

    if chunk_size_megabytes > target_size_mb:
        raise ValueError(
            f"Please, check your chunk size ({chunk_size_megabytes}) and target size ({target_size_mb})."
        )

    # Getting super chunks that will be sent to GPU for prediction
    zarr_iterator = BlockedZarrArrayIterator()

    super_chunks_size = zarr_iterator.get_block_shape(
        arr=dataset_lazy_data, target_size_mb=target_size_mb, mode="cycle"
    )

    new_super_chunksize = _closer_to_target_chunksize(
        super_chunksize=super_chunks_size,
        chunksize=prediction_chunksize,
    )

    print(
        f"Chunksize to fit in memory {target_size_mb} MiB: {super_chunks_size} - New super chunksize: {new_super_chunksize}"
    )

    list_super_chunks = list(
        zarr_iterator.gen_slices(
            arr_shape=dataset_lazy_data.shape,
            block_shape=new_super_chunksize,
            overlap_shape=overlap_prediction_chunksize,
        )
    )

    # Generating super chunks
    unpadded_super_chunks = []
    for idx, sc in enumerate(list_super_chunks):
        curr_sc = []
        for ax_pos, ax in enumerate(sc):

            start = 0
            stop = ax.stop - overlap_prediction_chunksize[ax_pos]
            if ax.start != 0:
                start = ax.start + overlap_prediction_chunksize[ax_pos]

            if ax.stop == dataset_lazy_data.shape[ax_pos]:
                stop = ax.stop

            curr_sc.append(slice(start, stop))

        unpadded_super_chunks.append(tuple(curr_sc))
        local_test_slices, global_test_slices = (
            zarr_iterator.gen_over_slices_on_over_superchunks(
                arr_shape=dataset_lazy_data[sc].shape,
                block_shape=prediction_chunksize,
                overlap_shape=overlap_prediction_chunksize,
                super_chunk_slices=sc,
                dataset_shape=dataset_lazy_data.shape,
            )
        )

        local_test_slices = list(local_test_slices)
        global_test_slices = list(global_test_slices)

        print(
            f"Super chunk {idx} is {sc} -> gen slices: {local_test_slices}\n"
        )  # -> unpadded: {curr_sc}")

    unpadded_super_chunks = tuple(unpadded_super_chunks)

    print(
        f"Prediction chunksize: {prediction_chunksize} - Overlap shape: {overlap_prediction_chunksize}"
    )

    # # Generating internal slices for workers
    # internal_slices = tuple(
    #     tuple(
    #         zarr_iterator.gen_slices(
    #             arr_shape=dataset_lazy_data[super_chunk_slice].shape,
    #             block_shape=prediction_chunksize,
    #             overlap_shape=overlap_prediction_chunksize, # Internal slices with overlap between them
    #         )
    #     )
    #     for super_chunk_slice in list_super_chunks#unpadded_super_chunks
    # )

    # for idx, sc in enumerate(internal_slices):
    #     curr_sc = []

    #     print(f"\nInternal chunk {idx} for {unpadded_super_chunks[idx]} is {sc}")

    exit()
    return (
        dataset_lazy_data,
        list_super_chunks,
        tuple(prediction_chunksize),
        zarr_iterator,
        tuple(overlap_shape),
    )

    # generators = []
    # for super_chunk in list_super_chunks:
    #     generators.append(
    #         zarr_iterator.gen_slices(
    #             arr_shape=dataset_lazy_data[super_chunk].shape,
    #             block_shape=prediction_chunksize,
    #         )
    #     )
    #     print(
    #         f"Shape for {super_chunk} is {dataset_lazy_data[super_chunk].shape} - {dataset_lazy_data[super_chunk].blocks.size}"
    #     )

    # for g in generators[0]:
    #     print(g)


def eval_loading_super_chunk(
    dataset_lazy_data,
    list_super_chunks,
    prediction_chunksize,
    zarr_iterator,
    overlap_shape,
):
    """
    Evaluates the performance of the super chunk loader
    """
    super_chunk = list_super_chunks[0]
    print(f"Total n blocks: {dataset_lazy_data[super_chunk].blocks.size}")
    data_in_memory = dataset_lazy_data[super_chunk].compute()
    for i, zarr_iter_slice in enumerate(
        zarr_iterator.gen_slices(
            arr_shape=data_in_memory.shape,
            block_shape=prediction_chunksize,
            overlap_shape=overlap_shape,
        )
    ):
        print(
            f"[{i}] - Shape for super chunk: {dataset_lazy_data[super_chunk].shape} - {data_in_memory.shape} slice of data in memory: {data_in_memory[zarr_iter_slice].shape} - slice: {zarr_iter_slice}"
        )


def eval_loading_chunks_directly(
    dataset_lazy_data, list_super_chunks, prediction_chunksize, zarr_iterator
):
    """
    Evaluating slicer
    """
    super_chunk = list_super_chunks[0]
    print(f"Total n blocks: {dataset_lazy_data[super_chunk].blocks.size}")

    for i, zarr_iter_slice in enumerate(
        zarr_iterator.gen_slices(
            arr_shape=dataset_lazy_data[super_chunk].shape,
            block_shape=prediction_chunksize,
        )
    ):
        chunk_in_memory = dataset_lazy_data[super_chunk][
            zarr_iter_slice
        ].compute()
        print(
            f"[{i}] - Shape for super chunk: {dataset_lazy_data[super_chunk].shape} - {dataset_lazy_data[super_chunk].shape} slice ofdata in memory: {chunk_in_memory.shape}"
        )


if __name__ == "__main__":
    import cProfile

    (
        dataset_lazy_data,
        list_super_chunks,
        prediction_chunksize,
        zarr_iterator,
        overlap_shape,
    ) = main()
    # cProfile.run(
    #     "eval_loading_super_chunk(dataset_lazy_data, list_super_chunks, prediction_chunksize, zarr_iterator, overlap_shape)",
    #     filename="super_chunk.dat",
    # )
    # cProfile.run('eval_loading_chunks_directly(dataset_lazy_data, list_super_chunks, prediction_chunksize, zarr_iterator)', filename="chunks_directly.dat")
