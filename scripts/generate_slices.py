from aind_large_scale_prediction.generator.zarr_slice_generator import (
    BlockedZarrArrayIterator,
)
from aind_large_scale_prediction.io import ImageReaderFactory
from aind_large_scale_prediction.io.utils import extract_data


def main():
    BUCKET_NAME = "aind-open-data"
    IMAGE_PATH = "diSPIM_685890_2023-06-29_14-39-56/diSPIM.zarr"
    TILE_NAME = "647_D1_X_0001_Y_0001_Z_0000_ch_488.zarr"

    dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    multiscale = "2"

    dataset_reader = ImageReaderFactory().create(
        data_path=dataset_path,
        parse_path=False,
        multiscale=multiscale,
    )

    print(f"Dataset read from: {dataset_reader.data_path}")
    dataset_lazy_data = extract_data(dataset_reader.as_dask_array())

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

    print(
        f"Chunksize to fit in memory {target_size_mb} MiB: {super_chunks_size}"
    )

    list_super_chunks = list(
        zarr_iterator.gen_slices(
            arr_shape=dataset_lazy_data.shape, block_shape=super_chunks_size
        )
    )
    prediction_chunksize = dataset_lazy_data.chunksize  # (64, 128, 128)

    generators = []
    for super_chunk in list_super_chunks:
        generators.append(
            zarr_iterator.gen_slices(
                arr_shape=dataset_lazy_data[super_chunk].shape,
                block_shape=prediction_chunksize,
            )
        )
        print(
            f"Shape for {super_chunk} is {dataset_lazy_data[super_chunk].shape}"
        )

    # for g in generators[0]:
    #     print(g)


if __name__ == "__main__":
    main()
