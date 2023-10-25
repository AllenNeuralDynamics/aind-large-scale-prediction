"""
Defines the PyTorch Datasets classes
to load the models
"""

import ctypes
import multiprocessing
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from aind_large_scale_prediction._shared.types import ArrayLike
from aind_large_scale_prediction.generator.utils import getsizeof
from aind_large_scale_prediction.generator.zarr_slice_generator import (
    BlockedZarrArrayIterator,
)
from aind_large_scale_prediction.io.utils import extract_data


class ZarrSuperChunks(Dataset):
    """
    Defines a Dataset based on a Zarr image
    that will retrieve super chunks to memory
    that then will be used in normal chunks
    to predict with the model
    """

    def __init__(
        self,
        lazy_data: ArrayLike,
        prediction_chunk_size: Tuple[int, int, int],
        super_chunk_size: Optional[Tuple[int, int, int]] = None,
        target_size_mb: Optional[int] = None,
    ) -> None:
        """
        Initializes the dataset class

        Parameters
        ----------
        lazy_data: ArrayLike
            Lazy array that contains the data to process

        prediction_chunk_size: Tuple[int, int, int]
            Prediction chunk size. Given a zarr array
            and an estimated/provided super chunk size,
            we will take a prediction chunk size from
            the super chunk that is already in memory.

        super_chunk_size: Tuple[int, int, int]
            Given a lazy array (not loaded in memory),
            this parameter determines how many chunks
            will be moved to memory once at a time.
            This parameter is calculated if target_size_mb
            is provided.

        target_size_mb: Optional[int]
            Target size of the super chunks in memory.
            This parameter needs to be provided in
        """
        super(ZarrSuperChunks, self).__init__()

        if super_chunk_size is None and target_size_mb is None:
            raise ValueError(
                "Please, provide the super chunk size or target_size_mb parameters"
            )

        self.lazy_data = extract_data(lazy_data)
        self.super_chunk_size = super_chunk_size
        self.prediction_chunk_size = prediction_chunk_size
        self.target_size_mb = target_size_mb
        self.curr_super_chunk_pos = 0
        (
            self.super_chunk_size,
            self.super_chunk_slices,
            self.internal_slices,
            self.zarr_iterator,
        ) = self.__init_super_chunks_iter()

        # Control variables
        self.super_chunk_slices_idx = 0
        self.use_cache = False

        # Initializing shared array
        self.super_chunk_in_memory = self.__init_shared_array(
            shape=self.super_chunk_size
        )

        self.internal_slice_sum = tuple(
            len(internal_slice) for internal_slice in self.internal_slices
        )

    def __init_shared_array(self, shape: Tuple[int]):
        """
        Initializes a shared memory array where
        the super chunks will live one at a time.

        Parameters
        ----------
        shape: Tuple[int]
            Shape of the shared memory array that
            all workers will access

        Returns
        -------
        torch.Tensor
            Tensor pointing to a shared memory
            space
        """
        shape = int(np.prod(shape, axis=0))
        print(shape)
        shared_array_base = multiprocessing.Array(
            typecode_or_type=ctypes.c_ushort,  # ctypes.c_short, #ctypes.c_float
            size_or_initializer=shape,
            lock=True,
        )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        # shared_array = torch.from_numpy(shared_array)

        print(
            "Size of shared array in bytes: ",
            getsizeof(shared_array),
            shared_array.dtype,
        )

        return shared_array

    def __init_super_chunks_iter(self):
        chunk_size_megabytes = (self.lazy_data.blocks[0, 0, 0].nbytes) / (
            1024 * 1024
        )

        if (
            self.target_size_mb is not None
            and chunk_size_megabytes > self.target_size_mb
        ):
            raise ValueError(
                f"Please, check your chunk size ({chunk_size_megabytes}) and target size ({self.target_size_mb})."
            )

        # Getting super chunks that will be sent to GPU for prediction
        zarr_iterator = BlockedZarrArrayIterator()

        # Overwriting super chunk size if target size in mb is provided
        new_super_chunk_size = self.super_chunk_size
        if self.target_size_mb:
            new_super_chunk_size = zarr_iterator.get_block_shape(
                arr=self.lazy_data,
                target_size_mb=self.target_size_mb,
                mode="cycle",
            )

            print(
                f"Chunksize to fit in memory {self.target_size_mb} MiB: {new_super_chunk_size}"
            )

        super_chunk_slices = tuple(
            zarr_iterator.gen_slices(
                arr_shape=self.lazy_data.shape,
                block_shape=new_super_chunk_size,
            )
        )

        internal_slices = tuple(
            tuple(
                zarr_iterator.gen_slices(
                    arr_shape=self.lazy_data[super_chunk_slice].shape,
                    block_shape=self.prediction_chunk_size,
                )
            )
            for super_chunk_slice in super_chunk_slices
        )

        return (
            new_super_chunk_size,
            super_chunk_slices,
            internal_slices,
            zarr_iterator,
        )

    def __map_index(self, index) -> int:
        """
        Maps the current worker index to the corresponding
        internal slice for the corresponding super chunk.
        The internal slice is Tuple[ Tuple[ List[slice] ] ]
        which contains the corresponding prediction chunk
        position for a super chunk.

        Parameters
        ----------
        index: int
            Current index limited by the Dataset.__len__
            method

        Returns
        -------
        int
            Integer with the current position to access
            the current slice of a super chunk
        """
        curr_sum = sum(
            internal_slice_count
            for internal_slice_count in self.internal_slice_sum[
                : self.curr_super_chunk_pos
            ]
        )

        return index - curr_sum

    def __check_super_chunk_position(index) -> int:
        # TODO increment current super chunk position
        # based on the current index
        pass

    def __getitem__(self, index):
        if not self.use_cache:
            # Load data into shared memory space
            print(f"Filling super chunk in index: {index}")
            lazy_super_chunk = self.lazy_data[
                self.self.super_chunk_slices[self.curr_super_chunk_pos]
            ]
            self.super_chunk_in_memory = lazy_super_chunk.compute()
            print(
                f"Pulled super chunk of size {lazy_super_chunk.shape} into shared array {self.super_chunk_in_memory.shape}"
            )

        # Increments the shuper chunk
        self.curr_super_chunk_pos = self.__check_super_chunk_position(index)

        curr_internal_super_chunk_slice = self.__map_index(index)

        return self.super_chunk_in_memory[
            self.internal_slices[self.curr_super_chunk_pos][
                curr_internal_super_chunk_slice
            ]
        ]

    # np.zeros((1), dtype=np.uint8)

    def __len__(self):
        return sum(internal_slice_sum)


def main():
    from torch.utils.data import DataLoader

    from aind_large_scale_prediction.io import ImageReaderFactory

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

    print(
        f"Read dataset: {dataset_reader.data_path} - dtype: {dataset_reader.as_dask_array().dtype} - Shape: {dataset_reader.shape} - Chunks: {dataset_reader.chunks}"
    )

    zarr_dataset = ZarrSuperChunks(
        lazy_data=dataset_reader.as_dask_array(),
        prediction_chunk_size=(64, 128, 128),
        super_chunk_size=None,
        target_size_mb=512,
    )

    dataloader = DataLoader(
        zarr_dataset, batch_size=64, shuffle=False, num_workers=10
    )

    for i, sample in enumerate(dataloader):
        print(i, sample.shape)


if __name__ == "__main__":
    main()
