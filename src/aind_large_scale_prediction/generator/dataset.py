"""
Defines the PyTorch Datasets classes
to load the models
"""

import multiprocessing
import time
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from aind_large_scale_prediction._shared.types import ArrayLike
from aind_large_scale_prediction.generator.dataloader import ZarrDataLoader
from aind_large_scale_prediction.generator.utils import (
    find_position_in_total_sum,
    getsizeof,
    map_dtype_to_ctype,
)
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
        locker=None,
        condition=None,
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

        # Lazy data after extraction. e.g., (1, 1, 1024, 512, 512) -> (1024, 512, 512)
        self.lazy_data = extract_data(lazy_data)
        self.super_chunk_size = super_chunk_size
        self.prediction_chunk_size = prediction_chunk_size
        self.target_size_mb = target_size_mb

        # Multiprocessing variables
        self.locker = locker
        self.condition = condition
        self.curr_super_chunk_pos = multiprocessing.Value("i", -1)

        # Initialization of super chunks
        (
            self.super_chunk_size,
            self.super_chunk_slices,
            self.internal_slices,
            self.zarr_iterator,
        ) = self.__init_super_chunks_iter()

        # Initializing shared array
        (
            self.super_chunk_in_memory,
            self.array_pointer,
        ) = self.__init_shared_array(
            shape=self.super_chunk_size, dtype=self.lazy_data.dtype
        )

        # Number of slices per super chunk
        self.internal_slice_sum = tuple(
            len(internal_slice) for internal_slice in self.internal_slices
        )
        self.pulled_chunks_per_super_chunk = multiprocessing.Array(
            "i", len(self.internal_slice_sum)
        )

    def __init_shared_array(
        self, shape: Tuple[int], dtype: type
    ) -> torch.Tensor:
        """
        Initializes a shared memory array where
        the super chunks will live one at a time.

        Parameters
        ----------
        shape: Tuple[int]
            Shape of the shared memory array that
            all workers will access

        dtype: type
            Array dtype.

        Returns
        -------
        torch.Tensor
            Tensor pointing to a shared memory
            space
        """
        shared_array_base = multiprocessing.Array(
            typecode_or_type=map_dtype_to_ctype(dtype=dtype, exact=False),
            size_or_initializer=int(np.prod(shape, axis=0)),
            lock=True,
        )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        shared_array = torch.from_numpy(shared_array)
        print(
            "Size of shared array in bytes: ",
            getsizeof(shared_array),
            shared_array.dtype,
        )

        return shared_array, shared_array_base

    def __init_super_chunks_iter(self) -> Tuple:
        """
        Initializes the super chunk slices.

        If target_size_mb is not None, then this method
        will try to estimate the best chunking size to
        fit the provided target_size_mb value in megabytes.
        If it's not provided, we will use the super chunk
        size as default.

        Returns
        -------
        Tuple[ Tuple, Tuple[Tuple], Tuple[Tuple], Generator]

            new_super_chunk_size [Tuple]: New super chunk size that was
            estimated if target_size_mb was provided.

            super_chunk_slices [ Tuple[Tuple] ]: Generated super chunk
            slices using the entire volume in order to partition the array.

            internal_slices [ Tuple[Tuple] ]: Generated slices per super
            chunk using the prediction chunking.

            zarr_iterator [ Generator ]: Generator of slices per dimension.
        """

        if self.target_size_mb is None and self.super_chunk_size is None:
            raise ValueError(
                "Please, provide a target size or super chunk size."
            )

        chunk_size_megabytes = (self.lazy_data.blocks[0, 0, 0].nbytes) / (
            1024 * 1024
        )

        # Validating that target size is actually smaller
        # than the total chunking size
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

        # Generating super chunk slices
        super_chunk_slices = tuple(
            zarr_iterator.gen_slices(
                arr_shape=self.lazy_data.shape,
                block_shape=new_super_chunk_size,
            )
        )

        # Generating internal slices per generated super chunk
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

    def __map_index(self, index, next_batch: Optional[bool] = False) -> int:
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

        next_batch: Optional[bool]
            If True, we compute the sum using the number
            of slices in the next super chunk position.
            Otherwise, only computes the sum of the slices
            until the current super chunk position.

        Returns
        -------
        int
            Integer with the current position to access
            the current slice of a super chunk
        """
        add_pos = 1 if next_batch else 0
        curr_sum = sum(
            internal_slice_count
            for internal_slice_count in self.internal_slice_sum[
                : self.curr_super_chunk_pos.value + add_pos
            ]
        )

        return index - curr_sum

    def __check_super_chunk_position(self, index) -> int:
        """
        Increments the super chunk position as corresponds.

        Parameters
        ----------
        index: int
            Current dataset index shared among workers

        Returns
        -------
        int:
            Integer with the new super chunk position
        """
        pos_in_super_chunk = find_position_in_total_sum(
            index, self.internal_slice_sum
        )
        return pos_in_super_chunk

    def __check_pulled_chunks(self) -> bool:
        """
        Checks the number of pulled chunks for a super chunk.

        Returns
        -------
        bool
            True, if all chunks were retrieved for that super
            chunk, False otherwise.
        """
        if self.curr_super_chunk_pos.value == -1:
            with self.pulled_chunks_per_super_chunk.get_lock():
                for i in range(len(self.internal_slice_sum)):
                    self.pulled_chunks_per_super_chunk[i] = 0

            return True

        number_pulled_chunks = self.pulled_chunks_per_super_chunk[
            self.curr_super_chunk_pos.value
        ]
        total_chunks_super_chunk = self.internal_slice_sum[
            self.curr_super_chunk_pos.value
        ]

        if number_pulled_chunks == total_chunks_super_chunk:
            return True

        return False

    def __parallel_get_item(self, index: int) -> torch.Tensor:
        """
        Method to retrieve the current chunk in the
        current super chunk.

        Parameters
        ----------
        index: int
            Current index in the worker controlled by
            __len__

        Returns
        -------
        torch.Tensor
            Tensor with the current chunk of the
            super chunk
        """

        self.locker.acquire()
        try:
            if self.__check_pulled_chunks():
                with self.curr_super_chunk_pos.get_lock():
                    self.curr_super_chunk_pos.value += 1

                # Setting super chunk in memory
                self.super_chunk_in_memory = self.lazy_data[
                    self.super_chunk_slices[self.curr_super_chunk_pos.value]
                ].compute()

                with self.condition:
                    self.condition.notify_all()

        except BaseException as e:
            raise BaseException(f"Error pulling data: {e}")

        finally:
            # Releasing lock
            self.locker.release()

        next_super_chunk_position = self.__check_super_chunk_position(index)

        if self.curr_super_chunk_pos.value != next_super_chunk_position:
            # If workers are ahead of current super chunk position, wait
            with self.condition:
                while (
                    self.curr_super_chunk_pos.value
                    != next_super_chunk_position
                ):
                    self.condition.wait()

        # Getting internal slice position
        curr_internal_super_chunk_position = self.__map_index(index)

        if (
            self.internal_slice_sum[self.curr_super_chunk_pos.value]
            < curr_internal_super_chunk_position
        ):
            raise RuntimeError(
                f"Worker got an out of bounds index: {curr_internal_super_chunk_position}"
            )

        # Pulling chunk
        pulled_chunk = self.super_chunk_in_memory[
            self.internal_slices[self.curr_super_chunk_pos.value][
                curr_internal_super_chunk_position
            ]
        ]

        # Updating number of chunks pulled per super chunk
        with self.pulled_chunks_per_super_chunk.get_lock():
            self.pulled_chunks_per_super_chunk[
                self.curr_super_chunk_pos.value
            ] += 1

        return pulled_chunk

    def __getitem__(self, index: int) -> np.array:
        """
        Get item procedure for the Lazy zarr dataset.
        It retrieves data based on the index variable
        that is shared among all workers

        Parameters
        ----------
        index: int
            location index that goes from 0 to
            ZarrSuperChunks().__len__(). This index
            is internally mapped to correspond in
            the generated slices per super chunk

        Returns
        -------
        torch.Tensor
            Tensor with size (batch_size, slice_size)
            where slice_size depends on your data
        """
        pulled_prediction_chunk = None

        worker_info = get_worker_info()

        if worker_info is None:
            # Main workers - No necessity of parallelism
            if (
                self.__check_super_chunk_position(index)
                != self.curr_super_chunk_pos.value
                or self.curr_super_chunk_pos.value == -1
            ):
                self.curr_super_chunk_pos.value += 1

                self.super_chunk_in_memory = self.lazy_data[
                    self.super_chunk_slices[self.curr_super_chunk_pos.value]
                ].compute()

            curr_internal_super_chunk_position = self.__map_index(index)

            if (
                self.internal_slice_sum[self.curr_super_chunk_pos.value]
                < curr_internal_super_chunk_position
            ):
                raise RuntimeError(
                    f"Worker got an out of bounds index: {curr_internal_super_chunk_position}"
                )

            # Pulling chunk
            pulled_prediction_chunk = self.super_chunk_in_memory[
                self.internal_slices[self.curr_super_chunk_pos.value][
                    curr_internal_super_chunk_position
                ]
            ]

        else:
            pulled_prediction_chunk = self.__parallel_get_item(index)

        return pulled_prediction_chunk

    def __len__(self):
        """
        Customized len method. Returns the sum
        of the slices per super chunk
        """
        return sum(self.internal_slice_sum)

    def __del__(self):
        """
        Releasing memory pointing variables to NULL
        """
        self.curr_super_chunk_pos = None
        self.array_pointer = None
        self.super_chunk_in_memory = None


def main():
    """
    Main function
    """
    import dask.array as da

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
    lazy_data = dataset_reader.as_dask_array().astype(np.int32)
    data_path = dataset_reader.data_path

    # data_path = dataset_reader.data_path
    # lazy_data = da.zeros(
    #     (1, 1, 5981, 512, 512), chunks=(1, 1, 128, 256, 256), dtype=np.int32
    # )

    print(
        f"Read dataset: {data_path} - dtype: {lazy_data.dtype} - Shape: {lazy_data.shape} - Chunksize: {lazy_data.chunksize}"
    )

    print(f"Array shape: {lazy_data.shape}")
    results = {}
    prediction_chunk_size = (64, 64, 64)
    target_size_mb = 512
    for n_workers in range(0, 10):
        locker = multiprocessing.Lock() if n_workers else None
        condition = multiprocessing.Condition()

        for batch_size in [8, 16]:
            print(
                f"{20*'='} Test Workers {n_workers} Batch Size {batch_size} {20*'='}"
            )
            start_time = time.time()
            zarr_dataset = ZarrSuperChunks(
                lazy_data=lazy_data,
                prediction_chunk_size=prediction_chunk_size,
                super_chunk_size=None,
                target_size_mb=target_size_mb,
                locker=locker,
                condition=condition,
            )

            zarr_data_loader = ZarrDataLoader(
                zarr_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
            )

            print("Starting the loading...")

            for i, sample in enumerate(zarr_data_loader):
                pass

            end_time = time.time()
            results[f"Workers {n_workers} - batch_size {batch_size}"] = (
                end_time - start_time
            )
            print(
                f"Time executing with {n_workers} workers and batch size {batch_size}: {end_time - start_time} seconds"
            )

    print(results)
    locker = None
    barrier = None
    condition = None


if __name__ == "__main__":
    main()
    # import cProfile
    # cProfile.run('main()', filename="compute_costs.dat")
