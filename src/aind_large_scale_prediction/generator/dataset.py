"""
Defines the PyTorch Datasets classes
to load the models
"""

import logging
import time
from functools import partial
from typing import Callable, List, Optional, Tuple

import dask.array as da
import numpy as np
import torch
import torch.multiprocessing as multp
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info

from aind_large_scale_prediction._shared.types import ArrayLike
from aind_large_scale_prediction.generator.dataloader import ZarrDataLoader
from aind_large_scale_prediction.generator.utils import (
    find_position_in_total_sum,
    get_suggested_cpu_count,
    map_dtype_to_ctype,
)
from aind_large_scale_prediction.generator.zarr_slice_generator import (
    BlockedZarrArrayIterator,
    _closer_to_target_chunksize,
)
from aind_large_scale_prediction.io.utils import extract_data


def reshape_dataset_to_prediction_chunks(
    lazy_data: ArrayLike, prediction_chunksize: Tuple[int, ...]
) -> ArrayLike:
    """
    Reshape dataset to have multiples of
    the prediction chunksize

    Parameters
    ----------
    lazy_data: ArrayLike
        Lazy dataset in dask

    prediction_chunksize: Tuple[int, ...]
        Prediction chunksize we are going to use
        to pull chunks from the super chunk

    Returns
    -------
    ArrayLike
        Modified array with the new shape
    """

    new_factor_shape = tuple(
        int(
            prediction_chunksize[idx]
            * np.ceil(lazy_data.shape[idx] / prediction_chunksize[idx])
        )
        for idx in range(len(prediction_chunksize))
    )

    pad_width = tuple(
        (
            0,
            int(
                new_factor_shape[idx] - lazy_data.shape[idx]
            ),  # padding the right and bottom of the volume only
        )
        for idx in range(len(new_factor_shape))
    )

    new_lazy_data = da.pad(
        array=lazy_data,
        pad_width=pad_width,
        mode="constant",
        constant_values=0,
    )

    return new_lazy_data


class ZarrCustomBatch:
    """
    Custom zarr batch object to manage pin memory
    """

    def __init__(
        self,
        batch_tensor: List[torch.Tensor],
        batch_super_chunk: List[Tuple[int]],
        batch_internal_slice: List[Tuple[int]],
        batch_internal_slice_global: List[Tuple[int]],
        device: Optional[torch.cuda.Device] = None,
    ):
        """
        Init method

        Parameters
        ----------
        batch_tensor: List[torch.Tensor]
            Batched image data given by the
            data loader.

        batch_super_chunk: List[Tuple[int]]
            Slices for the current super chunk

        batch_internal_slice: List[Tuple[int]]
            Slices for the internal slices within
            the pulled super chunk

        device: Optional[torch.cuda.Device]
            Device where the data will be placed.
            Default: None

        """
        self.batch_tensor = torch.stack(batch_tensor)
        self.batch_super_chunk = tuple(batch_super_chunk)
        self.batch_internal_slice = tuple(batch_internal_slice)
        self.batch_internal_slice_global = tuple(batch_internal_slice_global)

        if device is not None:
            self.batch_tensor = self.batch_tensor.to(device, non_blocking=True)

    def pin_memory(self):
        """
        Custom pin memory for GPU loading optimization
        """
        self.batch_tensor = self.batch_tensor.pin_memory()
        return self


def collate_fn(
    dataloader_return: Tuple,
    device: Optional[torch.cuda.Device] = None,
):
    """
    Collate function to deal with pulled chunks

    Parameters
    ----------
    dataloader_return: Tuple
        Tuple with the array data,
        super chunk positions and current internal
        slices for the batched data.

    Returns:
    torch.Tensor
        Returns the batch in a tensor format
    """

    len_chunks = len(dataloader_return[0][2])

    # batch_tensor
    batch_tensor = []
    batch_super_chunk = []
    batch_internal_slice = []
    batch_internal_slice_global = []

    for item in dataloader_return:
        batch_item = item[0]
        batch_super_chunk.append(item[1])
        batch_internal_slice.append(item[2])
        batch_internal_slice_global.append(item[3])

        prediction_chunksize = tuple(
            [slice_obj.stop - slice_obj.start for slice_obj in item[2]]
        )

        if isinstance(batch_item, np.ndarray):
            batch_item = torch.from_numpy(batch_item)

        if batch_item.shape != prediction_chunksize:
            # n-dim slices - batches 2D - 3D data
            min_slices = tuple(
                slice(0, min(prediction_chunksize[idx], batch_item.shape[idx]))
                for idx in range(len_chunks)
            )

            zeros = torch.zeros(
                size=prediction_chunksize, dtype=batch_item.dtype
            )

            # Filling zeros array with batch data
            zeros[min_slices] = batch_item[min_slices]
            batch_item = zeros

        batch_tensor.append(batch_item)

    return ZarrCustomBatch(
        batch_tensor=batch_tensor,
        batch_super_chunk=batch_super_chunk,
        batch_internal_slice=batch_internal_slice,
        batch_internal_slice_global=batch_internal_slice_global,
        device=device,
    )


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
        prediction_chunksize: Tuple[int, int, int],
        overlap_prediction_chunksize: Tuple[int, int, int] = None,
        super_chunksize: Optional[Tuple[int, int, int]] = None,
        target_size_mb: Optional[int] = None,
        locker: Callable[[int], Callable] = None,
        condition: Callable = None,
        locked_array: Optional[bool] = True,
    ) -> None:
        """
        Initializes the dataset class

        Parameters
        ----------
        lazy_data: ArrayLike
            Lazy array that contains the data to process

        prediction_chunksize: Tuple[int, int, int]
            Prediction chunk size. Given a zarr array
            and an estimated/provided super chunk size,
            we will take a prediction chunk size from
            the super chunk that is already in memory.

        overlap_prediction_chunksize: Tuple[int, int, int]
            Overlap between prediction chunks.
            Default: None

        super_chunksize: Tuple[int, int, int]
            Given a lazy array (not loaded in memory),
            this parameter determines how many chunks
            will be moved to memory once at a time.
            This parameter is calculated if target_size_mb
            is provided.

        target_size_mb: Optional[int]
            Target size of the super chunks in memory.
            This parameter needs to be provided in

        locker: Callable
            Locker object from multiprocessing.Locker library
            using the number of workers as a parameter

        condition: Callable[[int]]
            Condition object from multiprocessing.Condition library

        locked_array: Optional[bool]
            If we want the shared memory array to be
            locked for access.
            Default: True
        """
        super(ZarrSuperChunks, self).__init__()

        if super_chunksize is None and target_size_mb is None:
            raise ValueError(
                "Please, provide the super chunk size or target_size_mb parameters"
            )
        # Lazy data after extraction. e.g., (1, 1, 1024, 512, 512) -> (1024, 512, 512)
        self.lazy_data = extract_data(lazy_data)
        self.super_chunksize = super_chunksize
        self.prediction_chunksize = prediction_chunksize
        self.overlap_prediction_chunksize = overlap_prediction_chunksize
        self.target_size_mb = target_size_mb

        # Multiprocessing variables
        self.locker = locker
        self.condition = condition
        self.curr_super_chunk_pos = multp.Value("i", -1)

        # Initialization of super chunks
        (
            self.super_chunksize,
            self.super_chunk_slices,
            self.internal_slices,  # Local coord system within each superchunk
            self.global_internal_slices,
            self.zarr_iterator,
        ) = self.__init_super_chunks_iter()

        # Initializing shared array
        (
            self.super_chunk_in_memory,
            self.array_pointer,
        ) = self.__init_shared_array(
            shape=self.super_chunksize,
            dtype=self.lazy_data.dtype,
            locked_array=locked_array,
        )

        # Number of slices per super chunk
        self.internal_slice_sum = tuple(
            len(internal_slice) for internal_slice in self.internal_slices
        )
        self.pulled_chunks_per_super_chunk = multp.Array(
            "i", len(self.internal_slice_sum)
        )

    def __init_shared_array(
        self,
        shape: Tuple[int],
        dtype: type,
        locked_array: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, multp.Array]:
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

        If we want the shared memory array to be
            locked for access.
            Default: True

        Returns
        -------
        Tuple[torch.Tensor, multiprocessing.Array]
            torch.Tensor:
                Tensor pointing to a shared memory space

            multiprocessing.Array:
                Native shared memory object where torch
                Tensor is pointing to
        """
        # shared_array = torch.zeros(
        #     shape,
        #     dtype=torch.from_numpy(
        #         np.array(
        #             0,
        #             dtype=dtype
        #         )
        #     ).dtype
        # ).share_memory_()

        # Camilo Laiton's note:
        # This is faster and has been working reliably for me
        shared_array_base = multp.Array(
            typecode_or_type=map_dtype_to_ctype(dtype=dtype, exact=False),
            size_or_initializer=int(np.prod(shape, axis=0)),
            lock=locked_array,
        )
        if locked_array:
            shared_array_base = shared_array_base.get_obj()

        shared_array = np.ctypeslib.as_array(shared_array_base)
        shared_array = shared_array.reshape(shape)
        shared_array = torch.from_numpy(shared_array)

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

            new_super_chunksize [Tuple]: New super chunk size that was
            estimated if target_size_mb was provided.

            super_chunk_slices [ Tuple[Tuple] ]: Generated super chunk
            slices using the entire volume in order to partition the array.

            internal_slices [ Tuple[Tuple] ]: Generated slices per super
            chunk using the prediction chunking.

            zarr_iterator [ Generator ]: Generator of slices per dimension.
        """

        if self.target_size_mb is None and self.super_chunksize is None:
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
        new_super_chunksize = self.super_chunksize
        if self.target_size_mb and self.super_chunksize is None:

            print(
                f"Estimating super chunksize. Provided super chunksize: {self.super_chunksize} - Target MB: {self.target_size_mb}"
            )
            new_super_chunksize = zarr_iterator.get_block_shape(
                arr=self.lazy_data,
                target_size_mb=self.target_size_mb,
                mode="cycle",
            )

            new_super_chunksize = _closer_to_target_chunksize(
                super_chunksize=new_super_chunksize,
                chunksize=self.prediction_chunksize,
            )

            print(
                f"Estimated chunksize to fit in memory {self.target_size_mb} MiB: {new_super_chunksize}"
            )

        # Generating super chunk slices
        super_chunk_slices = tuple(
            zarr_iterator.gen_slices(
                arr_shape=self.lazy_data.shape,
                block_shape=new_super_chunksize,
                overlap_shape=self.overlap_prediction_chunksize,  # Super chunk slices with overlap between them
            )
        )

        local_internal_slices = None
        global_internal_slices = None

        np_overlap = np.array(self.overlap_prediction_chunksize) * 2
        if np.any(np_overlap):
            print(
                f"Adding overlap area to super chunk size: {new_super_chunksize} - {tuple(np.array(new_super_chunksize) + np_overlap)}"
            )
            new_super_chunksize = tuple(
                np.array(new_super_chunksize) + np_overlap
            )

            local_internal_slices = []
            global_internal_slices = []

            for sc in super_chunk_slices:
                local_slices, global_slices = (
                    zarr_iterator.gen_over_slices_on_over_superchunks(
                        arr_shape=self.lazy_data[sc].shape,
                        block_shape=self.prediction_chunksize,
                        overlap_shape=self.overlap_prediction_chunksize,
                        super_chunk_slices=sc,
                        dataset_shape=self.lazy_data.shape,
                    )
                )
                local_internal_slices.append(tuple(local_slices))
                global_internal_slices.append(tuple(global_slices))

            global_internal_slices = tuple(global_internal_slices)
            local_internal_slices = tuple(local_internal_slices)

        else:
            local_internal_slices = tuple(
                tuple(
                    zarr_iterator.gen_slices(
                        arr_shape=self.lazy_data[super_chunk_slice].shape,
                        block_shape=self.prediction_chunksize,
                    )
                )
                for super_chunk_slice in super_chunk_slices
            )

        return (
            new_super_chunksize,
            super_chunk_slices,
            local_internal_slices,
            global_internal_slices,
            zarr_iterator,
        )

    def __map_index(
        self, index: int, next_batch: Optional[bool] = False
    ) -> int:
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

    def __check_super_chunk_position(self, index: int) -> int:
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

        with self.pulled_chunks_per_super_chunk.get_lock():
            number_pulled_chunks = self.pulled_chunks_per_super_chunk[
                self.curr_super_chunk_pos.value
            ]

        with self.pulled_chunks_per_super_chunk.get_lock():
            total_chunks_super_chunk = self.internal_slice_sum[
                self.curr_super_chunk_pos.value
            ]

        if number_pulled_chunks == total_chunks_super_chunk:
            return True

        return False

    def __get_new_super_chunk(self, future_super_chunk_shape: Tuple[slice]):
        """
        Gets a new super chunk from cloud/local storage.
        This method is internal and should only be used
        inside of a locker so only one process pulls
        data from the cloud.

        Parameters
        ----------
        future_super_chunk_shape: Tuple[slice]
            Future super chunk shape, useful to check
            so arrays match in size.

        """
        curr_super_chunk_slices = self.super_chunk_slices[
            self.curr_super_chunk_pos.value
        ]

        # If super chunk in memory is not equal to future super chunk shape
        if self.super_chunk_in_memory.shape != future_super_chunk_shape:

            # It means that we are at the end of the block and we cannot
            # pull data to the overlap region from other blocks
            pad_width = tuple(
                (
                    0,
                    int(
                        self.super_chunk_in_memory.shape[idx]
                        - future_super_chunk_shape[idx]
                    ),  # padding the right and bottom of the volume only
                )
                for idx in range(len(future_super_chunk_shape))
            )

            # Padding if necessary
            padded_lazy_data = da.pad(
                array=self.lazy_data[curr_super_chunk_slices],
                pad_width=pad_width,
                mode="constant",
                constant_values=0,
            )

            try:
                self.super_chunk_in_memory[:] = torch.from_numpy(
                    padded_lazy_data.compute()
                )

            except Exception as e:
                print(
                    f"Slices: {curr_super_chunk_slices} Lazy data shape: {self.lazy_data[curr_super_chunk_slices].shape} - pad width: {pad_width} - Padded lazy: {padded_lazy_data.shape}"
                )
                raise e

        else:
            try:

                self.super_chunk_in_memory[:] = torch.from_numpy(
                    self.lazy_data[
                        self.super_chunk_slices[
                            self.curr_super_chunk_pos.value
                        ]
                    ].compute()
                )

            except ValueError as e:
                print(
                    f"Slices: {self.super_chunk_slices[self.curr_super_chunk_pos.value]} Lazy data shape: {self.lazy_data[self.super_chunk_slices[self.curr_super_chunk_pos.value]].shape}"
                )
                raise e

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

        # Checks if all chunks were returned for current super chunk
        self.locker.acquire()
        try:
            if self.__check_pulled_chunks():
                with self.curr_super_chunk_pos.get_lock():
                    self.curr_super_chunk_pos.value += 1

                # Setting new super chunk in memory
                future_super_chunk_shape = self.super_chunk_slices[
                    self.curr_super_chunk_pos.value
                ]

                future_super_chunk_shape = tuple(
                    [
                        axis_shape.stop - axis_shape.start
                        for axis_shape in future_super_chunk_shape
                    ]
                )

                # Pulls the new super chunk
                self.__get_new_super_chunk(
                    future_super_chunk_shape=future_super_chunk_shape
                )

                # Notify sleeping workers
                with self.condition:
                    self.condition.notify_all()

        except BaseException as e:
            raise BaseException(f"Error pulling data: {e}")

        finally:
            # Releasing lock
            self.locker.release()

        next_super_chunk_position = self.__check_super_chunk_position(index)

        if (
            self.curr_super_chunk_pos.value != next_super_chunk_position
            and next_super_chunk_position is not None
        ):
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
        current_internal_slice = self.internal_slices[
            self.curr_super_chunk_pos.value
        ][curr_internal_super_chunk_position]

        current_internal_slice_global = None
        if self.global_internal_slices is not None:
            current_internal_slice_global = self.global_internal_slices[
                self.curr_super_chunk_pos.value
            ][curr_internal_super_chunk_position]

        curr_super_chunk_position = self.super_chunk_slices[
            self.curr_super_chunk_pos.value
        ]

        pulled_chunk = self.super_chunk_in_memory[current_internal_slice]

        # Updating number of chunks pulled per super chunk
        with self.pulled_chunks_per_super_chunk.get_lock():
            self.pulled_chunks_per_super_chunk[
                self.curr_super_chunk_pos.value
            ] += 1

        return (
            pulled_chunk,
            curr_super_chunk_position,
            current_internal_slice,
            current_internal_slice_global,
        )

    def __getitem__(self, index: int) -> torch.Tensor:
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
        current_internal_slice = None
        curr_super_chunk_position = None

        worker_info = get_worker_info()

        if worker_info is None:
            # Main worker - No parallelism
            if (
                self.curr_super_chunk_pos.value == -1
                or self.__check_super_chunk_position(index)
                != self.curr_super_chunk_pos.value
            ):
                self.curr_super_chunk_pos.value += 1

                future_super_chunk_shape = self.super_chunk_slices[
                    self.curr_super_chunk_pos.value
                ]

                future_super_chunk_shape = tuple(
                    [
                        axis_shape.stop - axis_shape.start
                        for axis_shape in future_super_chunk_shape
                    ]
                )

                # Getting new super chunk
                self.__get_new_super_chunk(
                    future_super_chunk_shape=future_super_chunk_shape
                )

            # Mapping current index to internal batch index
            curr_internal_super_chunk_position = self.__map_index(index)

            if (
                self.internal_slice_sum[self.curr_super_chunk_pos.value]
                < curr_internal_super_chunk_position
            ):
                raise RuntimeError(
                    f"Worker got an out of bounds index: {curr_internal_super_chunk_position}"
                )

            # Pulling chunk
            curr_super_chunk_position = self.super_chunk_slices[
                self.curr_super_chunk_pos.value
            ]
            current_internal_slice = self.internal_slices[
                self.curr_super_chunk_pos.value
            ][curr_internal_super_chunk_position]

            current_internal_slice_global = None
            if self.global_internal_slices is not None:
                current_internal_slice_global = self.global_internal_slices[
                    self.curr_super_chunk_pos.value
                ][curr_internal_super_chunk_position]

            pulled_prediction_chunk = self.super_chunk_in_memory[
                current_internal_slice
            ]

        else:
            (
                pulled_prediction_chunk,
                curr_super_chunk_position,
                current_internal_slice,
                current_internal_slice_global,
            ) = self.__parallel_get_item(index)

        return (
            pulled_prediction_chunk,
            curr_super_chunk_position,
            current_internal_slice,
            current_internal_slice_global,
        )

    def __len__(self) -> int:
        """
        Customized len method. Returns the sum
        of the slices per super chunk.

        Returns
        -------
        int:
            Total sum of slices with size
            prediction_chunksize that are in
            every super chunk.
        """
        return sum(self.internal_slice_sum)

    def __del__(self):
        """
        Releasing memory pointing variables to NULL
        """
        self.curr_super_chunk_pos = None
        self.super_chunk_in_memory = None
        self.array_pointer = None


def helper_measure_dataloader_times(dataset: ZarrSuperChunks):
    """
    Measures the data loader times

    Parameters
    ----------
    dataset: ZarrSuperChunks
        Zarr super chunks dataset
    """
    dataloader = ZarrDataLoader(
        dataset, batch_size=32, pin_memory=True, num_workers=8, shuffle=False
    )

    i = 0
    times = []
    before = time.time()
    for _ in dataloader:
        after = time.time()
        time_taken = after - before
        print(f"iter {i}: time taken = {time_taken}")
        times.append(time_taken)
        i += 1
        before = time.time()
    print(
        f"average time (ignoring first iter): {sum(times[1:]) / (len(times) - 1)}"
    )


def measure_data_loader(start_method: str):
    """
    Measure the data loader times

    Parameters
    ----------
    start_method: str
        Start method for children processes
    """

    print("BENCHMARKING")

    data_path = ""
    lazy_data = da.zeros(
        (1, 1, 5981, 512, 512), chunks=(1, 1, 128, 256, 256), dtype=np.int32
    )

    print(
        f"Read dataset: {data_path} - dtype: {lazy_data.dtype} - Shape: {lazy_data.shape} - Chunksize: {lazy_data.chunksize}"
    )

    print(f"Array shape: {lazy_data.shape}")
    prediction_chunksize = (64, 64, 64)
    target_size_mb = 512

    locker = multp.Lock() if 2 else None
    condition = multp.Condition()

    zarr_dataset = ZarrSuperChunks(
        lazy_data=lazy_data,
        prediction_chunksize=prediction_chunksize,
        super_chunksize=None,
        target_size_mb=target_size_mb,
        locker=locker,
        condition=condition,
    )

    helper_measure_dataloader_times(zarr_dataset)
    zarr_dataset = ZarrSuperChunks(
        lazy_data=lazy_data,
        prediction_chunksize=prediction_chunksize,
        super_chunksize=None,
        target_size_mb=target_size_mb,
        locker=locker,
        condition=condition,
    )
    print(f"Multiprocessing times with start method {start_method}:")
    ctx = multp.get_context(start_method)
    process = ctx.Process(
        target=helper_measure_dataloader_times, args=(zarr_dataset,)
    )
    process.start()
    process.join()


def create_data_loader(
    lazy_data: ArrayLike,
    target_size_mb: int,
    prediction_chunksize: Tuple[int, ...],
    n_workers: int,
    batch_size: int,
    device: int,
    pin_memory: bool,
    dtype: type = np.float32,
    super_chunksize: Optional[Tuple[int, ...]] = None,
    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    override_suggested_cpus: Optional[bool] = True,
    drop_last: Optional[bool] = True,
    locked_array: Optional[bool] = True,
    overlap_prediction_chunksize: Tuple[int, ...] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Creates zarr data loader.

    Parameters
    ----------
    lazy_data: ArrayLike
        Lazy array to be iterated. This gives flexibility
        if you need to do some preprocessing before.

    target_size_mb: int
        Size in megabytes for each super chunk

    prediction_chunksize: Tuple[int, ...]
        Chunksize that will be sent toi the model

    n_workers: int
        Number of workers that will pull data

    batch_size: int
        Batch size

    dtype: type
        Dtype the data will be cast to. Be carefull with
        losing color information. Default: np.float32

    super_chunksize: Optional[Tuple[int, ...]]
        Super chunksize shape. If target_size_mb is None,
        then the super chunksize will be used for creating
        the super chunks. Default: None

    lazy_callback_fn: Optional[Callable[[ArrayLike], ArrayLike]]
        Lazy callback function to process lazy dataset
        before the data loader object is created.

    override_suggested_cpus: Optional[bool]
        Overrides the suggested number of CPUs used.
        Default: True

    drop_last: Optional[bool]
        Drops last batch of data

    locked_array: Optional[bool]
        If we want the shared memory array to be
        locked for access.
        Default: True

    logger: Optional[logging.Logger]
        Logger object to print how the volume is changing

    Returns
    -------
    Tuple[ZarrSuperChunks, ZarrDataLoader]
        ZarrSuperChunks pytorch dataset and
        the ZarrDataLoader objects
    """
    suggested_number_cpus = get_suggested_cpu_count()

    if override_suggested_cpus and suggested_number_cpus != n_workers:
        logger.info(
            f"Setting suggested number of CPUs in this system. From {n_workers} to {suggested_number_cpus}"
        )
        n_workers = suggested_number_cpus

    def _print(message: str):
        """
        Helper to print
        """
        if logger is not None:
            logger.info(message)

    lazy_data = extract_data(lazy_data).astype(dtype)

    _print(f"Initial shape: {lazy_data.shape}")

    overlapped_prediction_chunksize = np.array(prediction_chunksize) + (
        np.array(overlap_prediction_chunksize) * 2
    )

    lazy_data = reshape_dataset_to_prediction_chunks(
        lazy_data=lazy_data,
        prediction_chunksize=overlapped_prediction_chunksize,
    )

    _print(
        f"Shape after adding zeros in borders: {lazy_data.shape} with prediction chunksize: {overlapped_prediction_chunksize}"
    )

    if lazy_callback_fn is not None:
        lazy_data = lazy_callback_fn(lazy_data)
        _print(f"Shape after call back function: {lazy_data.shape}")

    multiprocessing_context = None
    if device is not None:
        device = torch.device(device)
        multiprocessing_context = "spawn"
        _print(
            f"Torch device: {device} and mp context {multiprocessing_context}"
        )

    partial_collate = partial(
        collate_fn,
        device=device,
    )

    locker = multp.Lock() if n_workers else None
    condition = multp.Condition()

    zarr_dataset = ZarrSuperChunks(
        lazy_data=lazy_data,
        prediction_chunksize=prediction_chunksize,
        overlap_prediction_chunksize=overlap_prediction_chunksize,
        super_chunksize=super_chunksize,
        target_size_mb=target_size_mb,
        locker=locker,
        condition=condition,
        locked_array=locked_array,
    )

    persistent_workers = True if n_workers else False

    zarr_data_loader = ZarrDataLoader(
        zarr_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_memory,  # To enable fast data transfers to GPU
        persistent_workers=persistent_workers,
        collate_fn=partial_collate,
        multiprocessing_context=multiprocessing_context,
        drop_last=drop_last,
    )

    return zarr_data_loader, zarr_dataset


def main():
    """
    Main function
    """
    # import dask.array as da

    from aind_large_scale_prediction.io import ImageReaderFactory

    BUCKET_NAME = "aind-open-data"
    IMAGE_PATH = "diSPIM_685890_2023-06-29_14-39-56/diSPIM.zarr"
    TILE_NAME = "647_D1_X_0001_Y_0001_Z_0000_ch_488.zarr"
    dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    multiscale = "3"
    pin_memory = True
    device = None

    suggested_number_cpus = get_suggested_cpu_count()

    dataset_reader = ImageReaderFactory().create(
        data_path=dataset_path,
        parse_path=False,
        multiscale=multiscale,
    )
    lazy_data = extract_data(dataset_reader.as_dask_array().astype(np.int32))
    data_path = dataset_reader.data_path

    # data_path = ""
    # lazy_data = da.zeros(
    #     (1, 1, 5981, 512, 512), chunks=(1, 1, 128, 256, 256), dtype=np.int32
    # )

    print(
        f"Read dataset: {data_path} - dtype: {lazy_data.dtype} - Shape: {lazy_data.shape} - Chunksize: {lazy_data.chunksize}"
    )

    print(f"Array shape: {lazy_data.shape}")
    results = {}
    prediction_chunksize = (64, 64, 64)

    lazy_data = reshape_dataset_to_prediction_chunks(
        lazy_data=lazy_data, prediction_chunksize=prediction_chunksize
    )
    print(f"New array shape: {lazy_data.shape} --")

    multiprocessing_context = None
    if device is not None:
        device = torch.device(device)
        multiprocessing_context = "spawn"
        print(
            f"Torch device: {device} and mp context {multiprocessing_context}"
        )

    # Creating partiall collate to provide prediction chunksize
    partial_collate = partial(
        collate_fn, prediction_chunksize=prediction_chunksize, device=device
    )
    super_chunksize = None  # (384, 768, 768)
    target_size_mb = 1024  # None

    for n_workers in [10]:  # range(0, suggested_number_cpus):
        locker = multp.Lock() if n_workers else None
        condition = multp.Condition()

        for batch_size in [1]:  # [1, 4, 8, 16]:
            print(
                f"{20*'='} Test Workers {n_workers} Batch Size {batch_size} {20*'='}"
            )

            start_time = time.time()
            zarr_dataset = ZarrSuperChunks(
                lazy_data=lazy_data,
                prediction_chunksize=prediction_chunksize,
                super_chunksize=super_chunksize,
                target_size_mb=target_size_mb,
                locker=locker,
                condition=condition,
            )
            print(f"Total super chunks: {zarr_dataset.__len__()}")

            persistent_workers = True if n_workers else False

            zarr_data_loader = ZarrDataLoader(
                zarr_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=pin_memory,  # To enable fast data transfers to GPU
                persistent_workers=persistent_workers,
                collate_fn=partial_collate,
                multiprocessing_context=multiprocessing_context,
                drop_last=True,
            )

            print("Starting the loading...")

            try:
                idx = 0
                for sample in zarr_data_loader:
                    print(
                        f"{n_workers} Batch {idx}: {sample.batch_tensor.shape} - current super chunk: {sample.batch_super_chunk} - current internal slice: {sample.batch_internal_slice} - Pinned?: {sample.batch_tensor.is_pinned()}"
                    )

                    idx += 1

            except BaseException as e:
                print(e)
                exit(1)

            end_time = time.time()
            results[f"Workers {n_workers} - batch_size {batch_size}"] = (
                end_time - start_time
            )
            print(
                f"Time executing with {n_workers} workers and batch size {batch_size}: {end_time - start_time} seconds"
            )

    print(results)
    locker = None
    condition = None


if __name__ == "__main__":
    # measure_data_loader(start_method="spawn")
    main()
    # import cProfile
    # cProfile.run("main()", filename="compute_costs.dat")
