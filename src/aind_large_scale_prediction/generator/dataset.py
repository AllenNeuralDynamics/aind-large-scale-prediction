"""
Defines the PyTorch Datasets classes
to load the models
"""
from __future__ import print_function

import ctypes
import multiprocessing
from collections import deque
from itertools import chain
from sys import getsizeof, stderr
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from aind_large_scale_prediction._shared.types import ArrayLike
from aind_large_scale_prediction.generator.zarr_slice_generator import (
    BlockedZarrArrayIterator,
)
from aind_large_scale_prediction.io.utils import extract_data

try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def divide_len_workers(size_len, worker_id, total_workers):
    if not worker_id:
        raise ValueError("Verify the numbers of workers")

    elif worker_id == 1 and worker_id == total_workers:
        return size_len

    if size_len < worker_id:
        raise ValueError("Check your parameters, oversplitting work")

    instances_per_worker = size_len // total_workers

    lower = (worker_id - 1) * (instances_per_worker)
    upper = (
        size_len
        if worker_id == total_workers
        else lower + instances_per_worker
    )

    return lower, upper


class ZarrSuperChunks2(Dataset):
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

        # Calculating total len of the dataset
        self.rechunked_super_chunk_slices = [
            self.lazy_data[super_chunk_slice]
            .rechunk(prediction_chunk_size)
            .blocks.size
            for super_chunk_slice in self.super_chunk_slices
        ]
        self.total_chunksize = sum(self.rechunked_super_chunk_slices)

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
        shared_array_base = multiprocessing.Array(
            ctypes.c_float, np.prod(shape)
        )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(shape)
        shared_array = torch.from_numpy(shared_array)

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
                block_shape=self.super_chunk_size,
            )
        )

        internal_slices = (
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

    def __getitem__(self, index):
        print(index)
        return np.zeros((1), dtype=np.uint8)

    def __len__(self):
        return sum(
            len(internal_slice) for internal_slice in self.internal_slices
        )


class ZarrSuperChunks(IterableDataset):
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
        tuple_slices,
    ) -> None:
        super(ZarrSuperChunks, self).__init__()

        self.lazy_data = lazy_data
        self.prediction_chunk_size = prediction_chunk_size

        # Rechunked lazy data to prediction chunk size to get metrics
        self.lazy_data_rechunked = self.lazy_data.rechunk(
            prediction_chunk_size
        )
        self.tuple_slices = tuple_slices

    def __iter__(self):
        worker_info = get_worker_info()

        # Single-process data loading
        if worker_info is None:
            # self.super_chunk_slices, self.zarr_iterator
            lower_limit, top_limit = 0, len(self.tuple_slices)

        else:
            print(
                "CHECK: ",
                len(self.tuple_slices),
                worker_info.id,
                worker_info.num_workers,
            )
            lower_limit, top_limit = divide_len_workers(
                size_len=len(self.tuple_slices),
                worker_id=worker_info.id + 1,
                total_workers=worker_info.num_workers,
            )

        return iter(self.tuple_slices[lower_limit:top_limit])


def init_super_chunks_iter(lazy_data, target_size_mb, super_chunk_size):
    chunk_size_megabytes = (lazy_data.blocks[0, 0, 0].nbytes) / (1024 * 1024)

    if target_size_mb is not None and chunk_size_megabytes > target_size_mb:
        raise ValueError(
            f"Please, check your chunk size ({chunk_size_megabytes}) and target size ({target_size_mb})."
        )

    # Getting super chunks that will be sent to GPU for prediction
    zarr_iterator = BlockedZarrArrayIterator()

    # Overwriting super chunk size if target size in mb is provided
    if target_size_mb:
        super_chunk_size = zarr_iterator.get_block_shape(
            arr=lazy_data, target_size_mb=target_size_mb, mode="cycle"
        )

        print(
            f"Chunksize to fit in memory {target_size_mb} MiB: {super_chunk_size}"
        )

    print(lazy_data.shape, super_chunk_size)
    super_chunk_slices = list(
        zarr_iterator.gen_slices(
            arr_shape=lazy_data.shape, block_shape=super_chunk_size
        )
    )

    return super_chunk_size, super_chunk_slices, zarr_iterator


def main():
    from torch.utils.data import DataLoader

    from aind_large_scale_prediction.io import ImageReaderFactory

    BUCKET_NAME = "aind-open-data"
    IMAGE_PATH = "diSPIM_685890_2023-06-29_14-39-56/diSPIM.zarr"
    TILE_NAME = "647_D1_X_0001_Y_0001_Z_0000_ch_488.zarr"

    dataset_path = f"s3://{BUCKET_NAME}/{IMAGE_PATH}/{TILE_NAME}"
    multiscale = "2"
    zarr_iterator = BlockedZarrArrayIterator()
    dataset_reader = ImageReaderFactory().create(
        data_path=dataset_path,
        parse_path=False,
        multiscale=multiscale,
    )

    print(
        f"Read dataset: {dataset_reader.data_path} - Shape: {dataset_reader.shape} - Chunks: {dataset_reader.chunks}"
    )

    lazy_data = extract_data(dataset_reader.as_dask_array())
    (
        super_chunk_size,
        super_chunk_slices,
        zarr_iterator,
    ) = init_super_chunks_iter(
        lazy_data=lazy_data, target_size_mb=512, super_chunk_size=None
    )
    # Should be aligned with the actual chunks of the dataset
    prediction_chunk_size = (64, 128, 128)

    for super_chunk_slice in super_chunk_slices:
        super_chunk_lazy_data = lazy_data[super_chunk_slice]
        print(f"Dataset: {super_chunk_lazy_data.shape}")
        chunk_slices = tuple(
            zarr_iterator.gen_slices(
                arr_shape=lazy_data[super_chunk_slice].shape,
                block_shape=prediction_chunk_size,
            )
        )

        # print(f"Total size: {total_size(chunk_slices)}")

        # Creating new ZarrSuperChunks dataset
        super_chunks_dataset = ZarrSuperChunks(
            lazy_data=lazy_data,
            prediction_chunk_size=prediction_chunk_size,
            tuple_slices=chunk_slices,
        )

        dataloader = DataLoader(
            super_chunks_dataset, batch_size=8, shuffle=False, num_workers=0
        )

        for i, retrieved_slices in enumerate(dataloader):
            print(f"{i} -> {retrieved_slices}")

        exit()

    # zarr_dataset = ZarrSuperChunks(
    #     lazy_data=dataset_reader.as_dask_array(),
    #     prediction_chunk_size=(64, 128, 128),
    #     super_chunk_size=None,
    #     target_size_mb=512
    # )

    # dataloader = DataLoader(zarr_dataset, batch_size=8, shuffle=False, num_workers=1)

    # for i, sample in enumerate(dataloader):
    #     print(i, sample)


if __name__ == "__main__":
    main()
