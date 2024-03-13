"""
Generator utility functions
"""

import ctypes
import os
from collections import deque
from itertools import chain
from sys import getsizeof, stderr
from typing import List, Optional, Tuple

import numpy as np

from .._shared.types import ArrayLike

try:
    from reprlib import repr
except ImportError:
    pass


def get_suggested_cpu_count():
    """
    Returns the suggested number of CPUs
    to be used in a system
    """
    cpus = None
    try:
        # Attempt to use os.sched_getaffinity if available (Linux specific)
        cpus = os.sched_getaffinity(0)
        cpus = len(cpus)
    except AttributeError:
        # Fallback to os.cpu_count() if os.sched_getaffinity is not available
        cpus = os.cpu_count()

    return cpus


def total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint an object and all of its contents.

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
        """
        Gets the size of an object
        """
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


def divide_len_workers(
    size_len: int, worker_id: int, total_workers: int
) -> Tuple[int, int]:
    """
    Divides the number of instances between workers

    Parameters
    ----------
    size_len: int
        Total size of elements to divide

    worker_id: int
        Worker id

    total_workers: int
        Total number of workers in the pool

    Returns
    -------
    int
        total number of element that worker
        will receive
    """
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


def map_dtype_to_ctype(dtype: type, exact: Optional[bool] = False) -> type:
    """
    Maps a dtype to a ctype useful to create
    share memory compartments.

    Parameters
    ----------
    dtype: type
        dtype trying to map. At the moment, only
        mapping numpy dtypes.

    exact: Optional[bool]
        If True, returns the exact corresponding
        ctype, otherwise returns the ctype that
        pytorch Datasets allow at the moment
        for tensors. These are: float64, float32,
        float16, complex64, complex128, int64, int32,
        int16, int8, uint8, and bool.

    Returns
    -------
    type
        Corresponding dtype.
    """
    if isinstance(dtype, np.dtype):
        dtype = dtype.name

    # Allowing numerical values
    ALLOWED_DTYPES = [
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ]

    if not exact:
        if dtype not in ALLOWED_DTYPES:
            raise ValueError(
                f"Allowed dtypes, {ALLOWED_DTYPES}, provided: {dtype}"
            )

    map_dict = {
        "int8": ctypes.c_byte,
        "uint8": ctypes.c_ubyte,
        "int16": ctypes.c_short,
        "uint16": ctypes.c_ushort,  # not allowed
        "int32": ctypes.c_int,
        "uint32": ctypes.c_uint,  # Not allowed
        "int64": ctypes.c_long,
        "uint64": ctypes.c_ulong,  # Not allowed
        "float32": ctypes.c_float,
        "float64": ctypes.c_double,
    }

    return map_dict[dtype]


def find_position_in_total_sum(number: int, numbers_list: List[int]) -> int:
    """
    Returns the position where the number belongs
    within a range based on the number list provided.

    e.g., 1 == number: 120 number_list: [100, 100, 100, 24]
    position of number 120 -> 1
    e.g., 2 == number: 510 number_list: [200, 200, 100, 24]
    position of number 510 -> 3
    e.g., 3 == number: -1 number_list: [200, 200, 100, 24]
    position of number -1 -> None
    e.g., 4 == number: 600 number_list: [200, 200, 100, 24]
    position of number 600 -> None

    Parameters
    ----------
    number: int
        Number to be found in the range list

    number_list: List[int]
        Range within the list will be found.

    Returns
    -------
    int
        Position in the list. None if provided number
        is out of bounds in the list
    """
    if not numbers_list:
        return None  # Handle the case where the list is empty
    total_sum = sum(numbers_list)
    if number < 0 or number >= total_sum:
        return None  # Number is out of bounds
    position = 0
    cumulative_sum = 0
    for idx, value in enumerate(numbers_list):
        cumulative_sum += value
        if number < cumulative_sum:
            position = idx
            break
    return position


def estimate_output_volume(
    image_shape: Tuple[int],
    chunk_shape: Tuple[int],
    overlap_per_axis: Tuple[int],
) -> Tuple[int]:
    """
    Estimates output volume based on the overlap region.

    Parameters
    ----------
    image_shape: Tuple[int]
        Original image shape

    chunk_shape: Tuple[int]
        Chunk shape

    overlap_per_axis: Tuple[int]
        Overlap per axis between chunks.

    Returns
    -------
    Tuple[int]
        Output volume considering that each subchunk
        will be resized to chunk_shape + overlap_per_axis
        depending in which block we're dealing with.
    """
    len_image_shape = len(image_shape)

    if len_image_shape != len(chunk_shape) or len(chunk_shape) != len(
        overlap_per_axis
    ):
        raise ValueError(
            "Please, verify the parameters for the output volume function"
        )

    dims = len(image_shape)
    iter_positions = range(0, dims)

    res = []
    for dim in iter_positions:
        chunk_size_axis = np.ceil(image_shape[dim] / chunk_shape[dim])
        new_axis_lenght = (chunk_size_axis * chunk_shape[dim]) + (
            (chunk_size_axis) * (overlap_per_axis[dim] * 2)
        )
        res.append(np.floor(new_axis_lenght).astype(int))

    return tuple(res)


def get_chunk_numbers(
    image_shape: Tuple[int],
    nd_positions: List[Tuple[int]],
    nd_chunk_size: Tuple[int],
) -> Tuple[int]:
    """
    Get chunk number in each axis

    Parameters
    ----------
    image_shape: Tuple[int]
        Image shape

    nd_positions: List[Tuple[int]]
        Positions to get the chunknumber

    nd_chunk_size: Tuple[int]
        Chunk size in the same order as positions.
        E.g., nd_positions = [[Z,Y,X]] then
        nd_chunk_size = [[Z,Y,X]]

    Returns
    -------
    Tuple[int]
        Tuple with the location of the chunk
    """
    nd_positions = np.array(nd_positions)
    # print("Position start: ", nd_positions, " chunksize: ", nd_chunk_size)
    nd_positions = np.clip(
        nd_positions, 1, np.array(image_shape) - 1, out=nd_positions
    )
    # print("CLipped pos: ", nd_positions)
    nd_positions = nd_positions.transpose()

    for axis in range(nd_positions.shape[0]):
        nd_positions[axis, :] = np.floor(
            nd_positions[axis, :] / nd_chunk_size[axis]
        ).astype(np.uint32)

    nd_positions = nd_positions.transpose()
    # print("Chunk: ", nd_positions)
    # exit()
    return nd_positions


def get_chunk_start_position(
    image_shape: Tuple[int],
    chunk_size_zyx: Tuple[int],
    dest_zyx: Tuple[int],
) -> Tuple[int]:
    """
    Returns the start position of a
    zyx location based on the chunksize.

    Parameters
    ----------
    image_shape: Tuple[int]
        Image shape

    chunk_size_zyx: Tuple[int]
        Chunk size in ZYX direction

    zyx_pos: Tuple[int]
        ZYX position to get the chunknumber

    Returns
    -------
    Tuple[int]
        Start ZYX location of the dest_zyx
        position in the image space
    """

    axis_chunk_number = get_chunk_numbers(
        image_shape=image_shape,
        zyx_pos=dest_zyx,
        chunk_size_zyx=chunk_size_zyx,
    )

    new_zyx_location = []
    for axis in range(0, len(chunk_size_zyx)):
        proposed_pos = min(
            axis_chunk_number[axis] * chunk_size_zyx[axis], image_shape[axis]
        )

        new_zyx_location.append(proposed_pos)

    return tuple(new_zyx_location)


def recover_global_position(
    super_chunk_slice: Tuple[int], internal_slices: List[Tuple[int]]
) -> Tuple[int]:
    """
    Recovers global coordinate position based on
    a local coordinate position in a super chunk.

    Parameters
    ----------
    super_chunk_slice: Tuple[int]
        Super chunk slices

    internal_slices: List[Tuple[int]]
        Internal slices in super chunks

    Returns
    -------
    Tuple[Tuple[int]]
        Global coordinate position of a
        internal slice of a super chunk.
        The tuple contains both start and
        end positions, start positions only
        and end only positions.
    """

    len_internal_slices = len(internal_slices[0])
    # Getting internal slices
    zyx_global_slices = []
    zyx_global_slices_start = []
    zyx_global_slices_end = []

    for internal_slice in internal_slices:

        zyx_internal_slice_start = []
        zyx_internal_slice_end = []
        zyx_internal_slice = []

        for idx_internal in range(len_internal_slices):
            start = (
                internal_slice[idx_internal].start
                + super_chunk_slice[idx_internal].start
            )
            stop = start + (
                internal_slice[idx_internal].stop
                - internal_slice[idx_internal].start
            )
            zyx_internal_slice.append(slice(start, stop))
            zyx_internal_slice_start.append(start)
            zyx_internal_slice_end.append(stop)

        zyx_global_slices_start.append(tuple(zyx_internal_slice_start))
        zyx_global_slices_end.append(tuple(zyx_internal_slice_end))
        zyx_global_slices.append(tuple(zyx_internal_slice))

    return (
        tuple(zyx_internal_slice),
        tuple(zyx_global_slices_start),
        tuple(zyx_global_slices_end),
    )


def get_output_coordinate_overlap(
    chunk_axis_numbers: np.array,
    prediction_chunksize_overlap: np.array,
    batch_img_tensor_shape: Tuple[int],
):
    """
    Get output coordinate when we are iterating
    the zarr dataset in overlaped chunks.

    Parameters
    ----------
    chunk_axis_numbers: np.array
        All the chunk positions for the
        current bach of data.

    prediction_chunksize_overlap: np.array
        Overlap between contiguous chunks.

    batch_img_tensor_shape: Tuple[int]
        Shape of the images excluding the
        batch number

    Returns
    -------
    Tuple[int]
        Destination coordinate where the data
        will be written.
    """

    n_dims = len(chunk_axis_numbers[0])
    if n_dims < len(batch_img_tensor_shape):
        batch_img_tensor_shape = batch_img_tensor_shape[-n_dims:]

    dest_pos_start = chunk_axis_numbers * prediction_chunksize_overlap
    dest_pos_end = dest_pos_start + np.array(batch_img_tensor_shape)

    dest_pos_slices = []
    for position in range(0, dest_pos_start.shape[0]):
        curr_pos = []
        for ix in range(0, dest_pos_start.shape[-1]):
            curr_pos.append(
                slice(
                    dest_pos_start[position][ix],
                    dest_pos_end[position][ix],
                )
            )

        dest_pos_slices.append(tuple(curr_pos))

    return tuple(dest_pos_slices)
