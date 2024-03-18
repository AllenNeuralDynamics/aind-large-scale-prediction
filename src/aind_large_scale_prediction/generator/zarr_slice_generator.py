"""
Zarr slice generator that traverses an
entire lazy array
"""

from typing import Generator, List, Optional, Tuple

import dask.array as da
import numpy as np

from aind_large_scale_prediction._shared.types import ArrayLike


def _get_size(shape: Tuple[int, ...], itemsize: int) -> int:
    """
    Return the size of an array with the given shape, in bytes
    Args:
        shape: the shape of the array
        itemsize: number of bytes per array element
    Returns:
        the size of the array, in bytes
    """
    if any(s <= 0 for s in shape):
        raise ValueError("shape must be > 0 in all dimensions")
    return np.product(shape) * itemsize


def _closer_to_target_chunksize(
    super_chunksize: Tuple[int, ...],
    chunksize: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Given two shapes with the same number of dimensions,
    find the super chunk size where all chunksizes could
    fit.

    Parameters
    ----------
    super_chunksize: Tuple[int, ...]
        Super chunksize where all chunksizes must fit

    chunksize: Tuple[int, ...]
        Chunksizes of same shape within super chunksize

    Raises
    ------
    ValueError:
        If chunksizes don't have the same shape or if
        chunksize > super_chunksize

    Returns
    -------
    Tuple:
        New super chunksize where all chunks shapes fit
    """

    super_chunksize_len = len(super_chunksize)
    chunksize_len = len(chunksize)

    if super_chunksize_len != chunksize_len:
        raise ValueError(
            f"Chunksizes must have the same shape. Super chunksize len: {super_chunksize_len} - chunksize len: {chunksize_len}"
        )

    chunksize_check = [
        True if super_chunksize[idx] >= chunksize[idx] else False
        for idx in range(super_chunksize_len)
    ]

    if False in chunksize_check:
        raise ValueError(
            f"Chunksize {chunksize} must be smaller than super chunk {super_chunksize}"
        )

    factors = np.array(
        [
            int(np.rint(super_chunksize[idx] / chunksize[idx]))
            for idx in range(super_chunksize_len)
        ]
    )

    return tuple(np.multiply(np.array(chunksize), factors))


def _closer_to_target(
    shape1: Tuple[int, ...],
    shape2: Tuple[int, ...],
    target_bytes: int,
    itemsize: int,
) -> Tuple[int, ...]:
    """
    Given two shapes with the same number of dimensions,
    find which one is closer to target_bytes.
    Args:
        shape1: the first shape
        shape2: the second shape
        target_bytes: the target size for the returned shape
        itemsize: number of bytes per array element
    """
    size1 = _get_size(shape1, itemsize)
    size2 = _get_size(shape2, itemsize)
    if abs(size1 - target_bytes) < abs(size2 - target_bytes):
        return shape1
    return shape2


def expand_chunks(
    chunks: Tuple[int, int, int],
    data_shape: Tuple[int, int, int],
    target_size: int,
    itemsize: int,
    mode: str = "iso",
) -> Tuple[int, int, int]:
    """
    Given the shape and chunk size of a pre-chunked 3D array, determine the optimal chunk shape
    closest to target_size. Expanded chunk dimensions are an integer multiple of the base chunk dimension,
    to ensure optimal access patterns.
    Args:
        chunks: the shape of the input array chunks
        data_shape: the shape of the input array
        target_size: target chunk size in bytes
        itemsize: the number of bytes per array element
        mode: chunking strategy. Must be one of "cycle", or "iso"
    Returns:
        the optimal chunk shape
    """
    if any(c < 1 for c in chunks):
        raise ValueError("chunks must be >= 1 for all dimensions")
    if any(s < 1 for s in data_shape):
        raise ValueError("data_shape must be >= 1 for all dimensions")
    if any(c > s for c, s in zip(chunks, data_shape)):
        raise ValueError(
            "chunks cannot be larger than data_shape in any dimension"
        )
    if target_size <= 0:
        raise ValueError("target_size must be > 0")
    if itemsize <= 0:
        raise ValueError("itemsize must be > 0")
    if mode == "cycle":
        # get the spatial dimensions only
        current = np.array(chunks, dtype=np.uint64)
        prev = current.copy()
        idx = 0
        ndims = len(current)
        while _get_size(current, itemsize) < target_size:
            prev = current.copy()
            current[idx % ndims] = min(
                data_shape[idx % ndims], current[idx % ndims] * 2
            )
            idx += 1
            if all(c >= s for c, s in zip(current, data_shape)):
                break
        expanded = _closer_to_target(current, prev, target_size, itemsize)
    elif mode == "iso":
        initial = np.array(chunks, dtype=np.uint64)
        current = initial
        prev = current
        i = 2
        while _get_size(current, itemsize) < target_size:
            prev = current
            current = initial * i
            current = (
                min(data_shape[0], current[0]),
                min(data_shape[1], current[1]),
                min(data_shape[2], current[2]),
            )
            i += 1
            if all(c >= s for c, s in zip(current, data_shape)):
                break
        expanded = _closer_to_target(current, prev, target_size, itemsize)
    else:
        raise ValueError(f"Invalid mode {mode}")

    return tuple(int(d) for d in expanded)


class BlockedZarrArrayIterator:
    """
    Static class to write a lazy array
    in big chunks to OMEZarr
    """

    @staticmethod
    def gen_slices(
        arr_shape: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        dimension: Optional[int] = 0,
        start_shape: Optional[List[int]] = None,
        overlap_shape: Optional[List[int]] = None,
    ) -> Generator:
        """
        Generate a series of slices that can be used to traverse an array in
        blocks of a given shape.

        The method generates tuples of slices, each representing a block of
        the array. The blocks are generated by iterating over the array in
        steps of the block shape along each dimension.

        Parameters
        ----------
        arr_shape : tuple of int
            The shape of the array to be sliced.

        block_shape : tuple of int
            The desired shape of the blocks. This should be a tuple of integers
            representing the size of each dimension of the block. The length of
            `block_shape` should be equal to the length of `arr_shape`.
            If the array shape is not divisible by the block shape along a dimension,
            the last slice along that dimension is truncated.

        dimension: Optional[int]
            First slicing dimension
            Default = 0

        start_shape: Optional[List[Int]]
            Start shape in the zarr volume.
            Default: None

        overlap_shape: Optional[List[int]]
            Overlap shape between chunks

        Returns
        -------
        generator of tuple of slice
            A generator yielding tuples of slices. Each tuple can
            be used to index the input array.
        """
        if start_shape is None:
            start_shape = [0] * len(arr_shape)

        if overlap_shape is None:
            overlap_shape = [0] * len(arr_shape)

        if sum(overlap_shape) >= sum(block_shape):
            raise ValueError(
                f"Overlap shape {overlap_shape} must be smaller than block shape: {block_shape}"
            )

        if len(arr_shape) != len(block_shape) or len(arr_shape) != len(
            start_shape
        ):
            raise Exception(
                "array shape and block shape have different lengths"
            )

        for idx in range(len(arr_shape)):
            if start_shape[idx] >= arr_shape[idx] or start_shape[idx] < 0:
                raise ValueError("Please, verify your start shape")

        def _slice_along_dim(dim: int) -> Generator:
            """A helper generator function that slices along one dimension."""
            # Base case: if the dimension is beyond the last one, return an empty tuple
            if dim >= len(arr_shape):
                yield ()
            else:
                # Iterate over the current dimension in steps of the block size
                for i in range(
                    start_shape[dim], arr_shape[dim], block_shape[dim]
                ):
                    # Calculate the start index for this block
                    start_i = max(i - overlap_shape[dim], 0)

                    # Calculate the end index for this block
                    end_i = min(
                        i + block_shape[dim] + overlap_shape[dim],
                        arr_shape[dim],
                    )
                    # Generate slices for the remaining dimensions
                    for rest in _slice_along_dim(dim + 1):
                        yield (slice(start_i, end_i),) + rest

        # Start slicing along the first dimension
        return _slice_along_dim(dim=dimension)

    @staticmethod
    def gen_over_slices_on_over_superchunks(
        arr_shape: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        overlap_shape: Tuple[int, ...],
        super_chunk_slices: Tuple[slice],
        dataset_shape: Tuple[int, ...],
        dimension: Optional[int] = 0,
        start_shape: Tuple[int, ...] = None,
    ) -> Generator:
        """
        Generate a series of overlapping slices that can be used to
        traverse an array in blocks of a given shape.

        The method generates tuples of slices, each representing a block of
        the array. The blocks are generated by iterating over the array in
        steps of the block shape along each dimension.

        Parameters
        ----------
        arr_shape : tuple of int
            The shape of the array to be sliced.

        block_shape : tuple of int
            The desired shape of the blocks. This should be a tuple of integers
            representing the size of each dimension of the block. The length of
            `block_shape` should be equal to the length of `arr_shape`.
            If the array shape is not divisible by the block shape along a dimension,
            the last slice along that dimension is truncated.

        overlap_shape: Tuple[int, ...]
            Tuple of ints that represent the overlapping area that happens
            between chunks.

        super_chunk_slices: Tuple[slice]
            Slices of the current super chunk that will be in memory. It is assumed
            that the super chunks also have overlapping areas.

        dimension: Optional[int]
            First slicing dimension
            Default = 0

        start_shape: Optional[List[Int]]
            Start shape in the zarr volume.
            Default: None

        Returns
        -------
        generator of tuple of slice
            A generator yielding tuples of overlapping slices. Each tuple can
            be used to index the input array.
        """

        # If start shape is None, we assume we start in 0
        if start_shape is None:
            start_shape = [0] * len(arr_shape)

        # Convert to numpy array and unpack super chunk slices
        arr_shape = np.array(arr_shape)
        overlap_shape = np.array(overlap_shape)

        start_super_chunk_slices = []
        end_super_chunk_slices = []

        for sc in super_chunk_slices:
            start_super_chunk_slices.append(sc.start)
            end_super_chunk_slices.append(sc.stop)

        # Convert global coords starts and ends to numpy arrays
        global_start_shape = np.array(start_super_chunk_slices)
        global_stop_shape = np.array(end_super_chunk_slices)

        # Create the local coordinate system
        local_start_shape = np.array([0] * len(arr_shape))
        local_stop_shape = np.array(arr_shape)

        # Validate that we have room to move the overlap within the global start shape
        dif_shape_start = global_start_shape - overlap_shape

        dif_shape_start[dif_shape_start > 0] = 1
        dif_shape_start[dif_shape_start < 0] = 0
        dif_shape_start = dif_shape_start.astype(bool)

        # Get the local and global starts
        local_start_shape[dif_shape_start] += overlap_shape[dif_shape_start]
        global_start_shape[dif_shape_start] += overlap_shape[dif_shape_start]

        # Check if we have room to move the overlap
        # If the global stop is greater than dataset shape, no room
        sum_shape_stop_val = global_stop_shape < dataset_shape
        local_stop_shape[sum_shape_stop_val] -= overlap_shape[
            sum_shape_stop_val
        ]
        global_stop_shape[sum_shape_stop_val] -= overlap_shape[
            sum_shape_stop_val
        ]

        if overlap_shape is None:
            overlap_shape = [0] * len(arr_shape)

        if sum(overlap_shape) >= sum(block_shape):
            raise ValueError(
                f"Overlap shape {overlap_shape} must be smaller than block shape: {block_shape}"
            )

        if len(arr_shape) != len(block_shape) or len(arr_shape) != len(
            start_shape
        ):
            raise Exception(
                "array shape and block shape have different lengths"
            )

        def _slice_along_dim(
            dim: int, slice_arr_shape, slice_start, slice_stop
        ):
            """A helper generator function that slices along one dimension."""
            # Base case: if the dimension is beyond the last one, return an empty tuple
            if dim >= len(slice_arr_shape):
                yield ()
            else:
                # Iterate over the current dimension in steps of the block size
                for i in range(
                    slice_start[dim], slice_stop[dim], block_shape[dim]
                ):
                    # Calculate the start index for this block
                    start_i = max(i - overlap_shape[dim], 0)

                    # Calculate the end index for this block
                    end_i = min(
                        i + block_shape[dim] + overlap_shape[dim],
                        slice_arr_shape[dim],
                    )
                    # Generate slices for the remaining dimensions
                    for rest in _slice_along_dim(
                        dim + 1, slice_arr_shape, slice_start, slice_stop
                    ):
                        yield (slice(start_i, end_i),) + rest

        # Start slicing along the given dimension
        local_super_chunk_slices = _slice_along_dim(
            dim=dimension,
            slice_arr_shape=arr_shape,
            slice_start=local_start_shape,
            slice_stop=local_stop_shape,
        )

        global_super_chunk_slices = _slice_along_dim(
            dim=dimension,
            slice_arr_shape=dataset_shape,
            slice_start=global_start_shape,
            slice_stop=global_stop_shape,
        )

        return local_super_chunk_slices, global_super_chunk_slices

    @staticmethod
    def get_block_shape(arr: ArrayLike, target_size_mb=409600, mode="cycle"):
        """
        Given the shape and chunk size of a pre-chunked array, determine the optimal block shape
        closest to target_size. Expanded block dimensions are an integer multiple of the chunk dimension
        to ensure optimal access patterns.
        Args:
            arr: the input array
            target_size_mb: target block size in megabytes, default is 409600
            mode: strategy. Must be one of "cycle", or "iso"
        Returns:
            the block shape
        """
        axes = len(arr.chunksize)
        if isinstance(arr, da.Array):
            chunks = arr.chunksize[-axes:]
        else:
            chunks = arr.chunks[-axes:]

        return expand_chunks(
            chunks,
            arr.shape[-axes:],
            target_size_mb * 1024**2,
            arr.itemsize,
            mode,
        )
