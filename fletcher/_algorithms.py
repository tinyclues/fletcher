# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from typing import Optional

import numba
import numpy as np
import pyarrow as pa
from numba import njit, prange

FASTMATH = True
PARALLEL = True


def _extract_data_buffer_as_np_array(array: pa.Array) -> np.ndarray:
    """Extract the data buffer of a numeric-typed pyarrow.Array as an np.ndarray."""
    dtype = array.type.to_pandas_dtype()
    start = array.offset
    end = array.offset + len(array)
    if pa.types.is_boolean(array.type):
        return np.unpackbits(
            _buffer_to_view(array.buffers()[1]).view(np.uint8), bitorder="little"
        )[start:end].astype(bool)
    else:
        return _buffer_to_view(array.buffers()[1]).view(dtype)[start:end]


EMPTY_BUFFER_VIEW = np.array([], dtype=np.uint8)


def _buffer_to_view(buf: Optional[pa.Buffer]) -> np.ndarray:
    """Extract the pyarrow.Buffer as np.ndarray[np.uint8]."""
    if buf is None:
        return EMPTY_BUFFER_VIEW
    else:
        return np.asanyarray(buf).view(np.uint8)


def _extract_isnull_bitmap(arr: pa.Array, offset: int, length: int):
    """
    Extract isnull bitmap with offset and padding.

    Ensures that even when pyarrow does return an empty bitmap that a filled
    one will be returned.
    """
    buf = _buffer_to_view(arr.buffers()[0])
    if len(buf) > 0:
        return buf[offset : offset + length]
    else:
        return np.full(length, fill_value=255, dtype=np.uint8)


@numba.jit(nogil=True, nopython=True)
def _extract_isnull_bytemap(bitmap, bitmap_length, bitmap_offset, dst_offset, dst):
    """Write the values of a valid bitmap as bytes to a pre-allocated isnull bytemap (internal).

    Parameters
    ----------
    bitmap: pyarrow.Buffer
        bitmap where a set bit indicates that a value is valid
    bitmap_length: int
        Number of bits to read from the bitmap
    bitmap_offset: int
        Number of bits to skip from the beginning of the bitmap.
    dst_offset: int
        Number of bytes to skip from the beginning of the output
    dst: numpy.array(dtype=bool)
        Pre-allocated numpy array where a byte is set when a value is null
    """
    for i in range(bitmap_length):
        idx = bitmap_offset + i
        byte_idx = idx // 8
        bit_mask = 1 << (idx % 8)
        dst[dst_offset + i] = (bitmap[byte_idx] & bit_mask) == 0


def extract_isnull_bytemap(chunked_array):
    """
    Extract the valid bitmaps of a chunked array into numpy isnull bytemaps.

    Parameters
    ----------
    chunked_array: pyarrow.ChunkedArray

    Returns
    -------
    valid_bytemap: numpy.array
    """
    if chunked_array.null_count == len(chunked_array):
        return np.ones(len(chunked_array), dtype=bool)

    result = np.zeros(len(chunked_array), dtype=bool)
    if chunked_array.null_count == 0:
        return result

    offset = 0
    for chunk in chunked_array.chunks:
        if chunk.null_count > 0:
            _extract_isnull_bytemap(
                chunk.buffers()[0], len(chunk), chunk.offset, offset, result
            )
        offset += len(chunk)

    return result


@numba.jit(nogil=True, nopython=True)
def isnull(sa):
    result = np.empty(sa.size, np.uint8)
    _isnull(sa, 0, result)
    return result


@numba.jit(nogil=True, nopython=True)
def _isnull(sa, offset, out):
    for i in range(sa.size):
        out[offset + i] = sa.isnull(i)


@numba.jit(nogil=True, nopython=True)
def _startswith(sa, needle, na, offset, out):
    for i in range(sa.size):
        if sa.isnull(i):
            out[offset + i] = na
            continue

        if sa.byte_length(i) < needle.length:
            out[offset + i] = 0
            continue

        for j in range(needle.length):
            if sa.get_byte(i, j) != needle.get_byte(j):
                out[offset + i] = 0
                break

        else:
            out[offset + i] = 1


@numba.jit(nogil=True, nopython=True)
def _endswith(sa, needle, na, offset, out):
    for i in range(sa.size):
        if sa.isnull(i):
            out[offset + i] = na
            continue

        string_length = sa.byte_length(i)
        needle_length = needle.length
        if string_length < needle.length:
            out[offset + i] = 0
            continue

        for j in range(needle_length):
            if sa.get_byte(i, string_length - needle_length + j) != needle.get_byte(j):
                out[offset + i] = 0
                break

        else:
            out[offset + i] = 1


@numba.jit(nogil=True, nopython=True)
def str_length(sa):
    result = np.empty(sa.size, np.uint32)

    for i in range(sa.size):
        result[i] = sa.length(i)

    return result


@numba.njit(locals={"valid": numba.bool_, "value": numba.bool_})
def _any_op(length, valid_bits, data):
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask
        if (valid and value) or (not valid):
            return True

    return False


@numba.njit(locals={"valid": numba.bool_, "value": numba.bool_})
def _any_op_skipna(length, valid_bits, data):
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask
        if valid and value:
            return True

    return False


@numba.njit(locals={"value": numba.bool_})
def _any_op_nonnull(length, data):
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        value = data[byte_offset] & mask
        if value:
            return True

    return False


def any_op(arr, skipna):
    if isinstance(arr, pa.ChunkedArray):
        return any(any_op(chunk, skipna) for chunk in arr.chunks)

    if arr.null_count == 0:
        return _any_op_nonnull(len(arr), arr.buffers()[1])
    if skipna:
        return _any_op_skipna(len(arr), *arr.buffers())
    return _any_op(len(arr), *arr.buffers())


@numba.njit(locals={"valid": numba.bool_, "value": numba.bool_})
def _all_op(length, valid_bits, data):
    # This may be specific to Pandas but we return True as long as there is not False in the data.
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        valid = valid_bits[byte_offset] & mask
        value = data[byte_offset] & mask
        if valid and not value:
            return False
    return True


@numba.njit(locals={"value": numba.bool_})
def _all_op_nonnull(length, data):
    for i in range(length):
        byte_offset = i // 8
        bit_offset = i % 8
        mask = np.uint8(1 << bit_offset)
        value = data[byte_offset] & mask
        if not value:
            return False
    return True


def all_op(arr, skipna):
    if isinstance(arr, pa.ChunkedArray):
        return all(all_op(chunk, skipna) for chunk in arr.chunks)

    if arr.null_count == 0:
        return _all_op_nonnull(len(arr), arr.buffers()[1])
    # skipna is not relevant in the Pandas behaviour
    return _all_op(len(arr), *arr.buffers())


def aggregate_fletcher_array(fr_arr, aggregator):
    """
    Return a scalar by aggregating the FletcherArray depending on the keyword aggregator.

    Parameters
    ----------
    fr_arr: fr.FletcherArray
    aggregator: string

    Returns
    -------
    scalar

    Notes
    -----
    For now, only max and min are implemented in this function.

    """
    if len(fr_arr) == 0:
        return None

    def aggregate_one_chunk(chunk):
        arr_buff = np.frombuffer(
            chunk.buffers()[1], dtype=chunk.type.to_pandas_dtype()
        )[chunk.offset : chunk.offset + len(chunk)]
        if chunk.null_count == 0:
            op_ma = {"max": np.max, "min": np.min}[aggregator]
            return op_ma(arr_buff)
        else:
            op_ma = {"max": max_ma, "min": min_ma}[aggregator]
            return op_ma(arr_buff, chunk.buffers()[0], chunk.offset)

    op = {"max": max, "min": min}[aggregator]
    return op(aggregate_one_chunk(ch) for ch in fr_arr.data.iterchunks())


@numba.jit(nogil=True, nopython=True)
def max_ma(arr, bitmap, offset):
    """Compute the max of a numpy array taking into account the null_mask."""
    res = -np.inf
    for i in range(len(arr)):
        byte_offset = (i + offset) // 8
        bit_offset = (i + offset) % 8
        mask = np.uint8(1 << bit_offset)
        if bitmap[byte_offset] & mask:
            a = arr[i]
            if a > res:
                res = a
    return res


@numba.jit(nogil=True, nopython=True)
def min_ma(arr, bitmap, offset):
    """Compute the min of a numpy array taking into account the null_mask."""
    res = np.inf
    for i in range(len(arr)):
        byte_offset = (i + offset) // 8
        bit_offset = (i + offset) % 8
        mask = np.uint8(1 << bit_offset)
        if bitmap[byte_offset] & mask:
            a = arr[i]
            if a < res:
                res = a
    return res


def integer_array_to_numpy(array: pa.IntegerArray, fill_null_value: int):
    """
    Transform pyarrow integer array to numpy array.

    It is done without copy (view to original data buffer)
    null values are replaced by given fill_null_value.

    Parameters
    ----------
    array: pa.IntegerArray
        pyarrow integer array to transform
    fill_null_value: int
        value to fill null values with

    Returns
    -------
    array : ndarray
        NumPy view on array's data
    """
    assert pa.types.is_integer(array.type)
    null_mask = extract_isnull_bytemap(pa.chunked_array([array]))
    res = np.frombuffer(array.buffers()[1], dtype=array.type.to_pandas_dtype())[
        array.offset : array.offset + len(array)
    ]
    res[null_mask] = fill_null_value
    return res


@njit(fastmath=FASTMATH)
def _get_new_indptr(self_indptr, indices, new_indptr):
    for i in range(len(indices)):
        row = indices[i]
        new_indptr[i + 1] = new_indptr[i] + self_indptr[row + 1] - self_indptr[row]
    return new_indptr


@njit(fastmath=FASTMATH, parallel=PARALLEL)
def _fill_up_indices(new_indptr, new_indices, self_indices, self_indptr, indices):
    for i in prange(len(indices)):
        row = indices[i]
        size = self_indptr[row + 1] - self_indptr[row]
        if size > 0:
            new_indices[new_indptr[i] : new_indptr[i + 1]] = self_indices[
                self_indptr[row] : self_indptr[row + 1]
            ]


def take_indices_on_pyarrow_list(array, indices):
    """Return a pyarrow.ListArray or pyarrow.LargeListArray containing only the rows of the given indices."""
    if len(array.flatten()) == 0:
        return array.take(pa.array(indices))

    dtype = np.int64 if pa.types.is_large_list(array.type) else np.int32

    self_indptr = np.frombuffer(array.buffers()[1], dtype=dtype)[
        array.offset : array.offset + len(array) + 1
    ]

    self_indices = np.frombuffer(
        array.buffers()[3], dtype=array.type.value_type.to_pandas_dtype()
    )[self_indptr[0] : self_indptr[-1]]

    self_indptr = self_indptr - self_indptr[0]
    self_indptr.setflags(write=0)
    array.validate()

    length = indices.shape[0]

    new_indptr = np.zeros(length + 1, dtype=self_indptr.dtype)
    new_indptr = _get_new_indptr(self_indptr, indices, new_indptr)
    new_indices = np.zeros(new_indptr[length], dtype=self_indices.dtype)

    _fill_up_indices(new_indptr, new_indices, self_indices, self_indptr, indices)
    if new_indptr[-1] < np.iinfo(np.int32).max:
        return pa.ListArray.from_arrays(new_indptr, new_indices)
    else:
        return pa.LargeListArray.from_arrays(new_indptr, new_indices)


@numba.jit(nogil=True, nopython=True)
def _merge_non_aligned_bitmaps(
    valid_a: np.ndarray,
    inner_offset_a: int,
    valid_b: np.ndarray,
    inner_offset_b: int,
    length: int,
    result: np.ndarray,
) -> None:
    for i in range(length):
        a_pos = inner_offset_a + i
        byte_offset_a = a_pos // 8
        bit_offset_a = a_pos % 8
        mask_a = np.uint8(1 << bit_offset_a)
        value_a = valid_a[byte_offset_a] & mask_a

        b_pos = inner_offset_b + i
        byte_offset_b = b_pos // 8
        bit_offset_b = b_pos % 8
        mask_b = np.uint8(1 << bit_offset_b)
        value_b = valid_b[byte_offset_b] & mask_b

        byte_offset_result = i // 8
        bit_offset_result = i % 8
        mask_result = np.uint8(1 << bit_offset_result)

        current = result[byte_offset_result]
        if (
            value_a and value_b
        ):  # must be logical, not bit-wise as different bits may be flagged
            result[byte_offset_result] = current | mask_result
        else:
            result[byte_offset_result] = current & ~mask_result


def _merge_valid_bitmaps(a: pa.Array, b: pa.Array) -> np.ndarray:
    """Merge two valid masks of pyarrow.Array instances.

    This method already assumes that both arrays are of the same length.
    This property is not checked again.
    """
    length = len(a) // 8
    if len(a) % 8 != 0:
        length += 1

    offset_a = a.offset // 8
    if a.offset % 8 != 0:
        pad_a = 1
    else:
        pad_a = 0
    valid_a = _extract_isnull_bitmap(a, offset_a, length + pad_a)

    offset_b = b.offset // 8
    if b.offset % 8 != 0:
        pad_b = 1
    else:
        pad_b = 0
    valid_b = _extract_isnull_bitmap(b, offset_b, length + pad_b)

    if a.offset % 8 == 0 and b.offset % 8 == 0:
        result = valid_a & valid_b

        # Mark trailing bits with 0
        if len(a) % 8 != 0:
            result[-1] = result[-1] & (2 ** (len(a) % 8) - 1)
        return result
    else:
        # Allocate result
        result = np.zeros(length, dtype=np.uint8)

        inner_offset_a = a.offset % 8
        inner_offset_b = b.offset % 8
        # TODO: We can optimite this when inner_offset_a == inner_offset_b
        _merge_non_aligned_bitmaps(
            valid_a, inner_offset_a, valid_b, inner_offset_b, len(a), result
        )

        return result
