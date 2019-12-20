# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numba
import numpy as np
import pyarrow as pa

from ._numba_compat import NumbaStringArray


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


@numba.jit(nogil=True, nopython=True)
def str_concat(sa1, sa2):
    # TODO: check overflow of size
    assert sa1.size == sa2.size

    result_missing = sa1.missing | sa2.missing
    result_offsets = np.zeros(sa1.size + 1, np.uint32)
    result_data = np.zeros(sa1.byte_size + sa2.byte_size, np.uint8)

    offset = 0
    for i in range(sa1.size):
        if sa1.isnull(i) or sa2.isnull(i):
            result_offsets[i + 1] = offset
            continue

        for j in range(sa1.byte_length(i)):
            result_data[offset] = sa1.get_byte(i, j)
            offset += 1

        for j in range(sa2.byte_length(i)):
            result_data[offset] = sa2.get_byte(i, j)
            offset += 1

        result_offsets[i + 1] = offset

    result_data = result_data[:offset]

    return NumbaStringArray(result_missing, result_offsets, result_data, 0)


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
