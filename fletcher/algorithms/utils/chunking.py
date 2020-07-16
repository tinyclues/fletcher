"""Utility functions to deal with chunked arrays."""
from typing import List, Tuple

import numpy as np
import pyarrow as pa


def _calculate_chunk_offsets(chunked_array: pa.ChunkedArray) -> np.ndarray:
    """Return an array holding the indices pointing to the first element of each chunk."""
    offset = 0
    offsets = []
    for chunk in chunked_array.iterchunks():
        offsets.append(offset)
        offset += len(chunk)
    return np.array(offsets)


def _in_chunk_offsets(
    arr: pa.ChunkedArray, offsets: List[int]
) -> List[Tuple[int, int, int]]:
    """Calculate the access ranges for a given list of offsets.

    All chunk start indices must be included as offsets and the offsets must be
    unique.
    Returns a list of tuples that contain:
     * The index of the given chunk
     * The position inside the chunk
     * The length of the current range
    """
    new_offsets = []
    pos = 0
    chunk = 0
    chunk_pos = 0
    for offset, offset_next in zip(offsets, offsets[1:] + [len(arr)]):
        diff = offset - pos
        chunk_remains = len(arr.chunk(chunk)) - chunk_pos
        step = offset_next - offset
        if diff == 0:  # The first offset
            new_offsets.append((chunk, chunk_pos, step))
        elif diff == chunk_remains:
            chunk += 1
            chunk_pos = 0
            pos += chunk_remains
            new_offsets.append((chunk, chunk_pos, step))
        else:  # diff < chunk_remains
            chunk_pos += diff
            pos += diff
            new_offsets.append((chunk, chunk_pos, step))
    return new_offsets


def _combined_in_chunk_offsets(
    a: pa.ChunkedArray, b: pa.ChunkedArray
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    offsets_a = _calculate_chunk_offsets(a)
    offsets_b = _calculate_chunk_offsets(b)
    offsets = sorted(set(list(offsets_a) + list(offsets_b)))
    in_a_offsets = _in_chunk_offsets(a, offsets)
    in_b_offsets = _in_chunk_offsets(b, offsets)
    return in_a_offsets, in_b_offsets


def _not_implemented_path(*args, **kwargs):
    raise NotImplementedError("Dispatching path not implemented")
