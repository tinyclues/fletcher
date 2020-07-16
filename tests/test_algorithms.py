from datetime import timedelta
from typing import List, Optional, Tuple

import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from hypothesis import example, given, settings

# fmt: off
import fletcher as fr
from fletcher._algorithms import (
    _extract_data_buffer_as_np_array,
    _merge_valid_bitmaps,
    all_op,
    any_op,
    integer_array_to_numpy,
    take_indices_on_pyarrow_list,
)
from fletcher.algorithms.utils.chunking import _calculate_chunk_offsets, _combined_in_chunk_offsets, _in_chunk_offsets

# fmt: on


def test_calculate_chunk_offsets():
    arr = pa.chunked_array([[1, 1, 1]])
    npt.assert_array_equal(_calculate_chunk_offsets(arr), np.array([0]))
    arr = pa.chunked_array([[1], [1, 1]])
    npt.assert_array_equal(_calculate_chunk_offsets(arr), np.array([0, 1]))
    arr = pa.chunked_array([[1, 1], [1]])
    npt.assert_array_equal(_calculate_chunk_offsets(arr), np.array([0, 2]))


def check_valid_in_offsets(
    arr: pa.ChunkedArray, in_offsets: List[Tuple[int, int, int]]
) -> None:
    if arr.num_chunks == 0:
        assert in_offsets == []
        return

    # We always start at the beginning
    assert in_offsets[0][0] == 0
    assert in_offsets[0][1] == 0

    # Overall, the chunk offsets must have the same length as the array
    assert sum(x[2] for x in in_offsets) == len(arr)


@given(data=st.lists(st.lists(st.integers(min_value=0, max_value=10))))
def test_in_chunk_offsets(data: List[List[int]]):
    arr = pa.chunked_array(data, type=pa.int64())
    # Simple case: Passing in the actual chunk offsets should yield a valid selection
    offsets = list(_calculate_chunk_offsets(arr))
    in_offsets = _in_chunk_offsets(arr, offsets)
    check_valid_in_offsets(arr, in_offsets)


def test_combined_in_chunk_offsets():
    a = pa.chunked_array([[]])
    b = pa.chunked_array([[]])
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)
    assert in_a_offsets == [(0, 0, 0)]
    assert in_b_offsets == [(0, 0, 0)]

    a = pa.chunked_array([[1]])
    b = pa.chunked_array([[2]])
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)
    assert in_a_offsets == [(0, 0, 1)]
    assert in_b_offsets == [(0, 0, 1)]

    a = pa.chunked_array([[1, 2], [3, 4, 5]])
    b = pa.chunked_array([[1], [2, 3], [4, 5]])
    in_a_offsets, in_b_offsets = _combined_in_chunk_offsets(a, b)
    assert in_a_offsets == [(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 2)]
    assert in_b_offsets == [(0, 0, 1), (1, 0, 1), (1, 1, 1), (2, 0, 2)]


@pytest.mark.parametrize("data", [[1, 2, 4, 5], [1.0, 0.5, 4.0, 5.0]])
def test_extract_data_buffer_as_np_array(data):
    arr = pa.array(data)
    result = _extract_data_buffer_as_np_array(arr)
    expected = np.array(data)
    npt.assert_array_equal(result, expected)
    result = _extract_data_buffer_as_np_array(arr[2:4])
    expected = np.array(data[2:4])
    npt.assert_array_equal(result, expected)


def assert_content_equals_array(result, expected):
    """Assert that the result is an Arrow structure and the content matches an array."""
    assert isinstance(result, (pa.Array, pa.ChunkedArray))
    if isinstance(result, pa.ChunkedArray):
        result = pa.concat_arrays(result.iterchunks())
    assert result.equals(expected)


def test_merge_valid_bitmaps():
    a = pa.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    b = pa.array([1, 1, 1, None, None, None, 1, 1, 1])

    expected = np.array([0xFF, 0x1], dtype=np.uint8)
    result = _merge_valid_bitmaps(a, a)
    npt.assert_array_equal(result, expected)
    expected = np.array([0xC7, 0x1], dtype=np.uint8)
    result = _merge_valid_bitmaps(a, b)
    npt.assert_array_equal(result, expected)

    expected = np.array([0x1], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(8, 1), a.slice(8, 1))
    npt.assert_array_equal(result, expected)

    expected = np.array([0xF], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(0, 4), a.slice(0, 4))
    npt.assert_array_equal(result, expected)
    expected = np.array([0x7], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(0, 4), b.slice(0, 4))
    npt.assert_array_equal(result, expected)

    expected = np.array([0xF], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 4), a.slice(5, 4))
    npt.assert_array_equal(result, expected)
    expected = np.array([0xE], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 4), b.slice(5, 4))
    npt.assert_array_equal(result, expected)

    expected = np.array([0x3], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), a.slice(5, 2))
    npt.assert_array_equal(result, expected)
    expected = np.array([0x2], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), b.slice(5, 2))
    npt.assert_array_equal(result, expected)

    expected = np.array([0x3], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), a.slice(3, 2))
    npt.assert_array_equal(result, expected)
    expected = np.array([0x0], dtype=np.uint8)
    result = _merge_valid_bitmaps(a.slice(5, 2), b.slice(3, 2))
    npt.assert_array_equal(result, expected)


@settings(deadline=timedelta(milliseconds=1000))
@given(data=st.lists(st.one_of(st.text(), st.none())))
def test_text_cat(data):
    if any("\x00" in x for x in data if x):
        # pytest.skip("pandas cannot handle \\x00 characters in tests")
        # Skip is not working properly with hypothesis
        return
    ser_pd = pd.Series(data, dtype=str)
    arrow_data = pa.array(data, type=pa.string())
    fr_array = fr.FletcherArray(arrow_data)
    ser_fr = pd.Series(fr_array)
    fr_other_array = fr.FletcherArray(arrow_data)
    ser_fr_other = pd.Series(fr_other_array)

    result_pd = ser_pd.str.cat(ser_pd)
    result_fr = ser_fr.fr_text.cat(ser_fr_other)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


def _optional_len(x: Optional[str]) -> int:
    if x is not None:
        return len(x)
    else:
        return 0


@settings(deadline=timedelta(milliseconds=1000))
@given(data=st.lists(st.one_of(st.text(), st.none())))
@pytest.mark.xfail(reason="Not implemented")
def test_text_zfill(data):
    if any("\x00" in x for x in data if x):
        # pytest.skip("pandas cannot handle \\x00 characters in tests")
        # Skip is not working properly with hypothesis
        return
    ser_pd = pd.Series(data, dtype=str)
    max_str_len = ser_pd.map(_optional_len).max()
    if pd.isna(max_str_len):
        max_str_len = 0
    arrow_data = pa.array(data, type=pa.string())
    fr_array = fr.FletcherArray(arrow_data)
    ser_fr = pd.Series(fr_array)

    result_pd = ser_pd.str.zfill(max_str_len + 1)
    result_fr = ser_fr.fr_text.zfill(max_str_len + 1)
    result_fr = result_fr.astype(object)
    # Pandas returns np.nan for NA values in cat, keep this in line
    result_fr[result_fr.isna()] = np.nan
    tm.assert_series_equal(result_fr, result_pd)


@settings(deadline=None)
@given(data=st.lists(st.one_of(st.booleans(), st.none())), skipna=st.booleans())
@example([], False)
@example([], True)
# Test with numpy.array as input.
# This has the caveat that the missing buffer is None.
@example(np.ones(10).astype(bool), False)
@example(np.ones(10).astype(bool), True)
def test_any_op(data, skipna):
    arrow = pa.array(data, type=pa.bool_())
    # https://github.com/pandas-dev/pandas/issues/27709 / https://github.com/pandas-dev/pandas/issues/12863
    pandas = pd.Series(data).astype(float)

    assert any_op(arrow, skipna) == pandas.any(skipna=skipna)

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [data[: len(data) // 2], data[len(data) // 2 :]], type=pa.bool_()
        )
        assert any_op(arrow, skipna) == pandas.any(skipna=skipna)


@given(data=st.lists(st.one_of(st.booleans(), st.none())), skipna=st.booleans())
# Test with numpy.array as input.
# This has the caveat that the missing buffer is None.
@example(np.ones(10).astype(bool), False)
@example(np.ones(10).astype(bool), True)
def test_all_op(data, skipna):
    arrow = pa.array(data, type=pa.bool_())
    # https://github.com/pandas-dev/pandas/issues/27709 / https://github.com/pandas-dev/pandas/issues/12863
    pandas = pd.Series(data).astype(float)

    assert all_op(arrow, skipna) == pandas.all(skipna=skipna)

    # Split in the middle and check whether this still works
    if len(data) > 2:
        arrow = pa.chunked_array(
            [data[: len(data) // 2], data[len(data) // 2 :]], type=pa.bool_()
        )
        assert all_op(arrow, skipna) == pandas.all(skipna=skipna)


@pytest.mark.parametrize(
    ("array", "fill_null_value", "expected"),
    [
        (pa.array([2, 1], type=pa.int16()), -1, np.array([2, 1], dtype=np.int16)),
        (pa.array([2, None], type=pa.int32()), -1, np.array([2, -1], dtype=np.int32)),
        (pa.array([2, None], type=pa.int64()), -1.5, np.array([2, -1], dtype=np.int64)),
        (pa.array([1, None], type=pa.uint8()), 257, np.array([1, 1], dtype=np.uint8)),
        (pa.array([None, None], type=pa.int8()), 5, np.array([5, 5], dtype=np.int8)),
        (pa.array([], type=pa.int8()), 5, np.array([], dtype=np.int8)),
    ],
)
def test_integer_array_to_numpy(array, fill_null_value, expected):
    actual = integer_array_to_numpy(array, fill_null_value)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("array", "indices"),
    [
        (
            pa.array([[k] for k in range(10 ** 4)]),
            np.random.randint(0, 10 ** 4, 10 ** 2),
        ),
        (
            pa.array([[float(k)] for k in range(10 ** 4)]),
            np.random.randint(0, 10 ** 4, 10 ** 2),
        ),
        (
            pa.array(np.random.randint(0, 100, 10) for _ in range(10 ** 4)),
            np.random.randint(0, 10 ** 4, 10 ** 5),
        ),
        (
            pa.LargeListArray.from_arrays(
                [k for k in range(10 ** 4 + 1)], [k for k in range(10 ** 4)]
            ),
            np.random.randint(0, 10 ** 4, 10 ** 2),
        ),
        (
            pa.LargeListArray.from_arrays(
                [k for k in range(10 ** 4 + 1)], [float(k) for k in range(10 ** 4)]
            ),
            np.random.randint(0, 10 ** 4, 10 ** 2),
        ),
        (pa.array([[]]), [0]),
    ],
)
def test_take_indices_on_pyarrow_list(array, indices):
    np.testing.assert_array_equal(
        array.take(pa.array(indices)).to_pylist(),
        take_indices_on_pyarrow_list(array, indices).to_pylist(),
    )
