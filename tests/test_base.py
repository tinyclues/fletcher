# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest
from tests.test_pandas_integration import test_array_chunked_nulls  # noqa: F401

import fletcher as fr
from fletcher.base import to_numpy


@pytest.fixture
def array_inhom_chunks():
    chunk1 = pa.array(list("abc"), pa.string())
    chunk2 = pa.array(list("12345"), pa.string())
    chunk3 = pa.array(list("Z"), pa.string())
    chunked_array = pa.chunked_array([chunk1, chunk2, chunk3])
    return fr.FletcherArray(chunked_array)


@pytest.mark.parametrize(
    "indices, expected",
    [
        (np.array(range(3)), np.full(3, 0)),
        (np.array(range(3, 8)), np.full(5, 1)),
        (np.array([8]), np.array([2])),
        (np.array([0, 1, 5, 7, 8]), np.array([0, 0, 1, 1, 2])),
        (np.array([5, 8, 0, 7, 1]), np.array([1, 2, 0, 1, 0])),
    ],
)
def test_get_chunk_indexer(array_inhom_chunks, indices, expected):

    actual = array_inhom_chunks._get_chunk_indexer(indices)
    npt.assert_array_equal(actual, expected)


def test_fletcherarray_constructor():
    with pytest.raises(ValueError):
        fr.FletcherArray(None)


def test_to_numpy():
    with pytest.raises(NotImplementedError):
        to_numpy(pa.array(["a", "b", "c"], pa.string()), None)
    with pytest.raises(ValueError):
        to_numpy(pa.array([2, 1, None]), None)

    npt.assert_array_equal(to_numpy(pa.array([2, 1, None]), -1), [2, 1, -1])
    npt.assert_array_equal(to_numpy(pa.array([2, 1, None]), -1.2), [2, 1, -1])
    npt.assert_array_equal(to_numpy(pa.array([2, 1.4, None]), -1.2), [2, 1.4, -1.2])
    npt.assert_array_equal(to_numpy(pa.array([2, 1, 2]), -1), [2, 1, 2])
    npt.assert_array_equal(
        to_numpy(pa.array([None, None], type=pa.int32()), -1.2), [-1, -1]
    )
    npt.assert_array_equal(to_numpy(pa.array([], type=pa.float16()), -1.2), [])


def test_pandas_from_arrow():
    arr = pa.array(["a", "b", "c"], pa.string())

    expected_series_woutname = pd.Series(fr.FletcherArray(arr))
    pdt.assert_series_equal(expected_series_woutname, fr.pandas_from_arrow(arr))

    rb = pa.RecordBatch.from_arrays([arr], ["column"])
    expected_df = pd.DataFrame({"column": fr.FletcherArray(arr)})
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(rb))

    table = pa.Table.from_arrays([arr], ["column"])
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(table))


def test_take_on_concatenated_chunks():
    test = [[1, 2, 8, 3], [4, 1, 5, 6], [7, 8, 9]]
    indices = np.array([4, 2, 8])
    expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
    result = fr.FletcherArray(pa.chunked_array(test))._take_on_concatenated_chunks(
        indices
    )
    npt.assert_array_equal(expected_result, result)


def test_take_on_concatenated_chunks_with_many_chunks():
    test = [[1, 2, 3] for _ in range(100)]
    fr_test = fr.FletcherArray(pa.chunked_array(test))
    indices1 = np.array([(30 * k + (k % 3)) for k in range(0, 10)])
    indices2 = np.array([2, 5] * 100)
    for indices in [indices1, indices2]:
        expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
        result = fr_test._take_on_concatenated_chunks(indices)
        npt.assert_array_equal(expected_result, result)


def test_take_on_chunks():
    test = [[1, 2, 8, 3], [4, 1, 5, 6], [7, 8, 9]]
    indices = np.array([2, 4, 8])
    limits_idx = np.array([0, 1, 2, 3])
    cum_lengths = np.array([0, 4, 8])
    expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
    result = fr.FletcherArray(pa.chunked_array(test))._take_on_chunks(
        indices, limits_idx=limits_idx, cum_lengths=cum_lengths
    )
    npt.assert_array_equal(expected_result, result)


def test_take_on_chunks_with_many_chunks():
    test = [[1, 2, 3] for _ in range(100)]
    fr_test = fr.FletcherArray(pa.chunked_array(test))

    indices1 = np.array([(30 * k + (k % 3)) for k in range(0, 10)])
    # bins will be already sorted
    indices2 = np.array([2, 5] * 100)
    # bins will have to be sorted

    limits_idx1 = np.array([0] + [k // 10 for k in range(10, 110)])
    limits_idx2 = np.array([0] + [100] + [200] * 99)

    sort_idx1 = None
    sort_idx2 = np.array(
        [2 * k for k in range(0, 100)] + [2 * k + 1 for k in range(100)]
    )

    indices2 = indices2[sort_idx2]

    cum_lengths = np.array([3 * k for k in range(100)])

    for indices, limits_idx, cum_lengths, sort_idx in [
        (indices1, limits_idx1, cum_lengths, sort_idx1),
        (indices2, limits_idx2, cum_lengths, sort_idx2),
    ]:
        expected_result = fr.FletcherArray([np.concatenate(test)[e] for e in indices])
        result = fr_test._take_on_chunks(
            indices, limits_idx=limits_idx, cum_lengths=cum_lengths, sort_idx=sort_idx
        )
        npt.assert_array_equal(expected_result, result)


def test_indices_dtype():
    arr1 = fr.FletcherArray(np.zeros(np.iinfo(np.int32()).max + 1))
    arr2 = fr.FletcherArray(np.zeros(np.iinfo(np.int32()).max + 2))
    for arr in [arr1, arr2]:
        npt.assert_equal(
            len(arr) - 1, np.array([len(arr) - 1], dtype=arr._indices_dtype)[0]
        )
    npt.assert_equal(arr1._indices_dtype, np.dtype(np.int32))
    npt.assert_equal(arr2._indices_dtype, np.dtype(np.int64))


def test_take():
    test = [[1, 2, 8, 3], [4, 1, 5, 6], [7, 8, 9]]
    indices = [4, 2, 8] * 100
    fr_test = fr.FletcherArray(pa.chunked_array(test))
    result = fr_test.take(indices)
    expected_result = fr.FletcherArray(
        pa.chunked_array([[4, 8, 7] for _ in range(100)])
    )
    npt.assert_array_equal(expected_result, result)


def test_reduce_sum():
    test = [[1, 2, 3], [1, 2, None]]

    fr_test_int = fr.FletcherArray(pa.chunked_array(test), dtype=pa.int64())
    fr_test_float = fr.FletcherArray(pa.chunked_array(test), dtype=pa.float64())

    result_int = fr_test_int._reduce("sum")
    result_float = fr_test_float._reduce("sum")

    expected_result_int = 9
    expected_result_float = 9.0

    assert result_int == expected_result_int
    assert result_float == expected_result_float

    assert fr.FletcherArray([], dtype=pa.int32())._reduce("sum") == 0


def test_reduce_mean():
    test = [[1, 2, 3], [1, 2, None]]
    fr_test_int = fr.FletcherArray(pa.chunked_array(test), dtype=pa.int64())
    fr_test_float = fr.FletcherArray(pa.chunked_array(test), dtype=pa.float64())
    result_int = fr_test_int._reduce("mean")
    result_float = fr_test_float._reduce("mean")
    expected_result = 9 / 5
    assert result_int == expected_result
    assert result_float == expected_result


def test_reduce_max_min():
    test = [[1, 2, 3], [-23, 75, None]]

    fr_test_int = fr.FletcherArray(pa.chunked_array(test), dtype=pa.int64())
    fr_test_float = fr.FletcherArray(pa.chunked_array(test), dtype=pa.float64())

    result_int_max = fr_test_int._reduce("max")
    result_int_min = fr_test_int._reduce("min")

    result_float_max = fr_test_float._reduce("max")
    result_float_min = fr_test_float._reduce("min")

    expected_result_int_max = 75
    expected_result_int_min = -23

    expected_result_float_max = 75.0
    expected_result_float_min = -23.0

    assert result_int_max == expected_result_int_max
    assert result_int_min == expected_result_int_min
    assert result_float_max == expected_result_float_max
    assert result_float_min == expected_result_float_min


class TestArrowArrayProtocol:
    def test_pa_array(self, array_inhom_chunks):
        npt.assert_array_equal(array_inhom_chunks.offsets, [0, 3, 8])

        expected = pa.concat_arrays(array_inhom_chunks.data.iterchunks())
        real = pa.array(array_inhom_chunks)
        assert isinstance(real, pa.Array)

        assert real.equals(expected)

        if pa.__version__ < "0.15":
            npt.assert_array_equal(array_inhom_chunks.offsets, [0, 3, 8])
        else:
            npt.assert_array_equal(array_inhom_chunks.offsets, [0])

    def test_arrow_array_modifies_data(self, test_array_chunked_nulls):  # noqa: F811
        expected = pa.concat_arrays(test_array_chunked_nulls.data.iterchunks())
        id1 = id(test_array_chunked_nulls.data)
        real = test_array_chunked_nulls.__arrow_array__()

        assert id1 != id(test_array_chunked_nulls.data)
        assert real.equals(expected)

    def test_arrow_array_types(self):  # noqa: F811
        fr_arr = fr.FletcherArray(pa.array([3, None, 4.4]))
        # non-safe casting
        assert fr_arr.__arrow_array__(type=pa.int64()).equals(pa.array([3, None, 4]))
        assert fr_arr.data.chunk(0).equals(pa.array([3, None, 4.4]))

        fr_arr = fr.FletcherArray(pa.array(["3", "-2", "4.4"]))
        # non-safe casting
        assert fr_arr.__arrow_array__(type=pa.float64()).equals(pa.array([3, -2, 4.4]))
        assert fr_arr.data.chunk(0).equals(pa.array(["3", "-2", "4.4"]))
