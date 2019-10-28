# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt
import pyarrow as pa
import pytest

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
    col = pa.Column.from_array("column", arr)

    expected_series_woutname = pd.Series(fr.FletcherArray(arr))
    pdt.assert_series_equal(expected_series_woutname, fr.pandas_from_arrow(arr))
    pdt.assert_series_equal(expected_series_woutname, fr.pandas_from_arrow(col.data))

    expected_series_wname = pd.Series(fr.FletcherArray(arr), name="column")
    pdt.assert_series_equal(expected_series_wname, fr.pandas_from_arrow(col))

    rb = pa.RecordBatch.from_arrays([arr], ["column"])
    expected_df = pd.DataFrame({"column": fr.FletcherArray(arr)})
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(rb))

    table = pa.Table.from_arrays([arr], ["column"])
    pdt.assert_frame_equal(expected_df, fr.pandas_from_arrow(table))
