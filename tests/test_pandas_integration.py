# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as tm
import pyarrow as pa
import pytest
from pandas.core.internals import ExtensionBlock

import fletcher as fr

# These unit tests are based upon the work done in Cyberpandas (see NOTICE):
#   https://github.com/ContinuumIO/cyberpandas/blob/master/cyberpandas/test_ip_pandas.py


TEST_LIST = ["Test", "string", None]
TEST_ARRAY = pa.array(TEST_LIST, type=pa.string())
TEST_ARRAY_CHUNKED = pa.chunked_array(
    [pa.array(["c", "b", "a", None] * 2), pa.array(["d", None, "b", "a"] * 2)]
)


@pytest.fixture
def test_array_chunked():
    return pa.chunked_array([pa.array(TEST_LIST) for _ in range(10)])


@pytest.fixture(
    params=["all", "all_float", "some_in_all_chunks", "only_in_some_chunk", "none"]
)
def test_array_chunked_nulls(request):
    case_dict = {
        "all": pa.chunked_array([pa.array([None] * 4) for _ in range(10)]),
        "all_float": pa.chunked_array(
            [pa.array([None] * 4, type=pa.float32()) for _ in range(10)]
        ),
        "some_in_all_chunks": pa.chunked_array(
            [pa.array(["a", "b", None] * 4), pa.array(["a", None, "b"] * 4)]
        ),
        "only_in_some_chunk": pa.chunked_array(
            [
                pa.array(["a", "x"]),
                pa.array(["a", "b", None] * 4),
                pa.array(["a", "b"] * 4),
            ]
        ),
        "none": pa.chunked_array([pa.array(["a", "b"] * 4) for _ in range(10)]),
    }
    return fr.FletcherArray(case_dict[request.param])


# ----------------------------------------------------------------------------
# Block Methods
# ----------------------------------------------------------------------------


def test_concatenate_blocks():
    v1 = fr.FletcherArray(TEST_ARRAY)
    s = pd.Series(v1, index=pd.RangeIndex(3), fastpath=True)
    result = pd.concat([s, s], ignore_index=True)
    expected = pd.Series(
        fr.FletcherArray(pa.array(["Test", "string", None, "Test", "string", None]))
    )
    tm.assert_series_equal(result, expected)


# ----------------------------------------------------------------------------
# Public Constructors
# ----------------------------------------------------------------------------


def test_series_constructor():
    v = fr.FletcherArray(TEST_ARRAY)
    result = pd.Series(v)
    assert result.dtype == v.dtype
    assert isinstance(result._data.blocks[0], ExtensionBlock)


def test_dataframe_constructor():
    v = fr.FletcherArray(TEST_ARRAY)
    df = pd.DataFrame({"A": v})
    assert isinstance(df.dtypes["A"], fr.FletcherDtype)
    assert df.shape == (3, 1)

    # Test some calls to typical DataFrame functions
    str(df)
    df.info()


def test_dataframe_from_series_no_dict():
    s = pd.Series(fr.FletcherArray(TEST_ARRAY))
    result = pd.DataFrame(s)
    expected = pd.DataFrame({0: s})
    tm.assert_frame_equal(result, expected)

    s = pd.Series(fr.FletcherArray(TEST_ARRAY), name="A")
    result = pd.DataFrame(s)
    expected = pd.DataFrame({"A": s})
    tm.assert_frame_equal(result, expected)


def test_dataframe_from_series():
    s = pd.Series(fr.FletcherArray(TEST_ARRAY))
    c = pd.Series(pd.Categorical(["a", "b"]))
    result = pd.DataFrame({"A": s, "B": c})
    assert isinstance(result.dtypes["A"], fr.FletcherDtype)


def test_getitem_scalar():
    ser = pd.Series(fr.FletcherArray(TEST_ARRAY))
    result = ser[1]
    assert result == "string"


def test_getitem_slice():
    ser = pd.Series(fr.FletcherArray(TEST_ARRAY))
    result = ser[1:]
    expected = pd.Series(fr.FletcherArray(TEST_ARRAY[1:]), index=range(1, 3))
    tm.assert_series_equal(result, expected)


def test_setitem_scalar():
    ser = pd.Series(fr.FletcherArray(TEST_ARRAY))
    ser[1] = "other_string"
    expected = pd.Series(fr.FletcherArray(pa.array(["Test", "other_string", None])))
    tm.assert_series_equal(ser, expected)


def test_isnull():
    df = pd.DataFrame({"A": fr.FletcherArray(TEST_ARRAY)})

    tm.assert_series_equal(df["A"].isnull(), pd.Series([False, False, True], name="A"))


def test_isna_empty():
    np.testing.assert_array_equal(
        fr.FletcherArray(pa.chunked_array([[], [None], [1]], type=pa.int32())).isna(),
        np.array([True, False]),
    )


def test_set_index():
    pd.DataFrame({"index": [3, 2, 1], "A": fr.FletcherArray(TEST_ARRAY)}).set_index(
        "index"
    )


def test_copy():
    df = pd.DataFrame({"A": fr.FletcherArray(TEST_ARRAY)})
    df["A"].copy()


def test_nbytes():
    array = fr.FletcherArray(pa.array(["A", None, "CC"]))
    # Minimal storage usage:
    # 1 byte for the valid bitmap
    # 4 bytes for the offset array
    # 3 bytes for the actual string content
    assert array.nbytes >= 8


def test_series_attributes():
    s = pd.Series(fr.FletcherArray(TEST_ARRAY))
    assert s.ndim == 1
    assert s.size == 3
    assert s.values is not None
    assert (s.T == s).all()
    assert s.memory_usage() > 8


def test_isna():
    s = pd.Series(fr.FletcherArray(TEST_ARRAY))
    expected = pd.Series([False, False, True])
    tm.assert_series_equal(s.isna(), expected)
    tm.assert_series_equal(s.notna(), ~expected)


def test_isna_chunked(test_array_chunked_nulls):
    fs = pd.Series(test_array_chunked_nulls)
    ps = pd.Series(np.array(test_array_chunked_nulls))
    tm.assert_series_equal(fs.isna(), ps.isna())
    tm.assert_series_equal(fs.notna(), ps.notna())


def test_np_asarray():
    s = pd.Series(fr.FletcherArray(TEST_ARRAY))
    expected = np.asarray(TEST_LIST)
    npt.assert_array_equal(np.asarray(s), expected)


def test_astype_object():
    s = pd.Series(fr.FletcherArray(TEST_ARRAY))
    expected = pd.Series(TEST_LIST)
    tm.assert_series_equal(s.astype(object), expected)


def test_factorize():
    arr = fr.FletcherArray(TEST_ARRAY)
    labels, uniques = arr.factorize()
    expected_labels, expected_uniques = pd.factorize(arr.astype(object))

    assert isinstance(uniques, fr.FletcherArray)

    uniques = uniques.astype(object)
    npt.assert_array_equal(labels, expected_labels)
    npt.assert_array_equal(uniques, expected_uniques)


def test_factorize_many_chunks():
    arr = fr.FletcherArray(TEST_ARRAY_CHUNKED)
    labels, uniques = arr.factorize()
    expected_labels, expected_uniques = pd.factorize(arr.astype(object))

    assert isinstance(uniques, fr.FletcherArray)

    uniques = uniques.astype(object)
    npt.assert_array_equal(labels, expected_labels)
    npt.assert_array_equal(uniques, expected_uniques)


def test_unique():
    arr = fr.FletcherArray(TEST_ARRAY)
    uniques = arr.unique()
    expected_uniques = pd.unique(arr.astype(object))

    assert isinstance(uniques, fr.FletcherArray)

    uniques = uniques.astype(object)
    npt.assert_array_equal(uniques, expected_uniques)


def test_unique_many_chunks():
    arr = fr.FletcherArray(TEST_ARRAY_CHUNKED)
    uniques = arr.unique()
    expected_uniques = pd.unique(arr.astype(object))

    assert isinstance(uniques, fr.FletcherArray)

    uniques = uniques.astype(object)
    npt.assert_array_equal(uniques, expected_uniques)


def test_groupby():
    arr = fr.FletcherArray(["a", "a", "b", None])
    df = pd.DataFrame({"str": arr, "int": [10, 5, 24, 6]})
    result = df.groupby("str").sum()

    expected = pd.DataFrame({"int": [15, 24]}, index=pd.Index(["a", "b"], name="str"))
    tm.assert_frame_equal(result, expected)


def test_groupby_numeric():
    arr_str = fr.FletcherArray(["a", "a", "b", None, "a", "b"])
    arr_num = fr.FletcherArray([1, None, 1, -2, 0, 2])

    df = pd.DataFrame({"str": arr_str, "fr_int": arr_num, "int": np.array(arr_num)})

    result_sum = df.groupby("str").sum()

    expected_sum = pd.DataFrame(
        {"fr_int": fr.FletcherArray([1, 3]), "int": [1.0, 3.0]},
        index=pd.Index(["a", "b"], name="str"),
    )
    tm.assert_frame_equal(result_sum, expected_sum)


@pytest.mark.parametrize("kind", ["quicksort", "mergesort", "heapsort"])
def test_argsort(test_array_chunked_nulls, kind):
    s = pd.Series(test_array_chunked_nulls)
    result = s.argsort(kind=kind)
    expected = s.astype(object).argsort(kind=kind)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("kind", ["quicksort", "mergesort", "heapsort"])
def test_argsort_many_chunks(test_array_chunked_nulls, kind):
    s = pd.Series(test_array_chunked_nulls)
    result = s.argsort(kind=kind)
    expected = s.astype(object).argsort(kind=kind)
    tm.assert_series_equal(result, expected)


def test_fillna_chunked(test_array_chunked):
    ser = pd.Series(fr.FletcherArray(test_array_chunked))
    ser = ser.fillna("filled")

    expected_list = TEST_LIST[:2] + ["filled"]
    chunks = []
    for _ in range(10):
        chunks.append(pa.array(expected_list))
    chunked_exp = pa.chunked_array(chunks)
    expected = pd.Series(fr.FletcherArray(chunked_exp))

    tm.assert_series_equal(ser, expected)


def test_setitem_chunked(test_array_chunked):
    ser = pd.Series(fr.FletcherArray(test_array_chunked))
    new_val = "new_value"
    old_val = ser[15]
    assert new_val != old_val
    ser[15] = new_val
    assert new_val == ser[15]


def test_setitem_chunked_bool_index(test_array_chunked):
    ser = pd.Series(fr.FletcherArray(test_array_chunked))
    bool_index = np.full(len(ser), False)
    bool_index[15] = True
    ser[bool_index] = "bool_value"
    assert ser[15] == "bool_value"


@pytest.mark.parametrize("indices", [[10, 15], [10, 11]])
def test_setitem_chunked_int_index(indices, test_array_chunked):
    ser = pd.Series(fr.FletcherArray(test_array_chunked))
    integer_index = indices
    ser[integer_index] = ["int", "index"]
    assert ser[indices[0]] == "int"
    assert ser[indices[1]] == "index"


def test_max_min_with_groupby():
    df = pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3, 1, 4],
            "B": fr.FletcherArray(list(range(8))),
            "C": list(range(8)),
        }
    )
    df_result_max = df.groupby("A").max()
    df_result_min = df.groupby("A").min()

    npt.assert_array_equal(df_result_max["B"], df_result_max["C"])
    npt.assert_array_equal(df_result_min["B"], df_result_min["C"])
