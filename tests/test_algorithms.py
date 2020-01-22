import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from hypothesis import example, given, settings

# fmt: off
from fletcher._algorithms import all_op, any_op, integer_array_to_numpy, take_indices_on_pyarrow_list

# fmt: on


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
