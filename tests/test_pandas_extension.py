# -*- coding: utf-8 -*-
import datetime
import string
import sys
from collections import namedtuple
from distutils.version import LooseVersion

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import six

from fletcher import FletcherArray, FletcherDtype

from pandas.tests.extension.base import (  # BaseArithmeticOpsTests,; BaseComparisonOpsTests,; BaseNumericReduceTests,
    BaseBooleanReduceTests,
    BaseCastingTests,
    BaseConstructorsTests,
    BaseDtypeTests,
    BaseGetitemTests,
    BaseGroupbyTests,
    BaseInterfaceTests,
    BaseMethodsTests,
    BaseMissingTests,
    BaseNoReduceTests,
    BaseParsingTests,
    BasePrintingTests,
    BaseReshapingTests,
    BaseSetitemTests,
)

if LooseVersion(pd.__version__) >= "0.25.0":
    # imports of pytest fixtures needed for derived unittest classes
    from pandas.tests.extension.conftest import (  # noqa: F401
        as_array,  # noqa: F401
        use_numpy,  # noqa: F401
        groupby_apply_op,  # noqa: F401
        as_frame,  # noqa: F401
        as_series,  # noqa: F401
    )

PANDAS_GE_1_1_0 = LooseVersion(pd.__version__) >= "1.1.0"

FletcherTestType = namedtuple(
    "FletcherTestType",
    [
        "dtype",
        "data",
        "data_missing",
        "data_for_grouping",
        "data_for_sorting",
        "data_missing_for_sorting",
        "data_repeated",
    ],
)

if sys.version_info >= (3, 6):
    from random import choices
else:
    from random import choice

    def choices(seq, k):
        return [choice(seq) for i in range(k)]


xfail_list_setitem_not_implemented = pytest.mark.xfail_by_type_filter(
    [pa.types.is_list], "__setitem__ is not implemented for lists"
)
fail_on_missing_dtype_in_from_sequence = pytest.mark.xfail(
    LooseVersion(pa.__version__) >= "0.10.1dev0",
    reason="Default return type of pa.array([datetime.date]) changed, Pandas tests don't pass the dtype to from_sequence",
)

test_types = [
    FletcherTestType(
        pa.string(),
        ["🙈", "Ö", "Č", "a", "B"] * 20,
        [None, "A"],
        ["B", "B", None, None, "A", "A", "B", "C"],
        ["B", "C", "A"],
        ["B", None, "A"],
        lambda: choices(list(string.ascii_letters), k=10),  # type: ignore
    ),
    FletcherTestType(
        pa.bool_(),
        [True, False, True, True, False] * 20,
        [None, False],
        [True, True, None, None, False, False, True, False],
        [True, False, False],
        [True, None, False],
        lambda: choices([True, False], k=10),
    ),
    FletcherTestType(
        pa.int64(),
        [2, 1, -1, 0, 66] * 20,
        [None, 1],
        [2, 2, None, None, -100, -100, 2, 100],
        [2, 100, -10],
        [2, None, -10],
        lambda: choices(list(range(100)), k=10),
    ),
    FletcherTestType(
        pa.float64(),
        [2.5, 1.0, -1.0, 0, 66.6] * 20,
        [None, 1.1],
        [2.5, 2.5, None, None, -100.1, -100.1, 2.5, 100.1],
        [2.5, 100.99, -10.1],
        [2.5, None, -10.1],
        lambda: choices([2.5, 1.0, -1.0, 0, 66.6], k=10),
    ),
    # Most of the tests fail as assert_extension_array_equal casts to numpy object
    # arrays and on them equality is not defined.
    pytest.param(
        FletcherTestType(
            pa.list_(pa.string()),
            [["B", "C"], ["A"], [None], ["A", "A"], []],
            [None, ["A"]],
            [["B"], ["B"], None, None, ["A"], ["A"], ["B"], ["C"]],
            [["B"], ["C"], ["A"]],
            [["B"], None, ["A"]],
            lambda: choices([["B", "C"], ["A"], [None], ["A", "A"]], k=10),
        ),
        marks=pytest.mark.xfail,
    ),
    FletcherTestType(
        pa.date64(),
        [
            datetime.date(2015, 1, 1),
            datetime.date(2010, 12, 31),
            datetime.date(1970, 1, 1),
            datetime.date(1900, 3, 31),
            datetime.date(1999, 12, 31),
        ]
        * 20,
        [None, datetime.date(2015, 1, 1)],
        [
            datetime.date(2015, 2, 2),
            datetime.date(2015, 2, 2),
            None,
            None,
            datetime.date(2015, 1, 1),
            datetime.date(2015, 1, 1),
            datetime.date(2015, 2, 2),
            datetime.date(2015, 3, 3),
        ],
        [
            datetime.date(2015, 2, 2),
            datetime.date(2015, 3, 3),
            datetime.date(2015, 1, 1),
        ],
        [datetime.date(2015, 2, 2), None, datetime.date(2015, 1, 1)],
        lambda: choices(list(pd.date_range("2010-1-1", "2011-1-1").date), k=10),
    ),
]


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series."""
    return request.param


@pytest.fixture(params=test_types)
def fletcher_type(request):
    return request.param


@pytest.fixture
def dtype(fletcher_type):
    return FletcherDtype(fletcher_type.dtype)


@pytest.fixture
def data(fletcher_type):
    return FletcherArray(fletcher_type.data, dtype=fletcher_type.dtype)


@pytest.fixture
def data_missing(fletcher_type):
    return FletcherArray(fletcher_type.data_missing, dtype=fletcher_type.dtype)


@pytest.fixture
def data_repeated(fletcher_type):
    """Return different versions of data for count times."""
    pass  # noqa

    def gen(count):
        for _ in range(count):
            yield FletcherArray(
                fletcher_type.data_repeated(), dtype=fletcher_type.dtype
            )

    yield gen


@pytest.fixture
def data_for_grouping(fletcher_type):
    """Fixture with data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    return FletcherArray(fletcher_type.data_for_grouping, dtype=fletcher_type.dtype)


@pytest.fixture
def data_for_sorting(fletcher_type):
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return FletcherArray(fletcher_type.data_for_sorting, dtype=fletcher_type.dtype)


@pytest.fixture
def data_missing_for_sorting(fletcher_type):
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return FletcherArray(
        fletcher_type.data_missing_for_sorting, dtype=fletcher_type.dtype
    )


class TestBaseCasting(BaseCastingTests):
    @pytest.mark.xfail(six.PY2, reason="Cast of UTF8 to `str` fails in py2.")
    def test_astype_str(self, data):
        BaseCastingTests.test_astype_str(self, data)


class TestBaseConstructors(BaseConstructorsTests):
    @pytest.mark.xfail(reason="Tries to construct dtypes with np.dtype")
    def test_from_dtype(self, data):
        if pa.types.is_string(data.dtype.arrow_dtype):
            pytest.xfail(
                "String construction is failing as Pandas wants to pass the FletcherDtype to NumPy"
            )
        BaseConstructorsTests.test_from_dtype(self, data)


class TestBaseDtype(BaseDtypeTests):
    pass


class TestBaseGetitemTests(BaseGetitemTests):
    def test_take_non_na_fill_value(self, data_missing):
        if pa.types.is_integer(data_missing.dtype.arrow_dtype):
            pytest.mark.xfail(reasion="Take is not yet correctly implemented for ints")
        else:
            BaseGetitemTests.test_take_non_na_fill_value(self, data_missing)

    def test_reindex_non_na_fill_value(self, data_missing):
        if pa.types.is_integer(data_missing.dtype.arrow_dtype):
            pytest.mark.xfail(reasion="Take is not yet correctly implemented for ints")
        else:
            BaseGetitemTests.test_reindex_non_na_fill_value(self, data_missing)

    def test_take_series(self, data):
        BaseGetitemTests.test_take_series(self, data)

    def test_loc_iloc_frame_single_dtype(self, data):
        if pa.types.is_string(data.dtype.arrow_dtype):
            pytest.mark.xfail(
                reason="https://github.com/pandas-dev/pandas/issues/27673"
            )
        else:
            BaseGetitemTests.test_loc_iloc_frame_single_dtype(self, data)

    @pytest.mark.skip
    def test_reindex(self):
        # No longer available in master and fails with pandas 0.23.1
        # due to a dtype assumption that does not hold for Arrow
        pass


class TestBaseGroupbyTests(BaseGroupbyTests):
    @pytest.mark.parametrize("as_index", [True, False])
    def test_groupby_extension_agg(self, as_index, data_for_grouping):
        if (
            pa.types.is_integer(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_floating(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype)
        ):
            pytest.mark.xfail(reasion="ExtensionIndex is not yet implemented")
        else:
            BaseGroupbyTests.test_groupby_extension_agg(
                self, as_index, data_for_grouping
            )

    def test_groupby_extension_no_sort(self, data_for_grouping):
        if (
            pa.types.is_integer(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_floating(data_for_grouping.dtype.arrow_dtype)
            or pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype)
        ):
            pytest.mark.xfail(reasion="ExtensionIndex is not yet implemented")
        else:
            BaseGroupbyTests.test_groupby_extension_no_sort(self, data_for_grouping)

    def test_groupby_extension_transform(self, data_for_grouping):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            valid = data_for_grouping[~data_for_grouping.isna()]
            df = pd.DataFrame({"A": [1, 1, 3, 3, 1, 4], "B": valid})

            result = df.groupby("B").A.transform(len)
            # Expected grouping is different as we only have two non-null values
            expected = pd.Series([3, 3, 3, 3, 3, 3], name="A")

            self.assert_series_equal(result, expected)
        else:
            BaseGroupbyTests.test_groupby_extension_transform(self, data_for_grouping)


class TestBaseInterfaceTests(BaseInterfaceTests):
    @pytest.mark.xfail(
        reason="view or self[:] returns a shallow copy in-place edits are not backpropagated"
    )
    def test_view(self, data):
        pass


class TestBaseMethodsTests(BaseMethodsTests):

    # https://github.com/pandas-dev/pandas/issues/22843
    @pytest.mark.skip(reason="Incorrect expected")
    @pytest.mark.parametrize("dropna", [True, False])
    def test_value_counts(self, all_data, dropna, dtype):
        pass

    def test_combine_le(self, data_repeated):
        # GH 20825
        # Test that combine works when doing a <= (le) comparison
        # Fletcher returns 'fletcher[bool]' instead of np.bool as dtype
        orig_data1, orig_data2 = data_repeated(2)
        s1 = pd.Series(orig_data1)
        s2 = pd.Series(orig_data2)
        result = s1.combine(s2, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            orig_data1._from_sequence(
                [a <= b for (a, b) in zip(list(orig_data1), list(orig_data2))]
            )
        )
        self.assert_series_equal(result, expected)

        val = s1.iloc[0]
        result = s1.combine(val, lambda x1, x2: x1 <= x2)
        expected = pd.Series(
            orig_data1._from_sequence([a <= val for a in list(orig_data1)])
        )
        self.assert_series_equal(result, expected)

    def test_combine_add(self, data_repeated, dtype):
        if dtype.name == "fletcher[date64[ms]]":
            pytest.skip(
                "unsupported operand type(s) for +: 'datetime.date' and 'datetime.date"
            )
        else:
            BaseMethodsTests.test_combine_add(self, data_repeated)

    def test_argsort(self, data_for_sorting):
        if pa.types.is_boolean(data_for_sorting.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_argsort(self, data_for_sorting)

    @pytest.mark.parametrize("ascending", [True, False])
    def test_sort_values(self, data_for_sorting, ascending):
        if pa.types.is_boolean(data_for_sorting.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_sort_values(self, data_for_sorting, ascending)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize(self, data_for_grouping, na_sentinel):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_factorize(self, data_for_grouping, na_sentinel)

    @pytest.mark.parametrize("na_sentinel", [-1, -2])
    def test_factorize_equivalence(self, data_for_grouping, na_sentinel):
        if pa.types.is_boolean(data_for_grouping.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_factorize_equivalence(
                self, data_for_grouping, na_sentinel
            )

    @pytest.mark.parametrize("box", [pd.Series, lambda x: x])
    @pytest.mark.parametrize("method", [lambda x: x.unique(), pd.unique])
    def test_unique(self, data, box, method):
        BaseMethodsTests.test_unique(self, data, box, method)

    def test_searchsorted(self, data_for_sorting, as_series):  # noqa: F811
        if pa.types.is_boolean(data_for_sorting.dtype.arrow_dtype):
            pytest.skip("Boolean has too few values for this test")
        else:
            BaseMethodsTests.test_searchsorted(self, data_for_sorting, as_series)


class TestBaseMissingTests(BaseMissingTests):
    @fail_on_missing_dtype_in_from_sequence
    def test_fillna_series(self, data_missing):
        BaseMissingTests.test_fillna_series(self, data_missing)

    @fail_on_missing_dtype_in_from_sequence
    @pytest.mark.parametrize("method", ["ffill", "bfill"])
    def test_fillna_series_method(self, data_missing, method):
        BaseMissingTests.test_fillna_series_method(self, data_missing, method)

    def test_fillna_frame(self, data_missing):
        BaseMissingTests.test_fillna_frame(self, data_missing)


class TestBaseReshapingTests(BaseReshapingTests):
    def test_concat_mixed_dtypes(self, data, dtype):
        if dtype.name in ["fletcher[int64]", "fletcher[double]", "fletcher[bool]"]:
            # https://github.com/pandas-dev/pandas/issues/21792
            pytest.skip("pd.concat(int64, fletcher[int64] yields int64")
        else:
            BaseReshapingTests.test_concat_mixed_dtypes(self, data)

    def test_concat_columns(self, data, na_value):
        BaseReshapingTests.test_concat_columns(self, data, na_value)

    def test_align(self, data, na_value):
        BaseReshapingTests.test_align(self, data, na_value)

    def test_align_frame(self, data, na_value):
        BaseReshapingTests.test_align_frame(self, data, na_value)

    def test_align_series_frame(self, data, na_value):
        BaseReshapingTests.test_align_series_frame(self, data, na_value)

    def test_merge(self, data, na_value):
        BaseReshapingTests.test_merge(self, data, na_value)


class TestBaseSetitemTests(BaseSetitemTests):
    @xfail_list_setitem_not_implemented
    def test_setitem_scalar_series(self, data, box_in_series):
        BaseSetitemTests.test_setitem_scalar_series(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_sequence(self, data, box_in_series):
        BaseSetitemTests.test_setitem_sequence(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_empty_indxer(self, data, box_in_series):
        BaseSetitemTests.test_setitem_empty_indxer(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_sequence_broadcasts(self, data, box_in_series):
        BaseSetitemTests.test_setitem_sequence_broadcasts(self, data, box_in_series)

    @pytest.mark.parametrize("setter", ["loc", "iloc"])
    @xfail_list_setitem_not_implemented
    def test_setitem_scalar(self, data, setter):
        BaseSetitemTests.test_setitem_scalar(self, data, setter)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_scalar_mixed(self, data):
        BaseSetitemTests.test_setitem_loc_scalar_mixed(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_scalar_single(self, data):
        BaseSetitemTests.test_setitem_loc_scalar_single(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_scalar_multiple_homogoneous(self, data):
        BaseSetitemTests.test_setitem_loc_scalar_multiple_homogoneous(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_iloc_scalar_mixed(self, data):
        BaseSetitemTests.test_setitem_iloc_scalar_mixed(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_iloc_scalar_single(self, data):
        BaseSetitemTests.test_setitem_iloc_scalar_single(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_iloc_scalar_multiple_homogoneous(self, data):
        BaseSetitemTests.test_setitem_iloc_scalar_multiple_homogoneous(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_nullable_mask(self, data):
        if not PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_nullable_mask(self, data)

    @pytest.mark.parametrize("as_callable", [True, False])
    @pytest.mark.parametrize("setter", ["loc", None])
    @xfail_list_setitem_not_implemented
    def test_setitem_mask_aligned(self, data, as_callable, setter):
        BaseSetitemTests.test_setitem_mask_aligned(self, data, as_callable, setter)

    @pytest.mark.parametrize("setter", ["loc", None])
    @xfail_list_setitem_not_implemented
    def test_setitem_mask_broadcast(self, data, setter):
        BaseSetitemTests.test_setitem_mask_broadcast(self, data, setter)

    @xfail_list_setitem_not_implemented
    def test_setitem_slice(self, data, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_slice(self, data, box_in_series)

    @xfail_list_setitem_not_implemented
    def test_setitem_loc_iloc_slice(self, data):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_loc_iloc_slice(self, data)

    @xfail_list_setitem_not_implemented
    def test_setitem_slice_array(self, data):
        BaseSetitemTests.test_setitem_slice_array(self, data)

    @xfail_list_setitem_not_implemented
    @pytest.mark.parametrize(
        "mask",
        [
            np.array([True, True, True, False, False]),
            pd.array([True, True, True, False, False], dtype="boolean"),
            pd.array([True, True, True, pd.NA, pd.NA], dtype="boolean"),
        ],
        ids=["numpy-array", "boolean-array", "boolean-array-na"],
    )
    def test_setitem_mask(self, data, mask, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_mask(self, data, mask, box_in_series)

    @pytest.mark.xfail(reason="Views don't update their parent #96")
    def test_setitem_preserves_views(self, data):
        pass

    @xfail_list_setitem_not_implemented
    def test_setitem_mask_boolean_array_with_na(self, data, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_mask_boolean_array_with_na(
                self, data, box_in_series
            )

    @pytest.mark.parametrize(
        "idx",
        [[0, 1, 2], pd.array([0, 1, 2], dtype="Int64"), np.array([0, 1, 2])],
        ids=["list", "integer-array", "numpy-array"],
    )
    @pytest.mark.xfail(reason="https://github.com/xhochy/fletcher/issues/110")
    def test_setitem_integer_array(self, data, idx, box_in_series):
        if PANDAS_GE_1_1_0:
            BaseSetitemTests.test_setitem_integer_array(self, data, idx, box_in_series)


class TestBaseParsingTests(BaseParsingTests):
    @pytest.mark.parametrize("engine", ["c", "python"])
    def test_EA_types(self, engine, data):
        pytest.mark.xfail(
            "pandas doesn't yet support registering ExtentionDtypes via a pattern"
        )


class TestBasePrintingTests(BasePrintingTests):
    pass


class TestBaseBooleanReduceTests(BaseBooleanReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_boolean_reductions, skipna):
        if pa.types.is_boolean(data.dtype.arrow_dtype):
            BaseBooleanReduceTests.test_reduce_series(
                self, data, all_boolean_reductions, skipna
            )
        else:
            pytest.skip("Boolean reductions are only tested with boolean types")


class TestBaseNoReduceTests(BaseNoReduceTests):
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna):
        # TODO: Implement for numeric types and then skip this test
        BaseNoReduceTests.test_reduce_series_numeric(
            self, data, all_numeric_reductions, skipna
        )

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        if pa.types.is_boolean(data.dtype.arrow_dtype):
            pytest.skip("BooleanArray does define boolean reductions, so don't raise")
        else:
            BaseNoReduceTests.test_reduce_series_boolean(
                self, data, all_boolean_reductions, skipna
            )


# TODO: Implement
# class TestBaseNumericReduceTests(BaseNumericReduceTests):
#    pass


# TODO: Implement
# class TestBaseComparisonOpsTests(BaseComparisonOpsTests):
#    pass

# TODO: Implement
# class TestBaseArithmeticOpsTests(BaseArithmeticOpsTests):
#     pass
