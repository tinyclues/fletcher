# -*- coding: utf-8 -*-

import operator

import pytest

# More information about the pandas extension interface tests can be found here
# https://github.com/pandas-dev/pandas/blob/master/pandas/tests/extension/base/__init__.py

# TODO: Import them using pytest_plugins = ['pandas.tests.extention.fixtures'], see also https://github.com/pandas-dev/pandas/issues/23664

_all_arithmetic_operators = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    """Fixture for dunder names for common arithmetic operations."""
    return request.param


@pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
def all_compare_operators(request):
    """
    Fixture for dunder names for common compare operations.

    * >=
    * >
    * ==
    * !=
    * <
    * <=
    """
    return request.param


@pytest.fixture(params=["__le__", "__lt__", "__ge__", "__gt__"])
def compare_operators_no_eq_ne(request):
    """
    Fixture for dunder names for compare operations except == and !=.

    * >=
    * >
    * <
    * <=
    """
    return request.param


_all_numeric_reductions = [
    "sum",
    "max",
    "min",
    "mean",
    "prod",
    "std",
    "var",
    "median",
    "kurt",
    "skew",
]


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    """Fixture for numeric reduction names."""
    return request.param


_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    """Fixture for boolean reduction names."""
    return request.param


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'."""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def na_cmp():
    """Binary operator for comparing NA values.

    Should return a function of two arguments that returns
    True if both arguments are (scalar) NA for your type.

    By default, uses ``operator.is_``
    """
    return operator.is_


@pytest.fixture
def na_value():
    """Fixture for the scalar missing value for this type. Default 'None'."""
    return None
