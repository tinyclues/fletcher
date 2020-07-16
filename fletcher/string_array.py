# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from typing import Optional

import numpy as np
import pandas as pd

from ._algorithms import _endswith, _startswith
from ._numba_compat import NumbaString, NumbaStringArray
from .algorithms.string import _text_cat, _text_cat_chunked, _text_cat_chunked_mixed
from .base import FletcherArray


@pd.api.extensions.register_series_accessor("fr_text")
class TextAccessor:
    """Accessor for pandas exposed as ``.str``."""

    def __init__(self, obj):
        if not isinstance(obj.values, FletcherArray):
            raise AttributeError("only FletcherArray[string] has text accessor")
        self.obj = obj
        self.data = self.obj.values.data

    def cat(self, others: Optional[FletcherArray]) -> pd.Series:
        """
        Concatenate strings in the Series/Index with given separator.

        If `others` is specified, this function concatenates the Series/Index
        and elements of `others` element-wise.
        If `others` is not passed, then all values in the Series/Index are
        concatenated into a single string with a given `sep`.
        """
        if not isinstance(others, pd.Series):
            raise NotImplementedError("other needs to be Series of FletcherArray")
        elif isinstance(others.values, FletcherArray):
            return pd.Series(
                FletcherArray(_text_cat_chunked(self.data, others.values.data))
            )
        elif not isinstance(others.values, FletcherArray):
            raise NotImplementedError("other needs to be FletcherArray")

        if isinstance(self.obj.values, FletcherArray):
            return pd.Series(
                FletcherArray(_text_cat_chunked_mixed(self.data, others.values.data))
            )
        else:  # FletcherArray
            return pd.Series(FletcherArray(_text_cat(self.data, others.values.data)))

    def zfill(self, width: int) -> pd.Series:
        """Pad strings in the Series/Index by prepending '0' characters."""
        # TODO: This will extend all strings to be at least width wide but we need to take uncode into account where the length could be smaller due to multibyte characters
        # This will require a StringBuilder class or a run where we pre-compute the size of the final array
        raise NotImplementedError("zfill")

    def startswith(self, needle, na=None):
        """Check whether a row starts with a certain pattern."""
        return self._call_x_with(_startswith, needle, na)

    def endswith(self, needle, na=None):
        """Check whether a row ends with a certain pattern."""
        return self._call_x_with(_endswith, needle, na)

    def _call_x_with(self, impl, needle, na=None):
        needle = NumbaString.make(needle)

        if isinstance(na, bool):
            result = np.zeros(len(self.data), dtype=np.bool)
            na_arg = np.bool_(na)

        else:
            result = np.zeros(len(self.data), dtype=np.uint8)
            na_arg = 2

        offset = 0
        for chunk in self.data.chunks:
            impl(NumbaStringArray.make(chunk), needle, na_arg, offset, result)
            offset += len(chunk)

        result = pd.Series(result, index=self.obj.index, name=self.obj.name)
        return (
            result if isinstance(na, bool) else result.map({0: False, 1: True, 2: na})
        )
