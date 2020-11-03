# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import functools


def get_ndtuple(value, *, n, allow_zero=True):
    r"""
    Converts possibly 1D tuple to nd tuple.

    :type allow_zero: bool
    :param allow_zero: whether to allow zero tuple value."""
    if not isinstance(value, collections.abc.Iterable):
        value = int(value)
        value = tuple([value for i in range(n)])
    else:
        assert len(value) == n, "tuple len is not equal to n: {}".format(value)
        spatial_axis = map(int, value)
        value = tuple(spatial_axis)
    if allow_zero:
        minv = 0
    else:
        minv = 1
    assert min(value) >= minv, "invalid value: {}".format(value)
    return value


_single = functools.partial(get_ndtuple, n=1, allow_zero=True)
_pair = functools.partial(get_ndtuple, n=2, allow_zero=True)
_pair_nonzero = functools.partial(get_ndtuple, n=2, allow_zero=False)
_triple = functools.partial(get_ndtuple, n=3, allow_zero=True)
_quadruple = functools.partial(get_ndtuple, n=4, allow_zero=True)
