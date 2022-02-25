# -*- coding: utf-8 -*-
import collections
import functools


def get_ndtuple(value, *, n, allow_zero=True):
    r"""Converts possibly 1D tuple to nd tuple.

    Args:
        allow_zero: whether to allow zero tuple value
    """
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
_triple_nonzero = functools.partial(get_ndtuple, n=3, allow_zero=False)
_quadruple = functools.partial(get_ndtuple, n=4, allow_zero=True)
