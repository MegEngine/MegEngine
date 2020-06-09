# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np


def assertTensorClose(
    v0, v1, *, max_err: float = 1e-6, allow_special_values: bool = False, name=None
):
    """
    :param allow_special_values: whether to allow :attr:`v0` and :attr:`v1` to contain inf and nan values.
    :param max_err: relative error
    """
    __tracebackhide__ = True  # pylint: disable=unused-variable

    assert (
        v0.dtype == v1.dtype
    ), "Two Tensor must have same dtype, but the inputs are {} and {}".format(
        v0.dtype, v1.dtype
    )
    v0 = np.ascontiguousarray(v0, dtype=np.float32).copy()
    v1 = np.ascontiguousarray(v1, dtype=np.float32).copy()
    if allow_special_values:
        # check nan and rm it
        v0_nan_mask = np.isnan(v0)
        if np.any(v0_nan_mask):
            assert np.array_equiv(v0_nan_mask, np.isnan(v1)), (v0, v1)
            v0[v0_nan_mask] = 0
            v1[v0_nan_mask] = 0
        # check inf and rm it
        v0_inf_mask = v0 == float("inf")
        if np.any(v0_inf_mask):
            assert np.array_equiv(v0_inf_mask, v1 == float("inf")), (v0, v1)
            v0[v0_inf_mask] = 0
            v1[v0_inf_mask] = 0
        # check -inf and rm it
        v0_inf_mask = v0 == float("-inf")
        if np.any(v0_inf_mask):
            assert np.array_equiv(v0_inf_mask, v1 == float("-inf")), (v0, v1)
            v0[v0_inf_mask] = 0
            v1[v0_inf_mask] = 0
    else:
        assert np.isfinite(v0.sum()) and np.isfinite(v1.sum()), (v0, v1)

    assert v0.shape == v1.shape, "Two tensor must have same shape({} v.s. {})".format(
        v0.shape, v1.shape
    )
    vdiv = np.max([np.abs(v0), np.abs(v1), np.ones_like(v0)], axis=0)
    err = np.abs(v0 - v1) / vdiv
    check = err > max_err
    if check.sum():
        idx = tuple(i[0] for i in np.nonzero(check))
        if name is None:
            name = "tensor"
        else:
            name = "tensor {}".format(name)
        raise AssertionError(
            "{} not equal: "
            "shape={} nonequal_idx={} v0={} v1={} err={}".format(
                name, v0.shape, idx, v0[idx], v1[idx], err[idx]
            )
        )
