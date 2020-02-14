# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np


def assertTensorClose(v0, v1, *, max_err=1e-6, name=None):
    """
    max_err: relative error
    """
    __tracebackhide__ = True  # pylint: disable=unused-variable

    assert (
        v0.dtype == v1.dtype
    ), "Two Tensor must have same dtype, but the inputs are {} and {}".format(
        v0.dtype, v1.dtype
    )
    v0 = np.ascontiguousarray(v0, dtype=np.float32)
    v1 = np.ascontiguousarray(v1, dtype=np.float32)
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
