# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import platform
import weakref

import numpy as np
import pytest
import torch

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim


def make_fake_params():
    shapes = [(1,), (3, 3), (5, 5, 5), (6, 7, 8, 9)]
    params = [np.random.randn(*shape).astype("float32") for shape in shapes]
    params_mge = []
    params_torch = []
    for param in params:
        t = torch.ones(param.shape)
        t.grad = torch.Tensor(param.copy())
        params_torch.append(t)

        t = mge.functional.ones(param.shape)
        t.grad = mge.tensor(param.copy())
        params_mge.append(t)
    return params_mge, params_torch


def test_clip_grad_norm_torch():
    max_norm = 1.0
    params_mge, params_torch = make_fake_params()
    norm_torch = torch.nn.utils.clip_grad_norm_(params_torch, max_norm, norm_type=2.0)
    norm_mge = optim.clip_grad_norm(params_mge, max_norm=max_norm, ord=2.0)
    np.testing.assert_allclose(norm_mge.numpy(), norm_torch.numpy(), atol=1e-4)
    for i in range(len(params_mge)):
        np.testing.assert_allclose(
            params_mge[i].grad.numpy(), params_torch[i].grad.numpy(), atol=1e-7
        )


def test_clip_grad_value_torch():
    max_val = 0.5
    min_val = -0.5
    params_mge, params_torch = make_fake_params()
    torch.nn.utils.clip_grad_value_(params_torch, clip_value=max_val)
    optim.clip_grad_value(params_mge, lower=min_val, upper=max_val)
    for i in range(len(params_mge)):
        np.testing.assert_allclose(
            params_mge[i].grad.numpy(), params_torch[i].grad.numpy(), atol=1e-7
        )
