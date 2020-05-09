# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import abstractmethod

import numpy as np

from .. import functional as F
from .._internal.dtype import _metadata_dict, get_quantized_dtype
from ..core import Buffer, Function, tensor
from ..module import Module


class Round(Function):
    def forward(self, x):
        return x.round()

    def backward(self, output_grads):
        return output_grads


class Observer(Module):
    r"""
    A base class for Observer Module.

    :param dtype: a string indicating to collect scale and zero_point of which dtype
    """

    def __init__(self, dtype="qint8"):
        super().__init__()
        if dtype not in _metadata_dict.keys():
            raise ValueError(
                "unknown dtype: {}, only support {}".format(
                    dtype, _metadata_dict.keys()
                )
            )
        self.dtype = dtype
        self.qmin = _metadata_dict[dtype].qmin
        self.qmax = _metadata_dict[dtype].qmax
        self.enabled = True

    def get_dtype(self):
        scale, zero_point = self.get_qparams()
        numpy_scale = None if scale is None else scale.numpy()[0]
        numpy_zero_point = None if zero_point is None else zero_point.numpy()[0]
        return get_quantized_dtype(self.dtype, numpy_scale, numpy_zero_point)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode:
            self.enable()
        else:
            self.disable()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_qparams(self, **kwargs):
        pass


class MinMaxObserver(Observer):
    def __init__(self, symmetric=True, eps=0.00001, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetric = symmetric
        if self.symmetric:
            # assert qmin + qmax == -1, 'when reduce_range, qmin + qmax shoule equal -1'
            self.zero_point = tensor((self.qmin + self.qmax + 1) // 2)

        self.min_val = Buffer(0.0, dtype=np.float32)
        self.max_val = Buffer(0.0, dtype=np.float32)
        self.scale_limit = eps
        # flag is used by cond_take, first time will be first flag, and after will be set as not_flag
        self.first_flag = Buffer(np.array([1, 0], dtype=np.int32))
        self.not_flag = Buffer(np.array([0, 1], dtype=np.int32))

    def set_min_max(self, tmp_min, tmp_max):
        # FIXME: cond_take will destory shape, use reshape to reset shape
        tmp_min = tmp_min.reshape(1)
        tmp_max = tmp_max.reshape(1)
        F.add_update(self.min_val, tmp_min, alpha=0.0, beta=1.0, bias=0.0)
        F.add_update(self.max_val, tmp_max, alpha=0.0, beta=1.0, bias=0.0)
        F.add_update(self.first_flag, self.not_flag, alpha=0.0, beta=1.0, bias=0.0)

    def get_qparams(self):
        if self.symmetric:
            symmetric_max_vals = F.maximum(-self.min_val, self.max_val)
            # use maximun to avoid scale too small at the begin
            scale = F.maximum(
                symmetric_max_vals / ((self.qmax - self.qmin) / 2), self.scale_limit
            )
            zero_point = self.zero_point
        else:
            # use maximun to avoid scale too small at the begin
            scale = F.maximum(
                (self.max_val - self.min_val) / (self.qmax - self.qmin),
                self.scale_limit,
            )
            # caculate zero_point
            zero_point = self.qmin - Round()((self.min_val / scale))

        return scale, zero_point

    def forward(self, x_orig):
        if self.enabled:
            # stop gradient
            x = F.zero_grad(x_orig)
            # find max and min
            tmp_min, _ = F.cond_take(
                self.first_flag, F.concat([x.min(), F.minimum(self.min_val, x.min())])
            )
            tmp_max, _ = F.cond_take(
                self.first_flag, F.concat([x.max(), F.maximum(self.max_val, x.max())])
            )
            self.set_min_max(tmp_min, tmp_max)
        return x_orig


class ExponentialMovingAverageObserver(MinMaxObserver):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = Buffer(momentum)

    def set_momentum(self, momentum):
        self.momentum.set_value(momentum)

    def forward(self, x_orig):
        if self.enabled:
            # stop gradient
            x = F.zero_grad(x_orig)
            # Exponential Moving Average
            tmp_min, _ = F.cond_take(
                self.first_flag,
                F.concat(
                    [
                        x.min(),
                        self.momentum * self.min_val + (1 - self.momentum) * x.min(),
                    ]
                ),
            )
            tmp_max, _ = F.cond_take(
                self.first_flag,
                F.concat(
                    [
                        x.max(),
                        self.momentum * self.max_val + (1 - self.momentum) * x.max(),
                    ]
                ),
            )
            self.set_min_max(tmp_min, tmp_max)
        return x_orig
