# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine.functional as F
from megengine import Parameter

from .init import ones_, zeros_
from .module import Module


class GroupNorm(Module):
    """
    Simple implementation of GroupNorm. Only support 4d tensor now.
    Reference: https://arxiv.org/pdf/1803.08494.pdf.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(num_channels, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_channels, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == self.num_channels

        x = x.reshape(N, self.num_groups, -1)
        mean = x.mean(axis=2, keepdims=True)
        var = (x * x).mean(axis=2, keepdims=True) - mean * mean

        x = (x - mean) / F.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        if self.affine:
            x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(1, -1, 1, 1)

        return x

    def _module_info_string(self) -> str:
        s = (
            "groups={num_groups}, channels={num_channels}, "
            "eps={eps}, affine={affine}"
        )
        return s.format(**self.__dict__)


class InstanceNorm(Module):
    """
    Simple implementation of InstanceNorm. Only support 4d tensor now.
    Reference: https://arxiv.org/abs/1607.08022.
    Note that InstanceNorm equals using GroupNome with num_groups=num_channels.
    """

    def __init__(self, num_channels, eps=1e-05, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(num_channels, dtype="float32"))
            self.bias = Parameter(np.zeros(num_channels, dtype="float32"))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == self.num_channels
        x = x.reshape(N, C, -1)
        mean = x.mean(axis=2, keepdims=True)
        var = (x ** 2).mean(axis=2, keepdims=True) - mean * mean

        x = (x - mean) / F.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        if self.affine:
            x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(1, -1, 1, 1)

        return x

    def _module_info_string(self) -> str:
        s = "channels={num_channels}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)


class LayerNorm(Module):
    """
    Simple implementation of LayerNorm. Only support 4d tensor now.
    Reference: https://arxiv.org/pdf/1803.08494.pdf.
    Note that LayerNorm equals using GroupNorm with num_groups=1.
    """

    def __init__(self, num_channels, eps=1e-05, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(num_channels, dtype="float32"))
            self.bias = Parameter(np.zeros(num_channels, dtype="float32"))
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.shape
        assert C == self.num_channels
        x = x.reshape(x.shape[0], -1)
        # NOTE mean will keepdims in next two lines.
        mean = x.mean(axis=1, keepdims=1)
        var = (x ** 2).mean(axis=1, keepdims=1) - mean * mean

        x = (x - mean) / F.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        if self.affine:
            x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(1, -1, 1, 1)

        return x

    def _module_info_string(self) -> str:
        s = "channels={num_channels}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)
