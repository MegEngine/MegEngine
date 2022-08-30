import numpy as np

import megengine as mge
import megengine.functional as F
from megengine import Parameter

from .init import ones_, zeros_
from .module import Module


class GroupNorm(Module):
    """Simple implementation of GroupNorm. Only support 4d tensor now.
    Reference: https://arxiv.org/pdf/1803.08494.pdf.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, **kwargs):
        super().__init__(**kwargs)
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
        x = F.nn.group_norm(
            x, self.num_groups, self.affine, self.weight, self.bias, self.eps
        )
        return x

    def _module_info_string(self) -> str:
        s = (
            "groups={num_groups}, channels={num_channels}, "
            "eps={eps}, affine={affine}"
        )
        return s.format(**self.__dict__)


class InstanceNorm(Module):
    """Simple implementation of InstanceNorm. Only support 4d tensor now.
    Reference: https://arxiv.org/abs/1607.08022.
    Note that InstanceNorm equals using GroupNome with num_groups=num_channels.
    """

    def __init__(self, num_channels, eps=1e-05, affine=True, **kwargs):
        super().__init__(**kwargs)
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
        format = x.format
        assert C == self.num_channels
        x = x.reshape(N, C, -1)
        mean = x.mean(axis=2, keepdims=True)
        var = (x ** 2).mean(axis=2, keepdims=True) - mean * mean

        x = (x - mean) / F.sqrt(var + self.eps)
        x = x.reshape(N, C, H, W)
        if self.affine:
            x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(1, -1, 1, 1)
        # FIXME(czh): remove this after making it a builtin op.
        if format == "nhwc":
            x = mge.amp.convert_tensor_format(x, inplace=False)
        return x

    def _module_info_string(self) -> str:
        s = "channels={num_channels}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)


class LayerNorm(Module):
    """Simple implementation of LayerNorm. Support tensor of any shape as input.
    Reference: https://arxiv.org/pdf/1803.08494.pdf.
    """

    def __init__(self, normalized_shape, eps=1e-05, affine=True, **kwargs):
        super().__init__(**kwargs)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(np.ones(self.normalized_shape, dtype="float32"))
            self.bias = Parameter(np.zeros(self.normalized_shape, dtype="float32"))
        else:
            self.weight = None
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        x = F.nn.layer_norm(
            x, self.normalized_shape, self.affine, self.weight, self.bias, self.eps
        )
        return x

    def _module_info_string(self) -> str:
        s = "normalized_shape={normalized_shape}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)
