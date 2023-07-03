import typing as T

import numpy as np

import megengine as mge
import megengine.functional as F
from megengine import Parameter

from ..logger import get_logger
from .init import ones_, zeros_
from .module import Module


class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs
    Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`__
    
    .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
            
    The mean and standard-deviation are calculated separately over the each group.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    attr:`num_channels` if :attr:`affine` is ``True``.
    
    Args:
        num_groups (int): number of groups that divided from channels.
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: this module has learnable affine parameters (weight, bias) when affine is set to be True.

    Shape:
        - Input: :math:`(N, C, H, W)` (now only support NCHW format tensor)
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples: 
        >>> import numpy as np
        >>> inp = Tensor(np.arange(2 * 3 * 4 * 4).astype(np.float32).reshape(2, 3, 4, 4))
        >>> m = M.GroupNorm(3, 3)
        >>> out = m(inp)
        >>> out.numpy().shape
        (2, 3, 4, 4)
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
    r"""Applies Instance Normalization over a mini-batch of inputs
    Refer to `Instance Normalization <https://arxiv.org/abs/1607.08022>`__
    
    .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for each object in a mini-batch.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    attr:`num_channels` if :attr:`affine` is ``True``.
    Note that InstanceNorm equals using GroupNorm with num_groups = num_channels.
    
    Args:
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: this module has learnable affine parameters (weight, bias) when affine is set to be True.
        
    Shape:
        - Input: :math:`(N, C, H, W)` (now only support NCHW format tensor)
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples:
        >>> import numpy as np
        >>> inp = Tensor(np.arange(2 * 3 * 4 * 4).astype(np.float32).reshape(2, 3, 4, 4))
        >>> m = M.InstanceNorm(3)
        >>> out = m(inp)
        >>> out.numpy().shape
        (2, 3, 4, 4)
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
        x = F.nn.instance_norm(x, self.affine, self.weight, self.bias, self.eps)
        return x

    def _module_info_string(self) -> str:
        s = "channels={num_channels}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)


class LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs
    Refer to `Layer Normalization <https://arxiv.org/pdf/1607.06450v1.pdf>`_
    
    .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
            
    The mean and standard-deviation are calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.
    
    Args:
        normalized_shape(int or tuple): input shape from an expected input of size
            size :math:`[*, normalized\_shape[0], normalized\_shape[1], ..., normalized\_shape[-1]]`.
            If it is a single integer, this module will normalize over the last dimension
            which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: this module has learnable affine parameters (weight, bias) when affine is set to be True.

    Shape:
        - Input: :math:`(N, *)` (2-D, 3-D, 4-D or 5-D tensor)
        - Output: :math:`(N, *)` (same shape as input)

    Examples:
        >>> import numpy as np
        >>> inp = Tensor(np.arange(2 * 3 * 4 * 4).astype(np.float32).reshape(2, 3, 4, 4))
        >>> m = M.LayerNorm((4, 4))
        >>> out = m(inp)
        >>> out.numpy().shape
        (2, 3, 4, 4)
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


class GeneralNorm(Module):
    r"""Applies General Normalization over a mini-batch of inputs

    .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated separately according to the axis 
    given by :attr:`normalized_axis`.
    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters
    if :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator.
    
    Args:
        normalized_shape(int, list or tuple): the shape of input needs to be normalized, normalized_shape must be specified when affine is true. When affine=true, we will directly use this shape to initialize weight/bias. Please ensure that the order is correct. Default: None
        normalized_axis(int, list or tuple): the axis of input needs to be normalized, one-to-one correspondence between normalized_axis and normalized_shape. Default: -1
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: this module has learnable affine parameters (weight, bias) when affine is set to be True.

    Shape:
        - Input: :math:`(N, *)` (2-D, 3-D, 4-D or 5-D tensor)
        - Output: :math:`(N, *)` (same shape as input)

    Examples:
        >>> import numpy as np
        >>> inp = Tensor(np.arange(2 * 3 * 4 * 4).astype(np.float32).reshape(2, 3, 4, 4))
        >>> m = M.GeneralNorm((2, 3), (0, 1))
        >>> out = m(inp)
        >>> out.numpy().shape
        (2, 3, 4, 4)
        >>> m = M.GeneralNorm((3, 4), (1, -2)) # Please be careful.
        >>> out = m(inp)
        >>> out.numpy().shape
        (2, 3, 4, 4)
        >>> m = M.GeneralNorm((2, 4, 3), (0, 2, 1)) # Incorrect initialization, the order of normalized_axis is incorrect, should be adjusted to m = M.GeneralNorm((2, 3, 4), (0, 1, 2)).
        >>> m = M.GeneralNorm((2, 4, 3), (0, -2, 1)) # Incorrect initialization, the order of normalized_axis is incorrect, should be adjusted to m = M.GeneralNorm((2, 3, 4), (0, 1, -2)).
        >>> m = M.GeneralNorm((3, 4), (3, -1)) # Incorrect initialization, because axis=-1 and axis=3 are the same axis, namely axis=3.
    """

    def __init__(
        self, normalized_shape=None, normalized_axis=0, eps=1e-05, affine=True, **kwargs
    ):
        super().__init__(**kwargs)

        self.eps = eps
        self.affine = affine

        if self.affine:
            assert (
                normalized_shape is not None
            ), "normalized_shape must be specified when affine is true"
            assert (
                normalized_axis is not None
            ), "normalized_axis must be specified when affine is true"
            if not isinstance(normalized_axis, T.Sequence):
                normalized_axis = [normalized_axis]
            if not isinstance(normalized_shape, T.Sequence):
                normalized_axis = [normalized_shape]
            assert isinstance(normalized_axis, (list, tuple))
            assert isinstance(normalized_shape, (list, tuple))

            assert len(normalized_axis) == len(
                normalized_shape
            ), "The size of normalized_axis and normalized_shape are different"
            assert len(set(normalized_axis)) == len(
                normalized_axis
            ), "there are duplicate axis in list normalized_axis"

            self.weight = Parameter(np.ones(normalized_shape, dtype="float32"))
            self.bias = Parameter(np.zeros(normalized_shape, dtype="float32"))
        else:
            self.weight = None
            self.bias = None

        self.normalized_shape = normalized_shape
        self.normalized_axis = normalized_axis
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            ones_(self.weight)
            zeros_(self.bias)

    def forward(self, x):
        self.normalized_axis = [
            num + x.ndim if num < 0 else num for num in self.normalized_axis
        ]
        assert self.normalized_axis == sorted(
            self.normalized_axis
        ), "The order of normalized_axis is incorrect, should be {}, but got {}. Please specify the values of axis in the correct order in normalized_axis".format(
            sorted(self.normalized_axis), self.normalized_axis
        )
        inp_shape = x.shape
        for i in range(len(self.normalized_axis)):
            assert (
                inp_shape[self.normalized_axis[i]] == self.normalized_shape[i]
            ), "inp.shape={}, normalized_axis={}, normalized_shape={}, inp.shape[normalized_axis[{}]]({}) != normalized_shape[{}]({})".format(
                x.shape,
                self.normalized_axis,
                self.normalized_shape,
                i,
                inp_shape[self.normalized_axis[i]],
                i,
                self.normalized_shape[i],
            )

        x = F.nn.general_norm(
            x, self.normalized_axis, self.affine, self.weight, self.bias, self.eps
        )
        return x

    def _module_info_string(self) -> str:
        s = "normalized_shape={normalized_shape}, normalized_axis={normalized_axis}, eps={eps}, affine={affine}"
        return s.format(**self.__dict__)
