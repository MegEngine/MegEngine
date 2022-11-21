import numpy as np

import megengine as mge
import megengine.functional as F
from megengine import Parameter

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
    Refer to `Instance Normalization https://arxiv.org/abs/1607.08022`__
    
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
