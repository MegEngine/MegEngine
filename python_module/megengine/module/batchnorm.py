# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

from ..core import Buffer, Parameter
from ..functional import batch_norm2d
from . import init
from .module import Module


class _BatchNorm(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
    ):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

        tshape = (1, self.num_features, 1, 1)

        if self.track_running_stats:
            self.running_mean = Buffer(np.zeros(tshape, dtype=np.float32))
            self.running_var = Buffer(np.ones(tshape, dtype=np.float32))
        else:
            self.running_mean = None
            self.running_var = None

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            init.zeros_(self.running_mean)
            init.ones_(self.running_var)

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def _check_input_ndim(self, inp):
        raise NotImplementedError

    def forward(self, inp):
        self._check_input_ndim(inp)

        _ndims = len(inp.shape)
        if _ndims != 4:
            origin_shape = inp.shapeof()
            if _ndims == 2:
                n, c = inp.shapeof(0), inp.shapeof(1)
                new_shape = (n, c, 1, 1)
            elif _ndims == 3:
                n, c, h = inp.shapeof(0), inp.shapeof(1), inp.shapeof(2)
                new_shape = (n, c, h, 1)

            inp = inp.reshape(new_shape)

        _iter_update = None
        if self.training and self.track_running_stats:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0  # useless

        output = batch_norm2d(
            inp,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )

        if _ndims != 4:
            output = output.reshape(origin_shape)

        return output


class BatchNorm1d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 2D/3D tensor.

    Refer to :class:`~.BatchNorm2d` for more information.
    """

    def _check_input_ndim(self, inp):
        if len(inp.shape) not in {2, 3}:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(len(inp.shape))
            )


class BatchNorm2d(_BatchNorm):
    r"""
    Applies Batch Normalization over a 4D tensor.

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable
    parameter vectors.

    By default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer will not
    keep running estimates, and batch statistics are instead used during
    evaluation time.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing
    statistics on `(N, H, W)` slices, it's common terminology to call this
    Spatial Batch Normalization.

    :type num_features: int
    :param num_features: usually the :math:`C` from an input of size
        :math:`(N, C, H, W)` or the highest ranked dimension of an input with
        less than 4D.
    :type eps: float
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5.
    :type momentum: float
    :param momentum: the value used for the `running_mean` and `running_var`
        computation.
        Default: 0.1
    :type affine: bool
    :param affine: a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``
    :type track_running_stats: bool
    :param track_running_stats: when set to ``True``, this module tracks the
        running mean and variance. When set to ``False``, this module does not
        track such statistics and always uses batch statistics in both training
        and eval modes. Default: ``True``.


    Examples:

    .. testcode::

        import megengine as mge
        import megengine.module as M

        # With Learnable Parameters
        m = M.BatchNorm2d(4)
        inp = mge.tensor(np.random.rand(64, 4, 32, 32))
        oup = m(inp)
        # Without Learnable Parameters
        m = M.BatchNorm2d(4, affine=False)
        oup = m(inp)

    """

    def _check_input_ndim(self, inp):
        if len(inp.shape) != 4:
            raise ValueError("expected 4D input (got {}D input)".format(len(inp.shape)))
