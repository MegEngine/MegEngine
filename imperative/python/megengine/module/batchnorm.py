# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import Optional

import numpy as np

from ..distributed.group import WORLD, Group
from ..functional.nn import batch_norm2d, sync_batch_norm
from ..tensor import Parameter, Tensor
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
        freeze=False,
    ):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._track_running_stats_saved = track_running_stats
        self.freeze = freeze
        if self.affine:
            self.weight = Parameter(np.ones(num_features, dtype=np.float32))
            self.bias = Parameter(np.zeros(num_features, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

        tshape = (1, self.num_features, 1, 1)

        if self.track_running_stats:
            self.running_mean = Tensor(np.zeros(tshape, dtype=np.float32))
            self.running_var = Tensor(np.ones(tshape, dtype=np.float32))
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
        if self._track_running_stats_saved == False:
            assert (
                self.track_running_stats == False
            ), "track_running_stats can not be initilized to False and changed to True later"

        _ndims = len(inp.shape)
        if _ndims != 4:
            origin_shape = inp.shape
            if _ndims == 2:
                n, c = inp.shape[0], inp.shape[1]
                new_shape = (n, c, 1, 1)
            elif _ndims == 3:
                n, c, h = inp.shape[0], inp.shape[1], inp.shape[2]
                new_shape = (n, c, h, 1)

            inp = inp.reshape(new_shape)

        if self.freeze and self.training and self._track_running_stats_saved:
            scale = self.weight.reshape(1, -1, 1, 1) * (
                self.running_var + self.eps
            ) ** (-0.5)
            bias = self.bias.reshape(1, -1, 1, 1) - self.running_mean * scale
            return inp * scale.detach() + bias.detach()

        if self.training and self.track_running_stats:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0  # useless

        output = batch_norm2d(
            inp,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            self.weight,
            self.bias,
            training=self.training
            or ((self.running_mean is None) and (self.running_var is None)),
            momentum=exponential_average_factor,
            eps=self.eps,
        )

        if _ndims != 4:
            output = output.reshape(origin_shape)

        return output

    def _module_info_string(self) -> str:
        s = (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}"
        )
        return s.format(**self.__dict__)


class SyncBatchNorm(_BatchNorm):
    r"""
    Applies Synchronization Batch Normalization.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
        freeze=False,
        group: Optional[Group] = WORLD,
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, freeze
        )
        self.group = group

    def _check_input_ndim(self, inp):
        if len(inp.shape) not in {2, 3, 4}:
            raise ValueError(
                "expected 2D, 3D or 4D input (got {}D input)".format(len(inp.shape))
            )

    def forward(self, inp):
        self._check_input_ndim(inp)

        _ndims = len(inp.shape)
        if _ndims != 4:
            new_shape = Tensor([1, 1, 1, 1], device=inp.device)
            origin_shape = inp.shape
            if _ndims == 2:
                new_shape[:2] = origin_shape[:2]
            elif _ndims == 3:
                new_shape[:3] = origin_shape[:3]
            else:
                raise ValueError(
                    "expected 2D, 3D or 4D input (got {}D input)".format(len(inp.shape))
                )

            inp = inp.reshape(new_shape)

        if self.training and self.track_running_stats:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0  # useless

        output = sync_batch_norm(
            inp,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
            group=self.group,
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
    of 0.9.

    If :attr:`track_running_stats` is set to ``False``, this layer will not
    keep running estimates, batch statistics is used during
    evaluation time instead.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = \text{momentum} \times \hat{x} + (1 - \text{momentum}) \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing
    statistics on `(N, H, W)` slices, it's common terminology to call this
    Spatial Batch Normalization.

    :type num_features: int
    :param num_features: usually :math:`C` from an input of shape
        :math:`(N, C, H, W)` or the highest ranked dimension of an input
        less than 4D.
    :type eps: float
    :param eps: a value added to the denominator for numerical stability.
        Default: 1e-5
    :type momentum: float
    :param momentum: the value used for the ``running_mean`` and ``running_var`` computation.
        Default: 0.9
    :type affine: bool
    :param affine: a boolean value that when set to True, this module has
        learnable affine parameters. Default: True
    :type track_running_stats: bool
    :param track_running_stats: when set to True, this module tracks the
        running mean and variance. When set to False, this module does not
        track such statistics and always uses batch statistics in both training
        and eval modes. Default: True

    :type freeze: bool
    :param freeze: when set to True, this module does not update the
        running mean and variance, and uses the running mean and variance instead of
        the batch mean and batch variance to normalize the input. The parameter takes effect
        only when the module is initilized with track_running_stats as True and
        the module is in training mode.
        Default: False

    Examples:

    .. testcode::

        import numpy as np
        import megengine as mge
        import megengine.module as M

        # With Learnable Parameters
        m = M.BatchNorm2d(4)
        inp = mge.tensor(np.random.rand(1, 4, 3, 3).astype("float32"))
        oup = m(inp)
        print(m.weight.numpy(), m.bias.numpy())
        # Without L`e`arnable Parameters
        m = M.BatchNorm2d(4, affine=False)
        oup = m(inp)
        print(m.weight, m.bias)

    Outputs:

    .. testoutput::

        [1. 1. 1. 1.] [0. 0. 0. 0.]
        None None
    """

    def _check_input_ndim(self, inp):
        if len(inp.shape) != 4:
            raise ValueError("expected 4D input (got {}D input)".format(len(inp.shape)))
