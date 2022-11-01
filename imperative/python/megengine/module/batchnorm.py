# -*- coding: utf-8 -*-
from typing import Optional

import numpy as np

from ..distributed.group import WORLD, Group
from ..functional.nn import batch_norm, sync_batch_norm
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
        **kwargs
    ):
        super(_BatchNorm, self).__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self._track_running_stats_saved = track_running_stats
        self.freeze = freeze
        if self.freeze:
            assert (
                self._track_running_stats_saved
            ), "track_running_stats must be initilized to True if freeze is True"
        tshape = (1, self.num_features, 1, 1)
        if self.affine:
            self.weight = Parameter(np.ones(tshape, dtype=np.float32))
            self.bias = Parameter(np.zeros(tshape, dtype=np.float32))
        else:
            self.weight = None
            self.bias = None

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

        _weight = self.weight
        _bias = self.bias

        if self.freeze:
            if _weight is not None:
                _weight = _weight.detach()
            if _bias is not None:
                _bias = _bias.detach()

            # fastpath excution for freeze
            scale = (self.running_var + self.eps) ** (-0.5)
            if _weight is not None:
                scale *= _weight
            bias = -self.running_mean * scale
            if _bias is not None:
                bias += _bias
            return inp * scale + bias

        if self.training and self.track_running_stats:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0  # useless

        output = batch_norm(
            inp,
            self.running_mean if self.track_running_stats else None,
            self.running_var if self.track_running_stats else None,
            _weight,
            _bias,
            training=self.training
            or ((self.running_mean is None) and (self.running_var is None)),
            momentum=exponential_average_factor,
            eps=self.eps,
        )

        return output

    def _module_info_string(self) -> str:
        s = (
            "{num_features}, eps={eps}, momentum={momentum}, affine={affine}, "
            "track_running_stats={track_running_stats}"
        )
        return s.format(**self.__dict__)


class SyncBatchNorm(_BatchNorm):
    r"""Applies Synchronized Batch Normalization for distributed training.

    Args:
        num_features: usually :math:`C` from an input of shape
            :math:`(N, C, H, W)` or the highest ranked dimension of an input
            less than 4D.
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the ``running_mean`` and ``running_var`` computation.
            Default: 0.9
        affine: a boolean value that when set to True, this module has
            learnable affine parameters. Default: True
        track_running_stats: when set to True, this module tracks the
            running mean and variance. When set to False, this module does not
            track such statistics and always uses batch statistics in both training
            and eval modes. Default: True
        freeze: when set to True, this module does not update the
            running mean and variance, and uses the running mean and variance instead of
            the batch mean and batch variance to normalize the input. The parameter takes effect
            only when the module is initilized with track_running_stats as True.
            Default: False
        group: communication group, caculate mean and variance between this group.
            Default: :obj:`~.distributed.WORLD`
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
        **kwargs
    ) -> None:
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, freeze, **kwargs
        )
        self.group = group

    def _check_input_ndim(self, inp):
        if len(inp.shape) not in {2, 3, 4}:
            raise ValueError(
                "expected 2D, 3D or 4D input (got {}D input)".format(len(inp.shape))
            )

    def forward(self, inp):
        self._check_input_ndim(inp)

        inp_shape = inp.shape
        _ndims = len(inp_shape)
        if _ndims != 4:
            new_shape = Tensor([1, 1, 1, 1], device=inp.device)
            origin_shape = inp_shape
            if _ndims == 2:
                new_shape[:2] = origin_shape[:2]
            elif _ndims == 3:
                new_shape[:3] = origin_shape[:3]
            else:
                raise ValueError(
                    "expected 2D, 3D or 4D input (got {}D input)".format(len(inp_shape))
                )

            inp = inp.reshape(new_shape)

        if self.training and self.track_running_stats:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0  # useless

        _weight = self.weight
        _bias = self.bias

        if self.freeze:
            if _weight is not None:
                _weight = _weight.detach()
            if _bias is not None:
                _bias = _bias.detach()

        output = sync_batch_norm(
            inp,
            self.running_mean,
            self.running_var,
            _weight,
            _bias,
            training=(self.training and not self.freeze)
            or ((self.running_mean is None) and (self.running_var is None)),
            momentum=exponential_average_factor,
            eps=self.eps,
            group=self.group,
        )

        if _ndims != 4:
            output = output.reshape(origin_shape)

        return output


class BatchNorm1d(_BatchNorm):
    r"""Applies Batch Normalization over a 2D/3D tensor.

    Refer to :class:`~.BatchNorm2d` for more information.
    """

    def _check_input_ndim(self, inp):
        if len(inp.shape) not in {2, 3}:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(len(inp.shape))
            )


class BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D tensor.

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

    Because the Batch Normalization is done over the `C` dimension, computing
    statistics on `(N, H, W)` slices, it's common terminology to call this
    Spatial Batch Normalization.

    .. note::

        The update formula for ``running_mean`` and ``running_var`` (taking ``running_mean`` as an example) is

        .. math::

            \textrm{running_mean} = \textrm{momentum} \times \textrm{running_mean} + (1 - \textrm{momentum}) \times \textrm{batch_mean}

        which could be defined differently in other frameworks. Most notably, ``momentum`` of 0.1 in PyTorch
        is equivalent to ``mementum`` of 0.9 here.

    Args:
        num_features: usually :math:`C` from an input of shape
            :math:`(N, C, H, W)` or the highest ranked dimension of an input
            less than 4D.
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the ``running_mean`` and ``running_var`` computation.
            Default: 0.9
        affine: a boolean value that when set to True, this module has
            learnable affine parameters. Default: True
        track_running_stats: when set to True, this module tracks the
            running mean and variance. When set to False, this module does not
            track such statistics and always uses batch statistics in both training
            and eval modes. Default: True
        freeze: when set to True, this module does not update the
            running mean and variance, and uses the running mean and variance instead of
            the batch mean and batch variance to normalize the input. The parameter takes effect
            only when the module is initilized with track_running_stats as True.
            Default: False

    Examples:
        >>> import numpy as np
        >>> # With Learnable Parameters
        >>> m = M.BatchNorm2d(4)
        >>> inp = mge.tensor(np.random.rand(1, 4, 3, 3).astype("float32"))
        >>> oup = m(inp)
        >>> print(m.weight.numpy().flatten(), m.bias.numpy().flatten())
        [1. 1. 1. 1.] [0. 0. 0. 0.]
        >>> # Without Learnable Parameters
        >>> m = M.BatchNorm2d(4, affine=False)
        >>> oup = m(inp)
        >>> print(m.weight, m.bias)
        None None
    """

    def _check_input_ndim(self, inp):
        if len(inp.shape) != 4:
            raise ValueError("expected 4D input (got {}D input)".format(len(inp.shape)))
