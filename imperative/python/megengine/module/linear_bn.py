import numpy as np

from ..functional import relu
from ..tensor import Parameter
from .batchnorm import BatchNorm1d
from .linear import Linear
from .module import Module


class _LinearBnActivation1d(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        compute_mode: str = "default",
        eps=1e-5,
        momentum=0.9,
        affine=True,
        track_running_stats=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.out_features = out_features
        self.in_features = in_features
        self.bias = None
        if bias:
            b_shape = (out_features,)
            self.bias = Parameter(np.zeros(b_shape, dtype=np.float32))
        self.linear = Linear(in_features, out_features, bias, compute_mode, **kwargs,)
        self.bn = BatchNorm1d(out_features, eps, momentum, affine, track_running_stats)


class LinearBn1d(_LinearBnActivation1d):
    r"""A fused :class:`~.Module` including :class:`~.module.Linear` and :class:`~.module.BatchNorm1d`.
    Could be replaced with :class:`~.QATModule` version :class:`~.qat.LinearBn1d` using
    :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return self.bn(self.linear(inp))


class LinearBnRelu1d(_LinearBnActivation1d):
    r"""A fused :class:`~.Module` including :class:`~.module.Linear`, :class:`~.module.BatchNorm1d` and :func:`~.relu`.
    Could be replaced with :class:`~.QATModule` version :class:`~.qat.LinearBnRelu1d` using :func:`~.quantize.quantize_qat`.
    """

    def forward(self, inp):
        return relu(self.bn(self.linear(inp)))
