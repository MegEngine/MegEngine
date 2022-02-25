import numpy as np

from ..functional import matmul, relu
from ..tensor import Parameter
from . import init
from .module import Module


class BatchMatMulActivation(Module):
    r"""Batched :func:`~.matmul` with activation(only :func:`~.relu` supported), no transpose anywhere."""

    def __init__(
        self,
        batch: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        nonlinear_mode="identity",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch = batch
        self.out_features = out_features
        self.in_features = in_features
        w_shape = (batch, out_features, in_features)
        self.weight = Parameter(np.zeros(w_shape, dtype=np.float32))
        self.bias = None
        if bias:
            b_shape = (out_features,)
            self.bias = Parameter(np.zeros(b_shape, dtype=np.float32))
        self.nonlinear_mode = nonlinear_mode.lower()
        self.reset_parameters()

    def _get_fanin(self):
        return self.in_features

    def reset_parameters(self) -> None:
        fanin = self._get_fanin()
        std = np.sqrt(1 / fanin)
        init.normal_(self.weight, 0.0, std)
        if self.bias is not None:
            init.zeros_(self.bias)

    def _calc_linear(self, x, weight, bias):
        res = matmul(weight, x)
        if self.bias is not None:
            res += bias
        if self.nonlinear_mode == "relu":
            res = relu(res)
        return res

    def forward(self, x):
        return self._calc_linear(x, self.weight, self.bias)

    def _module_info_string(self) -> str:
        return "batch={}, in_features={}, out_features={}, bias={}".format(
            self.batch, self.in_features, self.out_features, self.bias is not None
        )
