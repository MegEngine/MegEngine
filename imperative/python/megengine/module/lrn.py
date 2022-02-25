# -*- coding: utf-8 -*-
from typing import Tuple, Union

from ..functional import local_response_norm
from .module import Module


class LocalResponseNorm(Module):
    r"""
    Apply local response normalization to the input tensor.

    Args:
        kernel_size: the size of the kernel to apply LRN on.
        k: hyperparameter k. The default vaule is 2.0.
        alpha: hyperparameter alpha. The default value is 1e-4.
        beta: hyperparameter beta. The default value is 0.75.

    Example:
        >>> import numpy as np
        >>> inp = Tensor(np.arange(25, dtype=np.float32).reshape(1,1,5,5))
        >>> GT = np.array([[[[ 0.,         0.999925,   1.9994003,  2.9979765,  3.9952066],
        ...                  [ 4.9906454,  5.983851,   6.974385,   7.961814,   8.945709 ],
        ...                  [ 9.925651,  10.90122,   11.872011,  12.837625,  13.7976675],
        ...                  [14.751757,  15.699524,  16.640602,  17.574642,  18.501305 ],
        ...                  [19.420258,  20.331186,  21.233786,  22.127764,  23.012836 ]]]])
        >>> op = M.LocalResponseNorm(kernel_size=3, k=1.0, alpha=1e-4, beta=0.75)
        >>> out = op(inp)
        >>> np.testing.assert_allclose(GT, out.numpy(), rtol=1e-6, atol=1e-6)
    """

    def __init__(
        self,
        kernel_size: int = 5,
        k: float = 2.0,
        alpha: float = 1e-4,
        beta: float = 0.75,
        **kwargs
    ):
        super(LocalResponseNorm, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.k = k
        self.alpha = alpha
        self.beta = beta

    def forward(self, inp):
        return local_response_norm(inp, self.kernel_size, self.k, self.alpha, self.beta)
