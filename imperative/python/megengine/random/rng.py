# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
import time
from typing import Iterable, Optional, Union

from numpy.random import MT19937

from .. import Tensor
from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.ops import delete_rng_handle as _delete_rng_handle
from ..core._imperative_rt.ops import get_global_rng_seed as _get_global_rng_seed
from ..core._imperative_rt.ops import (
    get_rng_handle_compnode as _get_rng_handle_compnode,
)
from ..core._imperative_rt.ops import new_rng_handle as _new_rng_handle
from ..core._imperative_rt.ops import set_global_rng_seed as _set_global_rng_seed
from ..core.ops.builtin import (
    BetaRNG,
    GammaRNG,
    GaussianRNG,
    PermutationRNG,
    PoissonRNG,
    ShuffleRNG,
    UniformRNG,
)
from ..core.tensor import utils
from ..device import get_default_device

__all__ = [
    "seed",
    "RNG",
    "uniform",
    "normal",
    "gamma",
    "beta",
    "poisson",
    "permutation",
    "shuffle",
]

_rng = None


def _infer_broadcasted_shape(inps: Iterable[Tensor]) -> tuple:
    broadcasted_ndim = inps[0].ndim
    broadcasted_shape = list(inps[0]._tuple_shape)
    for i in range(1, len(inps)):
        cur_ndim = inps[i].ndim
        cur_shape = list(inps[i]._tuple_shape)
        n_dim = max(cur_ndim, broadcasted_ndim)
        for j in range(n_dim - 1, -1, -1):
            cur_dim = cur_ndim + j - n_dim
            broad_dim = broadcasted_ndim + j - n_dim
            cur_size = cur_shape[cur_dim] if cur_dim >= 0 else 1
            broad_size = broadcasted_shape[broad_dim] if broad_dim >= 0 else 1
            assert cur_size == broad_size or cur_size == 1 or broad_size == 1, (
                "The size of inps[{}] ({}) must match the size ({}) at "
                "dim {}".format(i, cur_size, broad_size, j)
            )
            broad_size = max(cur_size, broad_size)
            if broad_dim < 0:
                broadcasted_shape = [broad_size] + broadcasted_shape
                broadcasted_ndim += 1
            else:
                broadcasted_shape[broad_dim] = broad_size
    return tuple(broadcasted_shape)


def _broadcast_tensors_with_size(
    inps: Iterable[Tensor], size: Iterable[int]
) -> Iterable[Tensor]:
    assert inps, "The inps cloud not be empty"
    target_shape = _infer_broadcasted_shape(inps)
    if isinstance(size, collections.abc.Iterable):
        target_shape = tuple(size) + target_shape
    target_ndim = len(target_shape)
    for i in range(len(inps)):
        if inps[i]._tuple_shape != target_shape:
            inps[i] = (
                inps[i]
                .reshape((1,) * (target_ndim - inps[i].ndim) + inps[i]._tuple_shape)
                ._broadcast(target_shape)
            )
    return inps


def _uniform(
    low: float,
    high: float,
    size: Optional[Iterable[int]],
    seed: int,
    device: str,
    handle: int,
) -> Tensor:
    assert low < high, "Uniform is not defined when low >= high"
    if size is None:
        size = (1,)
    op = UniformRNG(seed=seed, handle=handle, dtype="float32")
    _ref = Tensor([], dtype="int32", device=device)
    shape = utils.astensor1d(size, _ref, dtype="int32", device=device)
    (output,) = apply(op, shape)
    return low + (high - low) * output


def _normal(
    mean: float,
    std: float,
    size: Optional[Iterable[int]],
    seed: int,
    device: str,
    handle: int,
) -> Tensor:
    if size is None:
        size = (1,)
    op = GaussianRNG(seed=seed, mean=mean, std=std, handle=handle, dtype="float32")
    _ref = Tensor([], dtype="int32", device=device)
    shape = utils.astensor1d(size, _ref, dtype="int32", device=device)
    (output,) = apply(op, shape)
    return output


def _gamma(
    shape: Union[Tensor, float],
    scale: Union[Tensor, float],
    size: Optional[Iterable[int]],
    seed: int,
    handle: int,
) -> Tensor:
    handle_cn = None if handle == 0 else _get_rng_handle_compnode(handle)
    if not isinstance(shape, Tensor):
        assert shape > 0, "Gamma is not defined when shape <= 0"
        shape = Tensor(shape, dtype="float32", device=handle_cn)
    if not isinstance(scale, Tensor):
        assert scale > 0, "Gamma is not defined when scale <= 0"
        scale = Tensor(scale, dtype="float32", device=handle_cn)
    assert (
        handle_cn is None or handle_cn == shape.device
    ), "The shape ({}) must be the same device with handle ({})".format(
        shape.device, handle_cn
    )
    assert (
        handle_cn is None or handle_cn == scale.device
    ), "The scale ({}) must be the same device with handle ({})".format(
        scale.device, handle_cn
    )
    if isinstance(size, int) and size != 0:
        size = (size,)
    shape, scale = _broadcast_tensors_with_size([shape, scale], size)
    op = GammaRNG(seed=seed, handle=handle)
    (output,) = apply(op, shape, scale)
    return output


def _beta(
    alpha: Union[Tensor, float],
    beta: Union[Tensor, float],
    size: Optional[Iterable[int]],
    seed: int,
    handle: int,
) -> Tensor:
    handle_cn = None if handle == 0 else _get_rng_handle_compnode(handle)
    if not isinstance(alpha, Tensor):
        assert alpha > 0, "Beta is not defined when alpha <= 0"
        alpha = Tensor(alpha, dtype="float32", device=handle_cn)
    if not isinstance(beta, Tensor):
        assert beta > 0, "Beta is not defined when beta <= 0"
        beta = Tensor(beta, dtype="float32", device=handle_cn)
    assert (
        handle_cn is None or handle_cn == alpha.device
    ), "The alpha ({}) must be the same device with handle ({})".format(
        alpha.device, handle_cn
    )
    assert (
        handle_cn is None or handle_cn == beta.device
    ), "The beta ({}) must be the same device with handle ({})".format(
        beta.device, handle_cn
    )
    if isinstance(size, int) and size != 0:
        size = (size,)
    alpha, beta = _broadcast_tensors_with_size([alpha, beta], size)
    op = BetaRNG(seed=seed, handle=handle)
    (output,) = apply(op, alpha, beta)
    return output


def _poisson(
    lam: Union[Tensor, float], size: Optional[Iterable[int]], seed: int, handle: int
) -> Tensor:
    handle_cn = None if handle == 0 else _get_rng_handle_compnode(handle)
    if not isinstance(lam, Tensor):
        assert lam > 0, "Poisson is not defined when lam <= 0"
        lam = Tensor(lam, dtype="float32", device=handle_cn)
    if isinstance(size, int) and size != 0:
        size = (size,)
    assert (
        handle_cn is None or handle_cn == lam.device
    ), "The lam ({}) must be the same device with handle ({})".format(
        lam.device, handle_cn
    )
    (lam,) = _broadcast_tensors_with_size([lam], size)
    op = PoissonRNG(seed=seed, handle=handle)
    (output,) = apply(op, lam)
    return output


def _permutation(n: int, seed: int, device: str, handle: int, dtype: str) -> Tensor:
    assert isinstance(n, int)
    assert n >= 0, "Permutation is not defined when n < 0"
    size = (n,)
    op = PermutationRNG(seed=seed, handle=handle, dtype=dtype)
    _ref = Tensor([], dtype="int32", device=device)
    shape = utils.astensor1d(size, _ref, dtype="int32", device=device)
    (output,) = apply(op, shape)
    return output


def _shuffle(inp: Tensor, seed: int, handle: int) -> Tensor:
    assert inp.size > 0, "size needs to be greater than 0"
    op = ShuffleRNG(seed=seed, handle=handle)
    output, _ = apply(op, inp)
    inp._reset(output)


class RNG:

    r""":class:`RNG` exposes a number of methods for generating random numbers.

    Args:
        seed: random seed used to initialize the pseudo-random number generator. Default: None
        device: the device of generated tensor. Default: None


    Examples:

        .. testcode::

            import megengine.random as rand
            rng = rand.RNG(seed=100)
            x = rng.uniform(size=(2, 2))
            print(x.numpy())

        Outputs:

        .. testoutput::
            :options: +SKIP

            [[0.84811664 0.6147553 ]
             [0.59429836 0.64727545]]

    """

    def __init__(self, seed: int = None, device: str = None):
        self._device = device if device else get_default_device()
        if seed is not None:
            self._seed = seed
            self._handle = _new_rng_handle(self._device, self._seed)
        else:
            self._seed = _get_global_rng_seed
            self._handle = 0
            self._device = None

    def uniform(
        self, low: float = 0, high: float = 1, size: Optional[Iterable[int]] = None
    ):
        r"""Random variable with uniform distribution $U(0, 1)$.

        Args:
            low: lower range. Default: 0
            high: upper range. Default: 1
            size: the size of output tensor. Default: None

        Returns:
            the output tensor.

        Examples:

            .. testcode::

                import megengine as mge
                import megengine.random as rand

                x = rand.uniform(size=(2, 2))
                print(x.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [[0.91600335 0.6680226 ]
                 [0.2046729  0.2769141 ]]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _uniform(
            low=low,
            high=high,
            size=size,
            seed=_seed,
            device=self._device,
            handle=self._handle,
        )

    def normal(
        self, mean: float = 0, std: float = 1, size: Optional[Iterable[int]] = None
    ):
        r"""Random variable with Gaussian distribution :math:`N(\mu, \sigma)`.

        Args:
            mean: the mean or expectation of the distribution. Default: 0
            std: the standard deviation of the distribution (variance = :math:`\sigma ^ 2`).
                Default: 1
            size: the size of output tensor. Default: None

        Returns:
            the output tensor.

        Examples:

            .. testcode::

                import megengine as mge
                import megengine.random as rand

                x = rand.normal(mean=0, std=1, size=(2, 2))
                print(x.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [[-1.4010863  -0.9874344 ]
                 [ 0.56373274  0.79656655]]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _normal(
            mean=mean,
            std=std,
            size=size,
            seed=_seed,
            device=self._device,
            handle=self._handle,
        )

    def gamma(
        self,
        shape: Union[Tensor, float],
        scale: Union[Tensor, float] = 1,
        size: Optional[Iterable[int]] = None,
    ):
        r"""Random variable with Gamma distribution :math:`\Gamma(k, \theta)`.

        The corresponding probability density function is

        .. math::

            p(x)=x^{k-1} \frac{e^{-x / \theta}}{\theta^{k} \Gamma(k)}
            \quad \text { for } x>0 \quad k, \theta>0,

        where :math:`\Gamma(k)` is the gamma function,

        .. math::
            \Gamma(k)=(k-1) !  \quad \text { for } \quad k>0.

        Args:
            shape: the shape parameter (sometimes designated "k") of the distribution.
                Must be non-negative.
            scale: the scale parameter (sometimes designated "theta") of the distribution.
                Must be non-negative. Default: 1
            size: the size of output tensor. If shape and scale are scalars and given size is, e.g.,
                `(m, n)`, then the output shape is `(m, n)`. If shape or scale is a Tensor and given size
                is, e.g., `(m, n)`, then the output shape is `(m, n) + broadcast(shape, scale).shape`.
                The broadcast rules are consistent with `numpy.broadcast`. Default: None

        Returns:
            the output tensor.

        Examples:

            .. testcode::

                import megengine as mge
                import megengine.random as rand

                x = rand.gamma(shape=2, scale=1, size=(2, 2))
                print(x.numpy())

                shape = mge.Tensor([[ 1],
                                    [10]], dtype="float32")
                scale = mge.Tensor([1,5], dtype="float32")

                x = rand.gamma(shape=shape, scale=scale)
                print(x.numpy())

                x = rand.gamma(shape=shape, scale=scale, size=2)
                print(x.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [[1.5064533  4.0689363 ]
                 [0.71639484 1.4551026 ]]

                [[ 0.4352188 11.399335 ]
                 [ 9.1888    52.009277 ]]

                [[[ 1.1726005   3.9654975 ]
                  [13.656933   36.559006  ]]
                 [[ 0.25848487  2.5540342 ]
                  [11.960409   21.031536  ]]]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _gamma(
            shape=shape, scale=scale, size=size, seed=_seed, handle=self._handle
        )

    def beta(
        self,
        alpha: Union[Tensor, float],
        beta: Union[Tensor, float],
        size: Optional[Iterable[int]] = None,
    ):
        r"""Random variable with Beta distribution :math:`\operatorname{Beta}(\alpha, \beta)`.

        The corresponding probability density function is

        .. math::

            p(x)=\frac{1}{\mathrm{~B}(\alpha, \beta)} x^{\alpha-1}(1-x)^{\beta-1}
            \quad \text { for } \alpha, \beta>0,

        where :math:`\mathrm{~B}(\alpha, \beta)` is the beta function,

        .. math::

            \mathrm{~B}(\alpha, \beta)=\int_{0}^{1} t^{\alpha-1}(1-t)^{\beta-1} d t.

        Args:
            alpha: the alpha parameter of the distribution. Must be non-negative.
            beta: the beta parameter of the distribution. Must be non-negative.
            size: the size of output tensor. If alpha and beta are scalars and given size is, e.g.,
                `(m, n)`, then the output shape is `(m, n)`. If alpha or beta is a Tensor and given size
                is, e.g., `(m, n)`, then the output shape is `(m, n) + broadcast(alpha, beta).shape`.

        Returns:
            the output tensor.

        Examples:

            .. testcode::

                import megengine as mge
                import megengine.random as rand

                x = rand.beta(alpha=2, beta=1, size=(2, 2))
                print(x.numpy())

                alpha = mge.Tensor([[0.5],
                                    [  3]], dtype="float32")
                beta = mge.Tensor([0.5,5], dtype="float32")

                x = rand.beta(alpha=alpha, beta=beta)
                print(x.numpy())

                x = rand.beta(alpha=alpha, beta=beta, size=2)
                print(x.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [[0.582565   0.91763186]
                 [0.86963767 0.6088103 ]]

                [[0.41503012 0.16438372]
                 [0.90159506 0.47588003]]

                [[[0.55195075 0.01111084]
                  [0.95298755 0.25048104]]
                 [[0.11680304 0.13859665]
                  [0.997879   0.43259275]]]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _beta(alpha=alpha, beta=beta, size=size, seed=_seed, handle=self._handle)

    def poisson(self, lam: Union[float, Tensor], size: Optional[Iterable[int]] = None):
        r"""Random variable with poisson distribution :math:`\operatorname{Poisson}(\lambda)`.

        The corresponding probability density function is

        .. math::

            f(k ; \lambda)=\frac{\lambda^{k} e^{-\lambda}}{k !},

        where k is the number of occurrences :math:`({\displaystyle k=0,1,2...})`.

        Args:
            lam: the lambda parameter of the distribution. Must be non-negative.
            size: the size of output tensor. If lam is a scalar and given size is, e.g., `(m, n)`,
                then the output shape is `(m, n)`. If lam is a Tensor with shape `(k, v)` and given
                size is, e.g., `(m, n)`, then the output shape is `(m, n, k, v)`. Default: None.

        Returns:
            the output tensor.



        Examples:

            .. testcode::

                import megengine as mge
                import megengine.random as rand

                x = rand.poisson(lam=2., size=(1, 3))
                print(x.numpy())

                lam = mge.Tensor([[1.,1.],
                                [10,10]], dtype="float32")

                x = rand.poisson(lam=lam)
                print(x.numpy())

                x = rand.poisson(lam=lam, size=(1,3))
                print(x.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [[3. 1. 3.]]

                [[ 2.  2.]
                 [12. 11.]]

                [[[[ 1.  1.]
                   [11.  4.]]
                  [[ 0.  0.]
                   [ 9. 13.]]
                  [[ 0.  1.]
                   [ 7. 12.]]]]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _poisson(lam=lam, size=size, seed=_seed, handle=self._handle)

    def permutation(self, n: int, *, dtype: str = "int32"):
        r"""Generates a random permutation of integers from :math:`0` to :math:`n - 1`.

        Args:
            n: the upper bound. Must be larger than 0.
            dtype: the output data type. int32, int16 and float32 are supported. Default: int32

        Returns:
            the output tensor.

        Examples:

            .. testcode::

                import megengine as mge
                import megengine.random as rand

                x = rand.permutation(n=10, dtype="int32")
                print(x.numpy())

                x = rand.permutation(n=10, dtype="float32")
                print(x.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [4 5 0 7 3 8 6 1 9 2]
                [3. 4. 9. 0. 6. 8. 7. 1. 5. 2.]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _permutation(
            n=n, seed=_seed, device=self._device, handle=self._handle, dtype=dtype
        )

    def shuffle(self, inp: Tensor):
        r"""Modify a sequence in-place by shuffling its contents. 
        This function only shuffles the Tensor along the first axis of a multi-dimensional Tensor.
        The order of sub-Tensors is changed but their contents remains the same.

        Args:
            inp: input tensor.

        Examples:

            .. testcode::

                import numpy as np
                import megengine as mge
                import megengine.random as rand

                x = mge.tensor(np.arange(10))
                rand.shuffle(x)
                print(x.numpy())
                y = mge.tensor(np.arange(18)).reshape(6,3)
                rand.shuffle(y)
                print(y.numpy())

            Outputs:

            .. testoutput::
                :options: +SKIP

                [7 9 3 0 8 2 4 5 6 1]
                [[12. 13. 14.]
                 [ 3.  4.  5.]
                 [15. 16. 17.]
                 [ 0.  1.  2.]
                 [ 9. 10. 11.]
                 [ 6.  7.  8.]]
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        _shuffle(inp=inp, seed=_seed, handle=self._handle)

    def __del__(self):
        if self._handle != 0:
            _delete_rng_handle(self._handle)


def _default_rng():
    r"""Default constructor for :class:`RNG`."""
    return RNG(seed=None, device=None)


_default_handle = _default_rng()

uniform = _default_handle.uniform
normal = _default_handle.normal
gamma = _default_handle.gamma
beta = _default_handle.beta
poisson = _default_handle.poisson
permutation = _default_handle.permutation
shuffle = _default_handle.shuffle


def _random_seed_generator():
    assert _rng
    while True:
        yield _rng.random_raw()


def seed(seed: int):
    global _rng  # pylint: disable=global-statement
    _rng = MT19937(seed=seed)
    _set_global_rng_seed(seed)


seed(int(time.time()))
