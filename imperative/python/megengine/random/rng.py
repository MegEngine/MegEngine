# -*- coding: utf-8 -*-
import collections
import time
from typing import Iterable, Optional, Union

from numpy.random import MT19937

from .. import Tensor
from ..core._imperative_rt.core2 import apply
from ..core._imperative_rt.core2 import sync as _sync
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
    if low == 0 and high == 1:
        return output
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
    return output


class RNG:

    r""":class:`RNG` exposes a number of methods for generating random numbers.

    Args:
        seed: random seed used to initialize the pseudo-random number generator. Default: None
        device: the device of generated tensor. Default: None


    Examples:
        >>> import megengine.random as rand
        >>> rng = rand.RNG(seed=100)
        >>> x = rng.uniform(size=(2, 2))
        >>> x.numpy()   # doctest: +SKIP
        array([[0.84811664, 0.6147553 ],
               [0.59429836, 0.64727545]], dtype=float32)
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
            >>> import megengine.random as rand
            >>> x = rand.uniform(size=(2, 2))
            >>> x.numpy()   # doctest: +SKIP
            array([[0.28603864, 0.3156649 ],
                   [0.42066026, 0.9805052 ]], dtype=float32)
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
            >>> import megengine.random as rand
            >>> x = rand.normal(mean=0, std=1, size=(2, 2))
            >>> x.numpy()   # doctest: +SKIP
            array([[ 1.5534291 , -0.28356555],
                   [ 2.2230418 , -0.92425716]], dtype=float32)
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
            >>> import megengine.random as rand
            >>> x = rand.gamma(shape=2, scale=1, size=(2, 2))
            >>> x.numpy()   # doctest: +SKIP
            array([[0.97447544, 1.5668875 ],
                   [1.0069491 , 0.3078318 ]], dtype=float32)
            >>> shape = mge.Tensor([[ 1],
            ...                     [10]], dtype="float32")
            >>> scale = mge.Tensor([1,5], dtype="float32")
            >>> x = rand.gamma(shape=shape, scale=scale)
            >>> x.numpy()   # doctest: +SKIP
            array([[ 0.11312152,  3.0799196 ],
                   [10.973469  , 29.596972  ]], dtype=float32)
            >>> x = rand.gamma(shape=shape, scale=scale, size=2)
            >>> x.numpy()   # doctest: +SKIP
            array([[[4.35868073e+00, 1.22415285e+01],
                    [1.02696848e+01, 4.19773598e+01]],

                   [[7.73875117e-02, 6.06766164e-01],
                    [1.22881927e+01, 8.13445740e+01]]], dtype=float32)
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
            >>> import megengine.random as rand
            >>> x = rand.beta(alpha=2, beta=1, size=(2, 2))
            >>> x.numpy()   # doctest: +SKIP
            array([[0.6172312 , 0.9789006 ],
                   [0.50004643, 0.9775796 ]], dtype=float32)
            >>> alpha = mge.Tensor([[0.5],
            ...                     [  3]], dtype="float32")
            >>> beta = mge.Tensor([0.5,5], dtype="float32")
            >>> x = rand.beta(alpha=alpha, beta=beta)
            >>> x.numpy()   # doctest: +SKIP
            array([[0.0075407 , 0.1275094 ],
                   [0.96331763, 0.22299217]], dtype=float32)
            >>> x = rand.beta(alpha=alpha, beta=beta, size=2)
            >>> x.numpy()   # doctest: +SKIP
            array([[[0.46863747, 0.13819647],
                    [0.8646759 , 0.16014215]],

                   [[0.0682759 , 0.04448463],
                    [0.97733796, 0.19206746]]], dtype=float32)
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
            >>> import megengine.random as rand
            >>> x = rand.poisson(lam=2., size=(1, 3))
            >>> x.numpy()   # doctest: +SKIP
            array([[1., 2., 2.]], dtype=float32)
            >>> lam = mge.Tensor([[1.,1.],
            ...                 [10,10]], dtype="float32")
            >>> x = rand.poisson(lam=lam)
            >>> x.numpy()   # doctest: +SKIP
            array([[ 1.,  2.],
                   [11., 11.]], dtype=float32)
            >>> x = rand.poisson(lam=lam, size=(1,3))
            >>> x.numpy()   # doctest: +SKIP
            array([[[[ 2.,  1.],
                     [10.,  8.]],

                    [[ 5.,  2.],
                     [10., 10.]],

                    [[ 1.,  2.],
                     [ 8., 10.]]]], dtype=float32)
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        return _poisson(lam=lam, size=size, seed=_seed, handle=self._handle)

    def permutation(self, n: Union[int, Tensor], *, dtype: str = "int32"):
        r"""Randomly permute a sequence, or return a permuted range.
            If ``n`` is a multi-dimensional tensor, it is only shuffled along its first index.

        Args:
            n: If ``n`` is an integer, random permutation of integers from :math:`0` to :math:`n - 1`. 
                If ``n`` is an tensor, make a copy and shuffle the elements randomly.
            dtype: the output data type when ``n`` is an integer.
                int32, int16 and float32 are supported. Default: int32

        Returns:
            the output tensor.

        Examples:
            >>> import numpy as np
            >>> import megengine.random as rand
            >>> x = rand.permutation(10, dtype="int32")
            >>> x.numpy()   # doctest: +SKIP
            array([8, 4, 0, 3, 5, 6, 2, 1, 7, 9], dtype=int32)
            >>> x = rand.permutation(10, dtype="float32")
            >>> x.numpy()   # doctest: +SKIP
            array([1., 3., 0., 2., 4., 8., 7., 9., 6., 5.], dtype=float32)
            >>> x = mge.tensor(np.arange(18)).reshape(6,3)
            >>> x = rand.permutation(x)
            >>> x.numpy()   # doctest: +SKIP
            array([[15, 16, 17],
                   [ 6,  7,  8],
                   [ 0,  1,  2],
                   [ 3,  4,  5],
                   [12, 13, 14],
                   [ 9, 10, 11]], dtype=int32)
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        if isinstance(n, int):
            return _permutation(
                n=n, seed=_seed, device=self._device, handle=self._handle, dtype=dtype
            )
        assert isinstance(n, Tensor)
        return _shuffle(inp=n, seed=_seed, handle=self._handle)

    def shuffle(self, inp: Tensor):
        r"""Modify a sequence in-place by shuffling its contents. 
        This function only shuffles the Tensor along the first axis of a multi-dimensional Tensor.
        The order of sub-Tensors is changed but their contents remains the same.

        Args:
            inp: input tensor.

        Examples:
            >>> import numpy as np
            >>> import megengine.random as rand
            >>> x = mge.tensor(np.arange(10))
            >>> rand.shuffle(x)
            >>> x.numpy()   # doctest: +SKIP
            array([4, 5, 9, 6, 2, 8, 1, 0, 3, 7], dtype=int32)
            >>> y = mge.tensor(np.arange(18)).reshape(6,3)
            >>> rand.shuffle(y)
            >>> y.numpy()   # doctest: +SKIP
            array([[ 3,  4,  5],
                   [ 6,  7,  8],
                   [15, 16, 17],
                   [ 0,  1,  2],
                   [12, 13, 14],
                   [ 9, 10, 11]], dtype=int32)
        """
        _seed = self._seed() if callable(self._seed) else self._seed
        inp._reset(_shuffle(inp=inp, seed=_seed, handle=self._handle))

    def __del__(self):
        if self._handle != 0:
            # RNG op might execute after handle released due to async dispatch, so
            # we need sync before delete a handle to avoid memory leak or
            # use-after-free
            _sync()
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
