import math
import numbers
from functools import lru_cache

import numpy as np

from ..core.ops import builtin
from ..core.tensor.utils import subgraph_fn
from ..functional import (
    arange,
    broadcast_to,
    clip,
    flatten,
    full_like,
    gather,
    mul,
    reshape,
    zeros,
)
from ..functional.elemwise import abs, add, log
from ..functional.math import sign
from ..functional.nn import conv2d, pad
from ..functional.tensor import broadcast_to
from ..random.rng import RNG
from ..tensor import Tensor
from .module import Module


class AdditiveElemwise(Module):
    def __init__(self, per_channel=False, **kwargs):
        self._per_channel = per_channel
        super().__init__(**kwargs)

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, but got {}".format(type(inp))
        if self._per_channel is True:
            noise = self.sample(inp.shape).to(inp.device)
        elif self._per_channel is False:
            if inp.format == "nchw":
                N, C, H, W = inp.shape
                c_noise = self.sample((N, 1, H, W))
            # TODO: fix this code because the inp.shape always nchw output, even if format is "nhwc", cjs.
            elif inp.format == "nhwc":
                N, H, W, C = inp.shape
                c_noise = self.sample((N, H, W, 1))
            else:
                raise RuntimeError(
                    "expect you create Tensor with format specified while per_channel is False, got format is {}".format(
                        inp.format
                    )
                )
            noise = broadcast_to(c_noise, inp.shape).to(inp.device)
        else:
            raise NotImplementedError("float point type per channel haven't impl")
        return add(inp, noise)

    def sample(self, size):
        raise NotImplementedError()

    @property
    def per_channel(self):
        return self._per_channel

    @per_channel.setter
    def per_channel(self, per_channel):
        self._per_channel = per_channel


class AdditiveLaplaceNoise(AdditiveElemwise):
    r"""Add random laplace noise to the input data.
    Laplace noise is generated with given mean and std, sampled from Laplace distribution
    ref to this page to learn more: https://en.wikipedia.org/wiki/Laplace_distribution

    Args:
        mean: laplace mean used to generate noise.
        std: laplace standard deviation used to generate noise.
        per_channel: whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
        seed: random number seed of generator
    """

    def __init__(self, mean=0.0, std=1.0, per_channel=False, seed=None):
        assert seed is None or isinstance(seed, int)
        super().__init__(per_channel)
        self.mean = Tensor(mean, dtype=np.float32)
        self.std = Tensor(std, dtype=np.float32)
        self.rng_func = RNG(seed).uniform
        self.finfo = np.finfo(np.dtype(self.mean.dtype))
        self._seed = seed

    def sample(self, size):
        u = self.rng_func((self.finfo.eps - 1).item(), 1, size)
        value = self.mean - self.std * sign(u) * log(1 - abs(u))
        return value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int)
        self._seed = seed
        self.rng_func = RNG(seed).uniform


class AdditivePoissonNoise(AdditiveElemwise):
    r"""Add random poisson noise to the input data.
    poission noise is generated with given mean and std.

    Args:
        lam: lam parameter of poisson distribution used to generate noise.
        per_channel: whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
        seed: random number seed of generator
    """

    def __init__(self, lam=1.0, per_channel=False, seed=None):
        assert seed is None or isinstance(seed, int)
        super().__init__(per_channel)
        self.lam = Tensor(lam, dtype=np.float32)
        self.rng_func = RNG(seed).poisson
        self._seed = seed

    def sample(self, size):
        value = self.rng_func(self.lam, size)
        return value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int)
        self._seed = seed
        self.rng_func = RNG(seed).poisson


class AdditiveGaussianNoise(AdditiveElemwise):
    r"""Add random gaussian noise to the input data.
    Gaussian noise is generated with given mean and std.

    Args:
        mean: gaussian mean used to generate noise.
        std: gaussian standard deviation used to generate noise.
        per_channel: whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
        seed: random number seed of generator
    """

    def __init__(self, mean=0.0, std=1.0, per_channel=False, seed=None):
        assert seed is None or isinstance(seed, int)
        super().__init__(per_channel)
        self.mean = Tensor(mean, dtype=np.float32)
        self.std = Tensor(std, dtype=np.float32)
        self.rng_func = RNG(seed).normal
        self._seed = seed

    def sample(self, size):
        value = self.rng_func(self.mean, self.std, size)
        return value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        assert isinstance(seed, int)
        self._seed = seed
        self.rng_func = RNG(seed).normal


def _get_value_range_of_dtype(dtype):
    if not dtype.kind in ["f", "u", "i", "b"]:
        raise Exception(
            "Cannot estimate value range of dtype '%s' "
            "(type: %s)" % (str(dtype), type(dtype))
        )
    if dtype.kind == "f":
        finfo = np.finfo(dtype)
        value_min = finfo.min
        value_mid = 0.0
        value_max = finfo.max
    if dtype.kind == "u":
        iinfo = np.iinfo(dtype)
        value_min = iinfo.min
        value_mid = iinfo.min + 0.5 * iinfo.max
        value_max = iinfo.max
    if dtype.kind == "i":
        iinfo = np.iinfo(dtype)
        value_min = iinfo.min
        value_mid = -0.5
        value_max = iinfo.max
    if dtype.kind == "b":
        value_min = 0
        value_mid = None
        value_max = 1
    return value_min, value_mid, value_max


def _check_out_dtype(inp, input_dtype):
    if input_dtype.name == "bool":
        inp = inp > 0.5
    elif input_dtype.name in ["uint8", "uint16", "int8", "int16", "int32", "float16"]:
        min_dtype, _, max_dtype = _get_value_range_of_dtype(input_dtype)
        inp = clip(inp, min_dtype, max_dtype)
        inp = inp.astype(input_dtype)
    return inp


class ActiveBlur(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, but got {}".format(type(inp))
        if inp.format == "nchw" or inp.format == "default":
            _norm_inp = inp
            N, C, H, W = inp.shape
        else:
            raise RuntimeError(
                "expect you create Tensor with format NCHW, got format is {}".format(
                    inp.format
                )
            )
        kernel = self.get_kernel(_norm_inp, C)
        pad_inp = pad(
            _norm_inp, pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode="reflect"
        )
        result = conv2d(pad_inp, kernel, groups=C)
        result = _check_out_dtype(result, inp.dtype)
        return result

    def _get_parameter(self, param):
        if isinstance(param, bool):
            raise TypeError("The input parameter cannot be of bool value type. ")
        if isinstance(param, (numbers.Integral, numbers.Real)):
            return float(param)
        elif isinstance(param, tuple):
            assert len(param) == 2, (
                "Expected parameter with type tuple to have exactly two "
                "entries, but got %d." % len(param)
            )
            param = self.rng_func(param[0], param[1])
            return float(param)
        else:
            raise TypeError("The input parameter has a wrong type. ")

    def get_kernel(self, inp, c):
        raise NotImplementedError()


@lru_cache(maxsize=None)
def _get_EmbossKernel_op(alpha, strength, *, dtype=None, device=None):
    @subgraph_fn(
        "EmbossKernel", dtype=dtype, device=device, nr_inputs=2, gopt_level=None,
    )
    def EmbossKernel(input, f, c):
        inp_e, inp_n = input[0:2]
        c_alp = c(alpha, dtype="float32")
        c_sub_alp = c(1 - alpha, dtype="float32")
        c_stg = c(strength, dtype="float32")
        c_1 = c(1, dtype="int32")
        c_2 = c(2, dtype="int32")
        c_3 = c(3, dtype="int32")

        def _subtensor(src, axis, begin, end):
            items = ((axis, (begin is not None), (end is not None), False, False),)
            args = ()
            if begin is not None:
                args += (begin,)
            if end is not None:
                args += (end,)
            return f(builtin.Subtensor(items=items), src, *args)

        def _kernel_init(x):
            k_1 = _subtensor(x, 0, None, c_1)
            k_2 = _subtensor(x, 0, c_1, c_2)
            k_3 = _subtensor(x, 0, c_2, c_3)
            k_11 = f("-", _subtensor(k_1, 1, None, c_1), c_stg)
            k_12_21 = f("-", _subtensor(k_1, 1, c_1, c_2), c_stg)
            k_23_32 = f("+", _subtensor(k_2, 1, c_2, c_3), c_stg)
            k_33 = f("+", _subtensor(k_3, 1, c_2, c_3), c_stg)
            k_13 = _subtensor(k_1, 1, c_2, c_3)
            k_22 = _subtensor(k_2, 1, c_1, c_2)
            k_31 = _subtensor(k_3, 1, None, c_1)
            nk_1 = f(builtin.Concat(axis=1), k_11, k_12_21, k_13,)
            nk_2 = f(builtin.Concat(axis=1), k_12_21, k_22, k_23_32,)
            nk_3 = f(builtin.Concat(axis=1), k_31, k_23_32, k_33,)
            return f(builtin.Concat(axis=0), nk_1, nk_2, nk_3,)

        def _kernel_calc(k_e, k_n):
            k1 = f("*", k_n, c_sub_alp)
            k2 = f("*", k_e, c_alp)
            return f("+", k1, k2)

        kernel_effect = _kernel_init(inp_e)
        kernel = _kernel_calc(kernel_effect, inp_n)
        return (kernel,), (False,)

    return EmbossKernel


class Emboss(ActiveBlur):
    r"""overlay emboss effect and alpha-blend the result with the original input
    The embossed version pronounces highlights and shadows, enhances the high-frequency information of the image, and retains the low-frequency information of the image

    Args:
        alpha: adjust visibility of embossed images. number or tuple of number,  At ``0.0``, only the original image is visible, at ``1.0`` only its embossed version is visible. If a tuple ``(a, b)``, a random value will be sampled from the interval ``[a, b)``.
        strength: emboss strength.Sane values are somewhere in the interval ``[0.0, 2.0)`` with ``1.0``, number or tuple of number,  If a tuple ``(a, b)``, a random value will be sampled from the interval ``[a, b)``.
        seed: random number seed of generator
    
    Examples:
        >>> import numpy as np
        >>> inp = mge.tensor(np.random.randint(0, 255, size=(160,3,128,128)).astype("float32"))
        >>> aug = mge.module.Emboss(alpha=(0.6, 0.8), strength=(0.6, 0.8), seed=1)
        >>> out = aug(inp)
    """

    def __init__(self, alpha, strength, seed=None):
        assert seed is None or isinstance(seed, int)
        super().__init__()
        self.alpha = alpha
        self.strength = strength
        self.rng_func = RNG(seed).uniform
        self.seed = seed
        self.matrix_nochange = Tensor(
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        )
        self.matrix_effect = Tensor(
            np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        )

    def get_kernel(self, inp, c):
        alpha = self._get_parameter(self.alpha)
        strength = self._get_parameter(self.strength)

        get_kernel_fn = _get_EmbossKernel_op(
            alpha,
            strength,
            dtype=self.matrix_effect.dtype,
            device=self.matrix_effect.device,
        )
        kernel, *_ = get_kernel_fn(self.matrix_effect, self.matrix_nochange)
        kernel = broadcast_to(kernel, (c, 1, 1, kernel.shape[0], kernel.shape[1]))
        return kernel


@lru_cache(maxsize=None)
def _get_SharpenKernel_op(alpha, lightness, *, dtype=None, device=None):
    @subgraph_fn(
        "SharpenKernel", dtype=dtype, device=device, nr_inputs=2, gopt_level=None,
    )
    def SharpenKernel(input, f, c):
        inp_e, inp_n = input[0:2]
        c_alp = c(alpha, dtype="float32")
        c_sub_alp = c(1 - alpha, dtype="float32")
        c_lts = c(lightness, dtype="float32")
        c_1 = c(1, dtype="int32")
        c_2 = c(2, dtype="int32")
        c_3 = c(3, dtype="int32")

        def _subtensor(src, axis, begin, end):
            items = ((axis, (begin is not None), (end is not None), False, False),)
            args = ()
            if begin is not None:
                args += (begin,)
            if end is not None:
                args += (end,)
            return f(builtin.Subtensor(items=items), src, *args)

        def _kernel_init(x):
            k_1 = _subtensor(x, 0, None, c_1)
            k_2 = _subtensor(x, 0, c_1, c_2)
            k_3 = _subtensor(x, 0, c_2, c_3)
            k_21 = _subtensor(k_2, 1, None, c_1)
            k_22 = f("+", _subtensor(k_2, 1, c_1, c_2), c_lts)
            k_23 = _subtensor(k_2, 1, c_2, c_3)
            nk_2 = f(builtin.Concat(axis=1), k_21, k_22, k_23,)
            return f(builtin.Concat(axis=0), k_1, nk_2, k_3,)

        def _kernel_calc(k_e, k_n):
            k1 = f("*", k_n, c_sub_alp)
            k2 = f("*", k_e, c_alp)
            return f("+", k1, k2)

        kernel_effect = _kernel_init(inp_e)
        kernel = _kernel_calc(kernel_effect, inp_n)
        return (kernel,), (False,)

    return SharpenKernel


class Sharpen(ActiveBlur):
    r"""Sharpen images and alpha-blend the result with the original input.

    Args:
        alpha: adjust visibility of sharpened images. number or tuple of number,  At ``0.0``, only the original image is visible, at ``1.0`` only its embossed version is visible. If a tuple ``(a, b)``, a random value will be sampled from the interval ``[a, b)``.
        lightness: controls the brightness of sharpened images. Sane values are somewhere in the interval ``[0.5, 2.0)`` with ``1.0``, number or tuple of number, If a tuple ``(a, b)``, a random value will be sampled from the interval ``[a, b)``.
        seed: random number seed of generator
    
    Examples:
        >>> import numpy as np
        >>> inp = mge.tensor(np.random.randint(0, 255, size=(160,3,128,128)).astype("float32"))
        >>> aug = mge.module.Sharpen(alpha=(0.6, 0.8), lightness=(0.6, 0.8), seed=1)
        >>> out = aug(inp)
    """

    def __init__(self, alpha, lightness, seed=None):
        assert seed is None or isinstance(seed, int)
        super().__init__()
        self.alpha = alpha
        self.lightness = lightness
        self.rng_func = RNG(seed).uniform
        self.seed = seed
        self.matrix_nochange = Tensor(
            np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        )
        self.matrix_effect = Tensor(
            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        )

    def get_kernel(self, inp, c):
        alpha = self._get_parameter(self.alpha)
        lightness = self._get_parameter(self.lightness)

        get_kernel_fn = _get_SharpenKernel_op(
            alpha,
            lightness,
            dtype=self.matrix_effect.dtype,
            device=self.matrix_effect.device,
        )
        kernel, *_ = get_kernel_fn(self.matrix_effect, self.matrix_nochange)
        kernel = broadcast_to(kernel, (c, 1, 1, kernel.shape[0], kernel.shape[1]))
        return kernel


class LinearContrast(Module):
    r"""Adjust contrast by scaling each pixel to ``127 + alpha*(v-127)``.

    Args:
        alpha: number or tuple of number. If a tuple ``(a, b)``, a random value will be sampled from the interval ``[a, b)``.
        per_channel:whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
        seed: random number seed of generator
    
    Examples:
        >>> import numpy as np
        >>> inp = mge.tensor(np.random.randint(0, 255, size=(160,3,128,128)).astype("float32"))
        >>> aug = mge.module.LinearContrast(alpha=(0.6, 0.8), per_channel=False, seed=1)
        >>> out = aug(inp)
    """

    def __init__(self, alpha, per_channel=False, seed=None):
        super().__init__()
        self.alpha = alpha
        self.seed = seed
        self.per_channel = per_channel
        self.rng_func = RNG(seed).uniform

    def _get_parameter(self, param, size):
        if isinstance(param, bool):
            raise TypeError("The input parameter cannot be of bool value type. ")
        if isinstance(param, (numbers.Integral, numbers.Real)):
            value = zeros(size, dtype="float32")
            value = full_like(value, param)
            return value
        elif isinstance(param, tuple):
            assert len(param) == 2, (
                "Expected parameter with type tuple to have exactly two "
                "entries, but got %d." % len(param)
            )
            value = self.rng_func(param[0], param[1], size)
            return value
        else:
            raise TypeError("The input parameter has a wrong type. ")

    def _get_table(self, size):
        shape = (size, 1)
        alpha = self._get_parameter(self.alpha, shape)
        table = arange(255).astype("float32")
        table = broadcast_to(table, (size, 255))
        table = 127 + mul((table - 127), alpha)
        return clip(table, 0, 255)

    def forward(self, inp: Tensor) -> Tensor:
        if inp.dtype.name == "uint8":
            if self.per_channel is True:
                flatten_inp = reshape(
                    inp, (inp.shape[0] * inp.shape[1], inp.shape[2] * inp.shape[3])
                ).astype("int32")
            else:
                flatten_inp = flatten(inp, 1).astype("int32")
            table = self._get_table(flatten_inp.shape[0])
            result = gather(table, 1, flatten_inp)
            result = reshape(result, inp.shape).astype("uint8")
            return result
        else:
            input_dtype = inp.dtype
            _, center_value, _ = _get_value_range_of_dtype(input_dtype)
            if self.per_channel is True:
                size = (inp.shape[0], inp.shape[1], 1, 1)
            else:
                size = (inp.shape[0], 1, 1, 1)
            alpha = self._get_parameter(self.alpha, size)
            if input_dtype.kind in ["u", "i"]:
                center_value = int(center_value)
            result = center_value + mul(inp.astype("float32") - center_value, alpha)
            result = result.astype(input_dtype)
            return result
