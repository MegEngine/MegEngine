import math
import numbers
from functools import lru_cache
from typing import Tuple, Union

import numpy as np

from ..core._imperative_rt.core2 import apply
from ..core.ops import builtin, custom
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
from ..functional.elemwise import abs, add, clip, floor, log, sqrt
from ..functional.math import sign
from ..functional.nn import conv2d, pad
from ..functional.tensor import broadcast_to, stack
from ..functional.vision import flip, remap, resize, rot90, rotate
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


class Mixup(Module):
    r"""Mixup is a data augmentation method that generates new training samples 
    by mixing the features and labels of two training samples at a certain ratio, 
    with the aim of improving the performance of the model.
    
    Args:
        beta: parameters of the Beta distribution will affect the ratio of the mixed samples
        seed: random number seed of generator

    Shape:
        - input_data: :math:`(N, C, H, W)` (now only support NCHW format tensor)
        - input_label: :math:`(N,)`
        - output_data: :math:`(N, C, H, W)` (same shape as input)
        - output_label: :math:`(N,)`
    
    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.Mixup(3)
            data1 = mge.tensor(np.random.randn(2, 3, 16, 16), dtype=np.float32)
            data2 = mge.tensor(np.random.randn(2, 3, 16, 16), dtype=np.float32)
            label1 = mge.tensor(np.random.randn(2,), dtype=np.float32)
            label2 = mge.tensor(np.random.randn(2,), dtype=np.float32)

            data, label = m(data1, label1, data2, label2)

            print(data.numpy().shape)
            print(label.numpy().shape)

        Outputs:

        .. code-block::

            (2, 3, 16, 16)
            (2,)
    """

    def __init__(self, beta, seed=None):
        assert seed is None or isinstance(seed, int)
        super(Mixup, self).__init__()
        self.beta_func = RNG(seed).beta
        self.beta = beta

    def sample(self, batch):
        return self.beta_func(self.beta, self.beta, size=(batch,))

    def forward(self, data1, label1, data2, label2):
        assert all(
            isinstance(inp, Tensor) for inp in [data1, label1, data2, label2]
        ), "expected input is megengine.Tensor"

        batch, C, H, W = data1.shape
        self.lamb = self.sample(batch)

        label = self.lamb * label1 + (1.0 - self.lamb) * label2

        data = (
            self.lamb.reshape(batch, 1, 1, 1) * data1
            + (1 - self.lamb).reshape(batch, 1, 1, 1) * data2
        )

        return data, label


class Cutmix(Module):
    r"""The Cutmix method is a technique that generates a random bounding box (bbox) based on a lambda value, 
    and assigns the values of the bounding box from another data sample to the current sample.
    
    Args:
        beta: parameters of the Beta distribution will affect the ratio of the mixed samples
        seed: random number seed of generator

    Shape:
        - input_data: :math:`(N, C, H, W)` (now only support NCHW format tensor)
        - input_label: :math:`(N,)`
        - output_data: :math:`(N, C, H, W)` (same shape as input)
        - output_label: :math:`(N,)`
    
    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.Cutmix(3)
            data1 = mge.tensor(np.random.randn(2, 3, 16, 16), dtype=np.float32)
            data2 = mge.tensor(np.random.randn(2, 3, 16, 16), dtype=np.float32)
            label1 = mge.tensor(np.random.randn(2,), dtype=np.float32)
            label2 = mge.tensor(np.random.randn(2,), dtype=np.float32)

            data, label = m(data1, label1, data2, label2)

            print(data.numpy().shape)
            print(label.numpy().shape)

        Outputs:

        .. code-block::

            (2, 3, 16, 16)
            (2,)
    """

    def __init__(self, beta, seed=None):
        assert seed is None or isinstance(seed, int)
        super(Cutmix, self).__init__()
        self.beta_func = RNG(seed).beta
        self.uniform_func = RNG(seed).uniform
        self.beta = beta
        self.custom_func = custom.cutmix_forward()

    def sample(self, batch):
        return self.beta_func(self.beta, self.beta, size=(batch,))

    def forward(self, data1, label1, data2, label2):
        assert all(
            isinstance(inp, Tensor) for inp in [data1, label1, data2, label2]
        ), "expected input is megengine.Tensor"

        batch, C, H, W = data1.shape

        self.lamb = self.sample(batch)
        lamb_rat = sqrt(1.0 - self.lamb)

        label = self.lamb * label1 + (1.0 - self.lamb) * label2

        self.cut_h = floor(H * lamb_rat)
        self.cut_w = floor(W * lamb_rat)

        # uniform
        self.cx = self.uniform_func(high=H, size=(batch,))
        self.cx = floor(self.cx)
        self.cy = self.uniform_func(high=W, size=(batch,))
        self.cy = floor(self.cy)

        data = apply(
            self.custom_func, data1, data2, self.cx, self.cy, self.cut_h, self.cut_w
        )[0]

        return data, label


class CropAndPad(Module):
    r"""Pad or Crop the input tensor with given percent value and resize to original shape.
    Percent supports mge.tensor type, which is the scaling ratio for each image. If value > 0, 
    image will be padded. If value < 0, image will be cropped.
    
    Args:
        mode: interpolation methods, acceptable values are:"bilinear", "nearest". Default: "bilinear"
        align_corners: this only has an effect when ``mode`` is "bilinear". If set to ``True``, the input
            and output tensors are aligned by the center points of their corner
            pixels, preserving the values at the corner pixels. If set to ``False``,
            the input and output tensors are aligned by the corner points of their
            corner pixels, and the interpolation uses edge value padding for
            out-of-boundary values, making this operation *independent* of input size. Default: "False"

    Shape:
        - input: :math:`(N, C, H, W)` (now only support NCHW format tensor)
        - percent: :math:`(N,)` 
        - pad_val: :math:`(N,)`
    
    Examples:

        .. code-block::

            import numpy as np
            import megengine as mge
            import megengine.module as M

            m = M.CropAndPad()
            inp = mge.tensor(np.random.randn(2, 1, 160, 160), dtype=np.float32)
            percent = (0.2) * mge.tensor(np.random.random(128,)).astype("float32") - 0.1
            pad_val = inp.mean(axis=[3,2,1])

            out = m(inp, percent, pad_val)

            print(out.numpy().shape)

        Outputs:

        .. code-block::

            (2, 1, 160, 160)
    """

    def __init__(self, mode="bilinear", align_corners=False):
        super().__init__()
        assert mode in ["bilinear", "nearest"]
        self.custom_func = custom.cropandpad_forward(
            mode=mode, align_corners=align_corners
        )

    def forward(self, inp: Tensor, percent: Tensor, pad_val: Tensor = None) -> Tensor:
        if pad_val is None:
            pad_val = Tensor([])

        res = apply(self.custom_func, inp, percent, pad_val)[0]

        return res


class Flip(Module):
    r"""Reverse an n-dimensional tensor according to the given parameters

    Args:
        vertical(bool, optional): Flip vertically or not. Default: True.
        horizontal(bool, optional): Flip horizontally or not. Default: True.

    Examples:
        >>> import numpy as np
        >>> x = Tensor(np.arange(0, 4, dtype=np.float32).reshape(1, 2, 2, 1))
        >>> flip = mge.module.Flip(vertical=True, horizontal=True)
        >>> y = flip(x)
        >>> y.numpy()
        array([[[[3.],
                [2.]],
                [[1.],
                 [0.]]]], dtype=float32)
    """

    def __init__(self, vertical=True, horizontal=True):
        super().__init__()
        self.vertical = vertical
        self.horizontal = horizontal

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, got {}".format(type(inp))
        return flip(inp, self.vertical, self.horizontal)


class RandomHorizontalFlip(Module):
    r"""Horizontally flip the given image randomly with a given probability.
    Input format must be nhwc.

    Args:
        prob(float, optional): probability of the image being flipped. Default value is 0.5.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, inp):
        if np.random.rand(1) < self.prob:
            return flip(inp, horizontal=True)
        return inp


class RandomVerticalFlip(Module):
    r"""Vertically flip the given image randomly with a given probability.
    Input format must be nhwc.

    Args:
        prob(float, optional): probability of the image being flipped. Default value is 0.5.
    """

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def forward(self, inp):
        if np.random.rand(1) < self.prob:
            return flip(inp, vertical=True)
        return inp


class Resize(Module):
    r"""Resize the input image to the given size.
    Input image is expected to have [N, C, H, W] or [N, H, W, C] shape.

    Args:
        size(Tensor): Desired output size.
        format(String): Input tensor format;
        imode(String): Interpolation mode.
    
    Examples:
        >>> import numpy as np
        >>> x = Tensor(np.arange(0, 16, dtype=np.float32).reshape(1, 1, 4, 4))
        >>> size = Tensor([2, 2])
        >>> resize = mge.module.Resize(size)
        >>> y = resize(x)
        >>> y.numpy()
        array([[[[ 2.5,  4.5],
         [10.5, 12.5]]]], dtype=float32)
    """

    def __init__(self, size, format="NCHW", imode="linear"):
        super().__init__()
        self.size = Tensor(size)
        self.format = format
        self.imode = imode

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, got {}".format(type(inp))
        if self.format not in ["NCHW", "NHWC"]:
            raise RuntimeError(
                "expect input Tensor format to be NCHW or NHWC, got format {}".format(
                    self.format
                )
            )
        return resize(inp, self.size, self.format, self.imode)


class Rot90(Module):
    r"""Rotate an n-D tensor by 90 degrees in the plane specified by dims axis. 

    Args:
        clockwise(bool, optional): Rotate 90° clockwise or 90° counterclockwise. Default: True.

    Examples:
        >>> import numpy as np
        >>> x = Tensor(np.arange(0, 4, dtype=np.float32).reshape(1, 2, 2, 1))
        >>> rot90 = mge.module.Rot90(clockwise=True)
        >>> y = rot90(x)
        >>> y.numpy()
        array([[[[2.],
                [0.]],
                [[3.],
                 [1.]]]], dtype=float32)
    """

    def __init__(self, clockwise=True):
        super().__init__()
        self.clockwise = clockwise

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, got {}".format(type(inp))
        return rot90(inp, self.clockwise)


class Rotate(Module):
    r"""Rotate a by given angle.

    Args:
        angle(float): rotation angle of the image.
        format(str, optional): format of the input tensor, currently only supports NCHW and NHWC.
        interp_mode(str, optional): interpoloation mode, currently only supports bilinear for NCHW format
            and area mode for NHWC format.
    """

    def __init__(self, angle=0.0, format="NCHW", interp_mode="bilinear"):
        super().__init__()
        self.angle = angle
        self.format = format
        self.interp_mode = interp_mode

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, got {}".format(type(inp))
        return rotate(inp, self.angle, self.format, self.interp_mode)


class Remap(Module):
    r"""
    Applies remap transformation to batched 2D images. Remap is an operation that relocates pixels in a image to another location in a new image.

    Note:
        Input format must be nchw.

    The input images are transformed to the output images by the tensor ``map_1`` and ``map_2``.
    The output's H and W are same as ``map_1`` and ``map_2``'s H and W.

    Args:
    inp: input image, its shape represents ``[b, c, in_h, in_w]``.
    map_1: transformation matrix, the first map of ethier (x,y) points or just x values.
        if map_2.size == 0(the map of (x,y)), its shape shoule be ``[b, o_h, o_w, 2]``. The shape of output is determined by o_h and o_w.
        For each element in output, its value is determined by inp and ``map_xy``.
        ``map_1[..., 0]`` and ``map_1[..., 1]`` are the positions of
        the current element in inp, respectively. Therefore, their ranges are ``[0, in_w - 1]`` and ``[0, in_h - 1]``.
        if map_2.size == map_2.size(the map of(x)), its shape should be ``[b, o_h, o_w]``. it's shape are [``o, in_w-1]``.

    map_2:
        the second map of y values.
        if map2.size == map_2.size(the map of (y)), its shape should be ``[b, o_h, o_w]``. it's range are [``o, in_h-1]``.
    interpolation: interpolation methods. Default: "linear". Currently also support "nearest" mode.
    borderMode: pixel extrapolation method. Default: "replicate". Currently also support "constant", "reflect", "reflect_101", "wrap".
        "replicate": repeatedly fills the edge pixel values of the duplicate image, expanding the new boundary pixel values with
        the edge pixel values.
        "constant": fills the edges of the image with a fixed numeric value.
    borderValue: value used in case of a constant border. Default: 0.0

    Returns:
        output tensor. [b, c, o_h, o_w]

    Examples:
        >>> from megengine.module import Remap
        >>> from megengine.tensor import Tensor
        >>> import numpy
        >>> inp_shape = (1, 1, 4, 4)
        >>> inp = Tensor(numpy.arange(16, dtype=numpy.float32)).reshape(inp_shape)
        >>> x_map = Tensor([[[1,0],[0,0]]],dtype=numpy.float32)
        >>> y_map = Tensor([[[1,1],[1,1]]],dtype=numpy.float32)
        >>> remap = Remap()
        >>> out = remap(inp, x_map, y_map)
        >>> out.numpy()
        array([[[[5., 4.],
                 [4., 4.]]]], dtype=float32)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        inp: Tensor,
        map_1: Tensor,
        map_2: Tensor,
        interpolation: str = "linear",
        borderMode: str = "replicate",
        borderValue: float = 0.0,
    ):
        assert all(
            isinstance(ele, Tensor) for ele in [inp, map_1, map_2]
        ), "expected input is megengine.Tensor"
        assert map_1.size > 0, "expect map_1.size > 0"
        assert map_2.size == 0 or (
            list(map_1.shape) == list(map_2.shape)
        ), "expected map_1.size == 0 or map_1.shape == map_2.shape"
        map_xy = map_1
        if map_2.size > 0:
            ndim = map_1.ndim
            map_xy = stack([map_1, map_2], axis=ndim)
        out = remap(inp, map_xy, borderMode, borderValue, interpolation)
        return out


class GaussianBlur(Module):
    r"""Blurs an image using a Gaussian filter.

    Note:
        Input format must be nhwc.

    Args:
        kernel_size(Union[int, Tuple[int, int]]): Gaussian kernel size consisting of height and weight. height and width can differ but they both must be positive and odd.
        sigma_x(int): Gaussian kernel standard deviation in X direction.
        sigma_y(int): Gaussian kernel standard deviation in Y direction.
        border_mode(str): pixel extrapolation mode.
    
    Examples:
        >>> import numpy
        >>> import torch
        >>> from torchvision import transforms
        >>> data = numpy.arange(75).reshape((1,3,5,5))
        >>> data = torch.Tensor(data)
        >>> gaussian_blur = transforms.GaussianBlur((3,3,),1)
        >>> dst = gaussian_blur(data)
        >>> dst[0][0].numpy()
        array([[ 3.2888236,  3.7406862,  4.7406864,  5.7406864,  6.1925488],
            [ 5.548137 ,  6.       ,  7.       ,  8.       ,  8.451862 ],
            [10.548138 , 11.       , 12.       , 13.       , 13.451862 ],
            [15.548139 , 16.000002 , 17.       , 17.999998 , 18.451864 ],
            [17.807451 , 18.259315 , 19.259314 , 20.259314 , 20.711178 ]],
            dtype=float32)

    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        sigma_x=0,
        sigma_y=0,
        border_mode="REPLICATE",
    ):
        super().__init__()
        assert border_mode in [
            "REPLICATE",
            "REFLECT",
            "REFLECT_101",
            "WRAP",
            "CONSTANT",
            "TRANSPARENT",
            "ISOLATED",
        ]
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            assert len(kernel_size) == 2
            self.kernel_size = tuple(kernel_size)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.border_mode = border_mode

    def forward(self, inp):
        assert isinstance(
            inp, Tensor
        ), "expected input is megengine.Tensor, got {}".format(type(inp))
        kernel_height, kernel_width = self.kernel_size
        op = builtin.GaussianBlur(
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            sigma_x=self.sigma_x,
            sigma_y=self.sigma_y,
            border_mode=self.border_mode,
        )
        out = apply(op, inp)[0]
        return out
