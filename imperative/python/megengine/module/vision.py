import numpy as np

from ..functional.elemwise import abs, add, log
from ..functional.math import sign
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
        per_channel: Whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
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
        per_channel: Whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
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
        mean: Gaussian mean used to generate noise.
        std: Gaussian standard deviation used to generate noise.
        per_channel: Whether to use (imagewise) the same sample(s) for all channels (False) or to sample value(s) for each channel (True). Setting this to True will therefore lead to different transformations per image and channel, otherwise only per image. 
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
