from typing import Iterable, Optional, Union
from .distribution import Distribution
from ..tensor import Tensor
from .. import functional as F
from ..random import uniform
from numbers import Number

class Exponential(Distribution):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.
    Examples:
        >>> import megengine as mge
        >>> from megengine.distributions import Exponential
        >>> m = Exponential(mge.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        Tensor([2.4835])
    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    @property
    def mean(self) -> Tensor:
        return 1. / self.rate

    @property
    def stddev(self) -> Tensor:
        return 1. / self.rate

    @property
    def variance(self) -> Tensor:
        return F.pow(self.rate, -2)

    def __init__(self, rate: Union[Tensor, float]):
        self.rate = Tensor(rate)
        batch_shape = () if isinstance(rate, Number) else rate.shape
        super().__init__(batch_shape=batch_shape)

    def rsample(self, sample_shape: Optional[Iterable[int]] = ()) -> Tensor:
        def exponential_(rate):
            uniform_rand = uniform(size=rate.shape)
            epsilon = 1e-8
            log_rand = F.log(uniform_rand) if uniform_rand.numpy().any() >= 1 - epsilon \
                else -epsilon
            return -1 / rate * log_rand
        shape = self._extended_shape(sample_shape)
        rate_broadcasted = self.rate._broadcast(shape)
        return exponential_(rate_broadcasted)

    def log_prob(self, value):
        return F.log(self.rate) - self.rate * value

    def cdf(self, value):
        return 1. - F.exp(-self.rate * value)

    def icdf(self, value):
        return -F.log1p(-value) / self.rate

    @property
    def _natural_params(self):
        return (self.rate,)

    def _log_normalizer(self, x):
        return -F.log(-x)