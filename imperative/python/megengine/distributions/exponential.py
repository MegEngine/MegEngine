from numbers import Number
from typing import Iterable, Optional, Union

from .. import functional as F
from ..random import exponential
from ..tensor import Tensor
from .distribution import Distribution


class Exponential(Distribution):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    This is a EXPERIMENTAL module that may be subject to change and/or deletion.
    
    Args:
        rate (float or Tensor): rate = 1 / scale of the distribution
    """

    def __init__(self, rate: Union[Tensor, float]):
        self.rate = Tensor(rate)
        batch_shape = () if isinstance(rate, Number) else rate.shape
        super().__init__(batch_shape=batch_shape)

    @property
    def mean(self) -> Tensor:
        return 1.0 / self.rate

    @property
    def stddev(self) -> Tensor:
        return 1.0 / self.rate

    @property
    def variance(self) -> Tensor:
        return F.pow(self.rate, -2)

    def sample(self, sample_shape: Optional[Iterable[int]] = ()) -> Tensor:
        return exponential(self.rate, sample_shape)

    def log_prob(self, value):
        return F.log(self.rate) - self.rate * value

    def cdf(self, value):
        return 1.0 - F.exp(-self.rate * value)

    def icdf(self, value):
        return -F.log1p(-value) / self.rate

    @property
    def _natural_params(self):
        return (self.rate,)

    def _log_normalizer(self, x):
        return -F.log(-x)
