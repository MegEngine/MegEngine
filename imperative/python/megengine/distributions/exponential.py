from typing import Iterable, Optional, Union
from .distribution import Distribution
from ..tensor import Tensor
from .. import functional as F
from ..random import uniform
from numbers import Number

class Exponential(Distribution):
    r"""
    Creates a Exponential distribution parameterized by :attr:`rate`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = Exponential(torch.tensor([1.0]))
        >>> m.sample()  # Exponential distributed with rate=1
        tensor([ 0.1046])

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
            return -1 / rate * F.log1p(-uniform(size=rate.shape))
        shape = self._extended_shape(sample_shape)
        rate_broadcasted = self.rate._broadcast(shape)
        return exponential_(rate_broadcasted)


    def log_prob(self, value):
        return F.log(self.rate) - self.rate * value

    def cdf(self, value):
        return 1. - F.exp(-self.rate * value)
    
    def icdf(self, value):
        return -F.log(1. - value) / self.rate
    
    @property
    def _natural_params(self):
        return (self.rate,)
    
    def _log_normalizer(self, x):
        return -F.log(-x)
