from .. import Tensor
from typing import Iterable, Optional

__all__ = ["Distribution"]

class Distribution:
    r"""
    Distribution is the abstract base class for probability distributions.
    """

    def __init__(
        self,
        batch_shape : Optional[Iterable[int]] = (),
        event_shape : Optional[Iterable[int]] = ()
    ):
        self._batch_shape = batch_shape
        self._event_shape = event_shape
    
    def expand(self, batch_shape: Optional[Iterable[int]], _instance=None):
        """
        Returns a new distribution instance (or populates an existing instance
        provided by a derived class) with batch dimensions expanded to
        `batch_shape`. This method calls :class:`~torch.Tensor.expand` on
        the distribution's parameters. As such, this does not allocate new
        memory for the expanded distribution instance. Additionally,
        this does not repeat any args checking or parameter broadcasting in
        `__init__.py`, when an instance is first created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance: new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            New distribution instance with batch dimensions expanded to
            `batch_size`.
        """
        raise NotImplementedError

    @property
    def batch_shape(self):
        """
        Returns the shape over which parameters are batched.
        """
        return self._batch_shape
    
    @property
    def event_shape(self):
        """
        Returns the shape of a single sample (without batching).
        """
        return self._event_shape

    @property
    def mean(self) -> Tensor:
        """
        Returns the mean of the distribution.
        """
        raise NotImplementedError

    @property
    def variance(self) -> Tensor:
        """
        Returns the variance of the distribution.
        """
        raise NotImplementedError

    @property
    def stddev(self) -> Tensor:
        """
        Returns the standard deviation of the distribution.
        """
        return self.variance.sqrt()

    def sample(self, sample_shape: Optional[Iterable[int]] = ()) -> Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        return self.rsample(sample_shape)

    def rsample(self, sample_shape: Optional[Iterable[int]] =  ()) -> Tensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.
        """
        raise NotImplementedError
    
    def log_prob(self, value: Tensor) -> Tensor:
        """
        Returns the log of the probability density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def cdf(self, value: Tensor) -> Tensor:
        """
        Returns the cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError

    def icdf(self, value: Tensor) -> Tensor:
        """
        Returns the inverse cumulative density/mass function evaluated at
        `value`.

        Args:
            value (Tensor):
        """
        raise NotImplementedError
    
    def _extended_shape(self, sample_shape : Optional[Iterable[int]] = None) -> Optional[Iterable[int]]:
        """
        Returns the size of the sample returned by the distribution, given
        a `sample_shape`. Note, that the batch and event shapes of a distribution
        instance are fixed at the time of construction. If this is empty, the
        returned shape is upcast to (1,).

        Args:
            sample_shape : the size of the sample to be drawn.
        """
        return sample_shape + self._batch_shape + self._event_shape
