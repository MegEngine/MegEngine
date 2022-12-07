# -*- coding: utf-8 -*-
import collections.abc
import math
from abc import ABC, abstractmethod
from itertools import count
from typing import Any, Generator, Iterator, List, Union

import numpy as np

from .. import distributed as dist


class Sampler(ABC):
    r"""An abstract base class for all Sampler"""

    @abstractmethod
    def __init__(self):
        pass


class MapSampler(Sampler):
    r"""Sampler for map dataset.

    Args:
        dataset: dataset to sample from.
        batch_size: batch size for batch method.
        drop_last: set ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch_size, then the last batch will
            be smaller. Default: False
        num_samples: number of samples assigned to one rank.
        world_size: number of ranks.
        rank: rank id, non-negative interger within 0 and ``world_size``.
        seed: seed for random operators.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        drop_last=False,
        num_samples=None,
        world_size=None,
        rank=None,
        seed=None,
    ):
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )

        if num_samples is not None and (
            not isinstance(num_samples, int) or num_samples <= 0
        ):
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(num_samples)
            )

        self.batch_size = batch_size
        self.dataset = dataset
        self.drop_last = drop_last

        if world_size is None:
            world_size = dist.get_world_size() if dist.is_distributed() else 1
        self.world_size = world_size
        if rank is None:
            rank = dist.get_rank() if dist.is_distributed() else 0
        self.rank = rank

        if num_samples is None:
            num_samples = len(self.dataset)
        self.num_samples = int(math.ceil(num_samples / self.world_size))

        if self.num_samples < self.batch_size:
            raise ValueError(
                "num_samples should be greater than batch_size "
                ", but got num_samples={} and batch_size={}".format(
                    self.num_samples, self.batch_size
                )
            )

        # Make sure seeds are the same at each rank
        if seed is None and self.world_size > 1:
            seed = 0
        self.rng = np.random.RandomState(seed)

    def __iter__(self) -> Union[Generator, Iterator]:
        return self.batch()

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return int(math.ceil(self.num_samples / self.batch_size))

    def sample(self):
        r"""Return a list contains all sample indices."""
        raise NotImplementedError

    def scatter(self, indices) -> List:
        r"""Scatter method is used for splitting indices into subset, each subset
        will be assigned to a rank. Indices are evenly splitted by default.
        If customized indices assignment method is needed, please rewrite this method.
        """
        total_size = self.num_samples * self.world_size

        # add extra indices to make it evenly divisible
        indices += indices[: (total_size - len(indices))]
        assert len(indices) == total_size

        # subsample
        indices = indices[self.rank : total_size : self.world_size]
        assert len(indices) == self.num_samples

        return indices

    def batch(self) -> Iterator[List[Any]]:
        r"""Batch method provides a batch indices generator."""
        indices = list(self.sample())

        # user might pass the world_size parameter without dist,
        # so dist.is_distributed() should not be used
        if self.world_size > 1:
            indices = self.scatter(indices)

        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch


class StreamSampler(Sampler):
    r"""Sampler for stream dataset.

    Warning:
        In the case of multiple machines, sampler should ensure that each worker gets
        different data. But this class cannot do it yet, please build your own
        dataset and sampler to achieve this goal.

    Usually, :meth:`~.StreamDataset.__iter__` can return different iterator by
    ``rank = dist.get_rank()``. So that they will get different data.
    """

    def __init__(self, batch_size=1):
        self.batch_size = batch_size

    def __iter__(self):
        return self.batch()

    def batch(self):
        batch = []
        for idx in self.sample():
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def sample(self):
        return count(start=0)


class SequentialSampler(MapSampler):
    r"""Sample elements sequentially.

    Args:
        dataset: dataset to sample from.
        batch_size: batch size for batch method.
        drop_last: set ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch_size, then the last batch will
            be smaller. Default: False
        indices: indice of samples.
        world_size: number of ranks.
        rank: rank id, non-negative interger within 0 and ``world_size``.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        drop_last=False,
        indices=None,
        world_size=None,
        rank=None,
    ):
        super().__init__(dataset, batch_size, drop_last, None, world_size, rank)
        if indices is not None and not isinstance(indices, collections.abc.Sequence):
            raise ValueError(
                "indices should be None or a sequence, "
                "but got indices={}".format(indices)
            )
        self.indices = indices

    def sample(self) -> Iterator[Any]:
        r"""Return a generator."""
        if self.indices is None:
            return iter(range(len(self.dataset)))
        else:
            return self.indices


class RandomSampler(MapSampler):
    r"""Sample elements randomly without replacement.

    Args:
        dataset: dataset to sample from.
        batch_size: batch size for batch method.
        drop_last: set ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch_size, then the last batch will
            be smaller. Default: False
        indices: indice of samples.
        world_size: number of ranks.
        rank: rank id, non-negative interger within 0 and ``world_size``.
        seed: seed for random operators.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        drop_last=False,
        indices=None,
        world_size=None,
        rank=None,
        seed=None,
    ):
        super().__init__(dataset, batch_size, drop_last, None, world_size, rank, seed)
        if indices is not None and not isinstance(indices, collections.abc.Sequence):
            raise ValueError(
                "indices should be None or a sequence, "
                "but got indices={}".format(indices)
            )
        self.indices = indices

    def sample(self) -> List:
        if self.indices is None:
            return self.rng.permutation(len(self.dataset)).tolist()
        else:
            return self.rng.permutation(self.indices).tolist()


class ReplacementSampler(MapSampler):
    r"""Sample elements randomly with replacement.

    Args:
        dataset: dataset to sample from.
        batch_size: batch size for batch method.
        drop_last: set ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch_size, then the last batch will
            be smaller. Default: False
        num_samples: number of samples assigned to one rank.
        weights: weights for sampling indices, it could be unnormalized weights.
        world_size: number of ranks.
        rank: rank id, non-negative interger within 0 and ``world_size``.
        seed: seed for random operators.
    """

    def __init__(
        self,
        dataset,
        batch_size=1,
        drop_last=False,
        num_samples=None,
        weights=None,
        world_size=None,
        rank=None,
        seed=None,
    ):
        super().__init__(
            dataset, batch_size, drop_last, num_samples, world_size, rank, seed
        )
        if weights is not None:
            if not isinstance(weights, collections.abc.Sequence):
                raise ValueError(
                    "weights should be None or a sequence, "
                    "but got weights={}".format(weights)
                )
            if len(weights) != len(dataset):
                raise ValueError(
                    "len(dataset)={} should be equal to"
                    "len(weights)={}".format(len(dataset), len(weights))
                )
        self.weights = weights
        if self.weights is not None:
            self.weights = np.array(weights) / sum(weights)

    def sample(self) -> List:
        n = len(self.dataset)
        sample_size = self.num_samples * self.world_size
        indices = self.rng.choice(n, size=sample_size, replace=True, p=self.weights)
        return indices.tolist()


class Infinite(Sampler):
    r"""Infinite Sampler warper for basic sampler."""

    def sample(self):
        raise NotImplementedError("sample method not supported in Infinite")

    def __init__(self, sampler):
        self.sampler = sampler
        self.sampler_iter = iter(self.sampler)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            index = next(self.sampler_iter)
        except StopIteration:
            self.sampler_iter = iter(self.sampler)
            index = next(self.sampler_iter)
        return index

    def __len__(self):
        return np.iinfo(np.int64).max

    def __getattr__(self, name):
        # if attribute could not be found in Infinite,
        # try to find it in self.sampler
        if name not in self.__dict__:
            return getattr(self.sampler, name)
        return self.__dict__[name]
