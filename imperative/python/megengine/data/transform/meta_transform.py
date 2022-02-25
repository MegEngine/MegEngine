# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class Transform(ABC):
    r"""Rewrite apply method in subclass."""

    def apply_batch(self, inputs: Sequence[Tuple]):
        return tuple(self.apply(input) for input in inputs)

    @abstractmethod
    def apply(self, input: Tuple):
        pass

    def __repr__(self):
        return self.__class__.__name__


class PseudoTransform(Transform):
    def apply(self, input: Tuple):
        return input
