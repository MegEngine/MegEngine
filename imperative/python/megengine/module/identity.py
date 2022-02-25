# -*- coding: utf-8 -*-
from ..functional.tensor import copy
from .module import Module


class Identity(Module):
    r"""A placeholder identity operator that will ignore any argument."""

    def forward(self, x):
        return copy(x)
