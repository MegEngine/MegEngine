# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from abc import ABC, abstractmethod
from typing import Sequence, Tuple


class Transform(ABC):
    """
    Rewrite apply method in subclass.
    """

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
