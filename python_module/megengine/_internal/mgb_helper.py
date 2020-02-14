# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""helper utils for the core mgb module"""

import collections
import inspect
import json
import threading
from abc import ABCMeta, abstractmethod


class callback_lazycopy:
    """wraps around a callable to be passed to :meth:`.CompGraph.compile`.

    This is used to disable eager copy, so we could get rid of an h2d copy and
    a d2h if values are to be passed from one callback to another
    :class:`.SharedND`.
    """

    def __init__(self, func):
        assert isinstance(func, collections.Callable)
        self.__func = func

    @property
    def func(self):
        return self.__func


class SharedNDLazyInitializer(metaclass=ABCMeta):
    """lazy initialization policy for :class:`.SharedND`"""

    @abstractmethod
    def get_shape(self):
        """get shape, without loading value"""

    @abstractmethod
    def get_value(self):
        """get value as numpy ndarray"""


class copy_output:
    """wraps a :class:`.SymbolVar` in outspec for :meth:`.CompGraph.compile`,
    to copy the output to function return value"""

    symvar = None
    borrow_mem = None

    def __init__(self, symvar, *, borrow_mem=False):
        """

        :param borrow_mem: see :meth:`.CompGraphCallbackValueProxy.get_value`
        """
        from .mgb import SymbolVar

        assert isinstance(
            symvar, SymbolVar
        ), "copy_output expects an SymbolVar, got {} instead".format(symvar)
        self.symvar = symvar
        self.borrow_mem = borrow_mem


class FuncOutputSaver:
    """instance could be used as callbacks for :meth:`.CompGraph.compile` to
    copy output to host buffer
    """

    _value = None
    _borrow_mem = None

    def __init__(self, borrow_mem=False):
        self._borrow_mem = borrow_mem

    def __call__(self, v):
        self._value = v.get_value(borrow_mem=self._borrow_mem)

    def get(self):
        assert (
            self._value is not None
        ), "{} not called; maybe due to unwaited async func".format(self)
        return self._value
