# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import warnings

from ..._imperative_rt.ops import OprAttr
from . import param_defs


def make_param(param, ptype, kwargs):
    if param is not None:
        if isinstance(param, ptype):
            return param

        param = [param]
        assert len(param) == len(
            ptype.__slots__
        ), "{} needs {} params, but {} are provided".format(
            ptype, len(ptype.__slots__), len(param)
        )
        return ptype(*param)

    ckw = {}
    for i in ptype.__slots__:
        val = kwargs.pop(i, ckw)
        if val is not ckw:
            ckw[i] = val
    return ptype(**ckw)


class PodOpVisitor:
    __name2subclass = {}
    __c = None

    name = None
    param_names = []
    config = None

    def __init__(self, config, **params):
        self.config = config
        assert set(params) == set(self.param_names)
        self.__dict__.update(params)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # python 3.5 does not have this
        name = cls.name
        if name in cls.__name2subclass:
            if not issubclass(cls, cls.__name2subclass[name]):
                warnings.warn("Multiple subclasses for bultin op: %s" % name)
        cls.__name2subclass[name] = cls

    def to_c(self):
        if self.__c:
            return self.__c
        op = OprAttr()
        op.type = self.name
        if self.config is not None:
            op.config = self.config
        # first 4 bytes is TAG, has to remove them currently
        op.param = b"".join(self.__dict__[k].serialize()[4:] for k in self.param_names)
        self.__c = op
        return op

    def __eq__(self, rhs):
        return self.to_c() == rhs.to_c()

    def __repr__(self):
        name = self.__class__.__name__

        if self.__c:
            return "{}(<binary data>)".format(name)

        kwargs = {}
        for i in self.param_names:
            p = self.__dict__[i]
            if isinstance(p, param_defs._ParamDefBase):
                for k in p.__slots__:
                    v = getattr(p, k)
                    if isinstance(v, param_defs._EnumBase):
                        v = v.name
                    kwargs[k] = repr(v)
            else:
                kwargs[i] = repr(p)
        if self.config:
            if len(self.config.comp_node_arr) == 1:
                kwargs["device"] = "'%s'" % self.config.comp_node
        return "{}({})".format(
            name, ", ".join("{}={}".format(k, v) for k, v in kwargs.items())
        )
