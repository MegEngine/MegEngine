# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from ..._imperative_rt.ops import _custom

__all__ = []

for k, v in _custom.__dict__.items():
    globals()[k] = v
    __all__.append(k)


def gen_custom_op_maker(custom_op_name):
    def op_maker(**kwargs):
        return make_custom_op(custom_op_name, kwargs)
    return op_maker


def load(lib_path):
    op_in_this_lib = install(lib_path[0:-3], lib_path)
    for op in op_in_this_lib:
        op_maker = gen_custom_op_maker(op)
        globals()[op] = op_maker
        __all__.append(op)
