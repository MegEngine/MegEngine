# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import copy
from typing import Any, Dict, List

from ..expr import Expr, is_constant, is_getattr
from ..node import Node, TensorNode


def register_obj(objs: List[Any], _dict: Dict):
    if not isinstance(objs, List):
        objs = [objs]

    def _register(any_obj: Any):
        for obj in objs:
            _dict[obj] = any_obj
        return any_obj

    return _register


def get_const_value(expr: Expr, fall_back: Any = None):
    value = fall_back
    if isinstance(expr, Node):
        expr = expr.expr
    if is_getattr(expr) and isinstance(expr.outputs[0], TensorNode):
        module = expr.inputs[0].owner
        assert module is not None
        value = copy.deepcopy(expr.interpret(module)[0])
    elif is_constant(expr):
        value = copy.deepcopy(expr.interpret()[0])
    return value
