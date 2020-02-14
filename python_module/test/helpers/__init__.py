# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from typing import List, Tuple

import numpy as np

import megengine._internal as mgb
import megengine.functional as F
from megengine import Graph, jit
from megengine.module import Linear, Module
from megengine.test import assertTensorClose

from .env import modified_environ


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.dense0 = Linear(28, 50)
        self.dense1 = Linear(50, 20)

    def forward(self, x):
        x = self.dense0(x)
        x = F.relu(x)
        x = self.dense1(x)
        return x


def has_gpu(num=1):
    try:
        mgb.comp_node("gpu{}".format(num - 1))
    except mgb.MegBrainError:
        return False

    return True


def randomNp(*args):
    for arg in args:
        assert isinstance(arg, int)
    return np.random.random(args)


def randomTorch(*args):
    import torch  # pylint: disable=import-outside-toplevel

    for arg in args:
        assert isinstance(arg, int)
    return torch.tensor(randomNp(*args), dtype=torch.float32)


def graph_mode(*modes):
    if not set(modes).issubset({"eager", "static"}):
        raise ValueError("graph mode must be in (eager, static)")

    def decorator(func):
        def wrapper(*args, **kwargs):
            if "eager" in set(modes):
                func(*args, **kwargs)
            if "static" in set(modes):
                with Graph() as cg:
                    cg.set_option("eager_evaluation", False)
                    func(*args, **kwargs)

        return wrapper

    return decorator


def _default_compare_fn(x, y):
    assertTensorClose(x.numpy(), y)


def opr_test(
    cases,
    func,
    mode=("eager", "static", "dynamic_shape"),
    compare_fn=_default_compare_fn,
    ref_fn=None,
    **kwargs
):
    """
    mode: the list of test mode which are eager, static and dynamic_shape
          will test all the cases if None.
    func: the function to run opr.
    compare_fn: the function to compare the result and expected, use assertTensorClose if None.
    ref_fn: the function to generate expected data, should assign output if None.
    cases: the list which have dict element, the list length should be 2 for dynamic shape test.
           and the dict should have input,
           and should have output if ref_fn is None.
           should use list for multiple inputs and outputs for each case.
    kwargs: The additional kwargs for opr func.

    simple examples:

        dtype = np.float32
        cases = [{"input": [10, 20]}, {"input": [20, 30]}]
        opr_test(cases,
                 F.eye,
                 ref_fn=lambda n, m: np.eye(n, m).astype(dtype),
                 dtype=dtype)

    """

    def check_results(results, expected):
        if not isinstance(results, Tuple):
            results = (results,)
        for r, e in zip(results, expected):
            compare_fn(r, e)

    def get_trace_fn(func, enabled, symbolic):
        jit.trace.enabled = enabled
        return jit.trace(func, symbolic=symbolic)

    def get_param(cases, idx):
        case = cases[idx]
        inp = case.get("input", None)
        outp = case.get("output", None)
        if inp is None:
            raise ValueError("the test case should have input")
        if not isinstance(inp, List):
            inp = (inp,)
        else:
            inp = tuple(inp)
        if ref_fn is not None and callable(ref_fn):
            outp = ref_fn(*inp)
        if outp is None:
            raise ValueError("the test case should have output or reference function")
        if not isinstance(outp, List):
            outp = (outp,)
        else:
            outp = tuple(outp)

        return inp, outp

    if not set(mode).issubset({"eager", "static", "dynamic_shape"}):
        raise ValueError("opr test mode must be in (eager, static, dynamic_shape)")

    if len(cases) == 0:
        raise ValueError("should give one case at least")

    if "dynamic_shape" in set(mode):
        if len(cases) != 2:
            raise ValueError("should give 2 cases for dynamic shape test")

    if not callable(func):
        raise ValueError("the input func should be callable")

    inp, outp = get_param(cases, 0)

    def run(*args, **kwargs):
        return func(*args, **kwargs)

    if "eager" in set(mode):
        f = get_trace_fn(run, False, False)
        results = f(*inp, **kwargs)
        check_results(results, outp)

    if "static" in set(mode) or "dynamic_shape" in set(mode):
        f = get_trace_fn(run, True, True)
        results = f(*inp, **kwargs)
        check_results(results, outp)
        if "dynamic_shape" in set(mode):
            inp, outp = get_param(cases, 1)
            results = f(*inp, **kwargs)
            check_results(results, outp)
