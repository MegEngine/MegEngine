# -*- coding: utf-8 -*-
import collections
import itertools
from typing import Iterable, Union

import numpy as np

from .. import _config
from .._imperative_rt import make_const
from .._imperative_rt.core2 import (
    Const,
    Tensor,
    _get_convert_inputs,
    _set_convert_inputs,
    apply,
    astensor1d_cpp,
    astype_cpp,
    convert_inputs_cpp,
    convert_single_value_cpp,
    dtype_promotion,
    get_device,
    make_shape_tuple,
)
from .._imperative_rt.ops import SubgraphBuilder as _SubgraphBuilder
from .._imperative_rt.ops import jit_supported
from .._wrap import as_device
from ..autodiff.grad import Function
from ..ops import builtin
from .amp import _get_amp_high_prec_dtype, _get_amp_low_prec_dtype
from .dtype import is_dtype_equal, is_quantize

jit_supported = False


def get_convert_inputs():
    r"""get the curerent state of `_enable_convert_inputs`"""
    return _get_convert_inputs()


def set_convert_inputs(flag):
    r"""This function is a temporary workaround for reducing the overhead of operator
    invocations. The function `convert_inputs` is disabled if the global state
    `_enable_convert_inputs` is set to `False`, otherwise enabled. This function is for
    internal use only, and should be removed when the tensor-like system is refactored.
    """
    return _set_convert_inputs(flag)


def convert_single_value(v, *, dtype=None, device=None):
    return convert_single_value_cpp(v, dtype, device)


def convert_inputs(*args, device=None):
    if not _get_convert_inputs():
        return args
    return convert_inputs_cpp(*args, device)


def cast_tensors(*args, promote=False):
    if promote:
        dtype = _get_amp_high_prec_dtype()
    else:
        dtype = _get_amp_low_prec_dtype()
    return tuple(arg.astype(dtype) if arg is not None else None for arg in args)


def result_type(*args):
    dtypes = []
    for i in args:
        if isinstance(i, Tensor):
            dtypes.append(i.dtype)
            continue
        try:
            dtypes.append(np.dtype(i))
        except TypeError:
            pass
    return np.result_type(*dtypes)


def isscalar(x):

    if isinstance(x, Tensor):
        return x._isscalar()

    return np.isscalar(x)


def astensor1d(x, *reference, dtype=None, device=None):
    """Convert something to 1D tensor. Support following types

      * sequence of scalar literal / tensor
      * numpy array
      * tensor (returned as is, regardless of dtype and device)
    """
    return astensor1d_cpp(x, dtype, device, reference)


def _normalize_axis(
    ndim: int, axis: Union[int, Iterable], reverse=False
) -> Union[int, list]:
    def convert(x):
        x_org = x
        if x < 0:
            x = ndim + x
        assert (
            x >= 0 and x < ndim
        ), "axis {} is out of bounds for tensor of dimension {}".format(x_org, ndim)
        return x

    if isinstance(axis, int):
        return convert(axis)
    elif isinstance(axis, Iterable):
        axis_org = axis
        axis = list(sorted(map(convert, axis), reverse=reverse))
        for i in range(len(axis) - 1):
            assert axis[i] != axis[i + 1], "axis {} contains duplicated indices".format(
                axis_org
            )
        return axis
    raise


_opr_map = {
    ("-", 1): builtin.Elemwise(mode="negate"),
    ("abs", 1): builtin.Elemwise(mode="abs"),
    ("exp", 1): builtin.Elemwise(mode="exp"),
    ("log1p", 1): builtin.Elemwise(mode="log1p"),
    ("relu", 1): builtin.Elemwise(mode="relu"),
    ("cond_leq_mov", 3): builtin.Elemwise(mode="cond_leq_mov"),
    ("fma3", 3): builtin.Elemwise(mode="FUSE_MUL_ADD3"),
    ("fma4", 4): builtin.Elemwise(mode="FUSE_MUL_ADD4"),
    ("[?:]", 2): builtin.Subtensor(items=[(0, True, False, False, False)]),
    ("[:?]", 2): builtin.Subtensor(items=[(0, False, True, False, False)]),
}

for name, mode in [
    ("+", "add"),
    ("-", "sub"),
    ("*", "mul"),
    ("/", "true_div"),
    ("//", "floor_div"),
    ("**", "pow"),
    ("max", "max"),
    ("min", "min"),
    ("additive", "add"),
    ("exp", "EXP"),
    ("switch_gt0", "switch_gt0"),
    ("abs_grad", "abs_grad"),
]:
    _opr_map[(name, 2)] = builtin.Elemwise(mode=mode)


def subgraph(
    name, dtype, device, nr_inputs, gopt_level=None, jit_fusion=False, custom_grad=False
):
    if not device.physical_name.startswith("gpu"):
        jit_fusion = False

    if jit_fusion and not jit_supported:
        jit_fusion = False  # jit unusable, fallback to graph compile
        gopt_level = 2

    def as_op(op, nargs):
        if isinstance(op, str):
            assert (op, nargs) in _opr_map, "unknown operator"
            op = _opr_map[(op, nargs)]
        return op

    def decorator(func):
        builder = _SubgraphBuilder(name)

        def apply_expr(op, *args, nr_out=None):
            op = as_op(op, len(args))
            results = builder.apply(op, args, 1 if nr_out is None else nr_out)
            if nr_out is None:
                assert len(results) == 1
                return results[0]
            else:
                assert len(results) == nr_out
                return results

        def apply_const(value, dtype=dtype, device=device):
            return builder.apply_const(value, dtype, device)

        def build(builder, outputs, outputs_has_grad):
            builder = type(builder)(builder)
            builder.outputs(outputs)
            builder.outputs_has_grad(outputs_has_grad)
            if jit_fusion:
                assert gopt_level is None
                op = lambda: builder.jit_fuse()
            elif gopt_level is None:
                op = lambda: builder.get()
            else:
                op = lambda: builder.compile(gopt_level)
            return op

        inputs = [builder.input() for _ in range(nr_inputs)]
        if not custom_grad:
            outputs, outputs_has_grad = func(inputs, apply_expr, apply_const)
            return build(builder, outputs, outputs_has_grad)
        else:
            gen = func(inputs, apply_expr, apply_const)
            outputs = gen.send(None)
            nr_outputs = len(outputs)
            forward_fn = build(builder, outputs, [False] * nr_outputs)
            output_grads = [builder.input() for _ in range(nr_outputs)]
            input_grads = gen.send(output_grads)
            assert len(input_grads) == nr_inputs
            input_grads_mask = [input_grad is not None for input_grad in input_grads]
            indices = [
                i - 1 if mask else None
                for i, mask in zip(
                    itertools.accumulate(input_grads_mask), input_grads_mask
                )
            ]
            encoded_input_grads = [grad for grad in input_grads if grad is not None]
            backward_fn = build(
                builder, encoded_input_grads, [True] * len(encoded_input_grads)
            )

            class SubgraphOp(Function):
                def __init__(self):
                    self.inputs = None
                    self.output_shapes = None

                def forward(self, *inputs):
                    self.inputs = inputs
                    outputs = apply(forward_fn(), *inputs)
                    if len(outputs) > 1:
                        self.output_shapes = [output.shape for output in outputs]
                    return outputs

                def backward(self, *output_grads):
                    inputs = self.inputs
                    any_valid = False
                    all_valid = True
                    for output_grad in output_grads:
                        if output_grad is None:
                            all_valid = False
                        else:
                            any_valid = True
                    if not any_valid:
                        input_grads = [None] * len(indices)
                    else:
                        if not all_valid:
                            assert self.output_shapes is not None
                            from ...functional import zeros

                            output_grads = [
                                zeros(self.output_shapes[i]) if grad is None else grad
                                for i, grad in enumerate(output_grads)
                            ]
                        self = None
                        encoded_input_grads = apply(
                            backward_fn(), *inputs, *output_grads
                        )
                        input_grads = [
                            encoded_input_grads[i] if i is not None else None
                            for i in indices
                        ]
                    return input_grads

            gen.close()
            return SubgraphOp

    return decorator


def interpret_subgraph(func, dtype, device):
    def as_op(op, nargs):
        if isinstance(op, str) and (op, nargs) in _opr_map:
            op = _opr_map[(op, nargs)]
        return op

    def decorated_func(*args):
        def apply_expr(op, *args, nr_out=None):
            op = as_op(op, len(args))
            results = apply(op, *args)
            if nr_out is None:
                assert len(results) == 1
                return results[0]
            else:
                assert len(results) == nr_out
                return results

        def apply_const(value, dtype=dtype, device=device):
            return Const(value, dtype, device)

        outputs, outputs_has_grad = func(args, apply_expr, apply_const)
        outputs = [
            output if has_grad else output.detach()
            for output, has_grad in zip(outputs, outputs_has_grad)
        ]
        return outputs

    return decorated_func


def subgraph_fn(
    name,
    dtype,
    device,
    nr_inputs,
    gopt_level=None,
    jit_fusion=False,
    custom_grad=False,
    *,
    interpret=False
):
    def decorator(func):
        if not interpret:
            op = subgraph(
                name,
                dtype,
                device,
                nr_inputs,
                gopt_level=gopt_level,
                jit_fusion=jit_fusion,
                custom_grad=custom_grad,
            )(func)

            def wrapped_func(*args):
                if custom_grad:
                    outputs = op()(*args)
                else:
                    outputs = apply(op(), *args)
                return outputs

            return wrapped_func
        else:
            return interpret_subgraph(func, dtype, device)

    return decorator
