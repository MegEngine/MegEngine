import numpy as np

from megengine.functional.tensor import zeros

from ..core.ops.builtin import BatchNorm
from .expr import CallMethod, Constant
from .node import TensorNode
from .serialization import (
    register_functional_loader,
    register_module_loader,
    register_opdef_loader,
    register_tensor_method_loader,
)
from .utils import _convert_kwargs_to_args


"""
# Expr loaders examples

from ..core.ops.builtin import Elemwise

@register_opdef_loader(Elemwise)
def add_opdef_loader(expr):
    if expr.opdef_state["mode"] == "ADD":
        expr.opdef_state["mode"] == "MUL"
        node = expr.inputs[1]
        astype_expr = CallMethod(node, "astype")
        oup = TensorNode(
            astype_expr,
            shape=node.shape,
            dtype=expr.inputs[0].dtype,
            qparams=node.qparams,
        )

        astype_expr.set_args_kwargs(node, expr.inputs[0].dtype)
        astype_expr.return_val = (oup,)
        expr.inputs[1] = oup


@register_functional_loader(("megengine.functional.nn", "conv2d"))
def conv2df_loader(expr):
    # expr.func = ("megengine.functional.nn","conv2d")
    kwargs = expr.kwargs
    orig_weight = expr.named_args["weight"]

    astype_expr = CallMethod(orig_weight, "astype")
    oup = TensorNode(
        astype_expr,
        shape=orig_weight.shape,
        dtype=orig_weight.dtype,
        qparams=orig_weight.qparams,
    )

    astype_expr.set_args_kwargs(orig_weight, expr.named_args["inp"].dtype)
    astype_expr.return_val = (oup,)

    expr.set_arg("weight", oup)


@register_module_loader(("megengine.module.conv", "Conv2d"))
def conv2dm_loader(expr):
    module = expr.inputs[0].owner
    args = list(expr.args)
    orig_inp = args[1]
    astype_expr = CallMethod(orig_inp, "astype")
    oup = TensorNode(
        astype_expr,
        shape=orig_inp.shape,
        dtype=orig_inp.dtype,
        qparams=orig_inp.qparams,
    )
    astype_expr.set_args_kwargs(orig_inp, module.weight.dtype)
    astype_expr.return_val = (oup,)
    args[1] = oup
    expr.set_args_kwargs(*args)


@register_tensor_method_loader("__add__")
def add_loader(expr):
    args = list(expr.args)
    if not isinstance(args[1], TensorNode):
        args[1] = tensor(args[1])
        node = Constant(args[1], "const").outputs[0]

        astype_expr = CallMethod(node, "astype")
        oup = TensorNode(
            astype_expr, shape=node.shape, dtype=node.dtype, qparams=node.qparams,
        )

        astype_expr.set_args_kwargs(node, expr.inputs[0].dtype)
        astype_expr.return_val = (oup,)
        args[1] = oup
        expr.set_args_kwargs(*args)
"""


@register_module_loader(
    ("megengine.module.batchnorm", "BatchNorm1d"),
    ("megengine.module.batchnorm", "BatchNorm2d"),
    ("megengine.module.batchnorm", "SyncBatchNorm"),
)
def bn2d_module_loader(expr):
    module = expr.inputs[0].owner
    if hasattr(module, "param_dim"):
        assert module.param_dim == "dim_1c11"
        delattr(module, "param_dim")


@register_module_loader(
    ("megengine.module.conv_bn", "ConvBn2d"),
    ("megengine.module.conv_bn", "ConvBnRelu2d"),
    ("megengine.module.qat.conv_bn", "ConvBn2d"),
    ("megengine.module.qat.conv_bn", "ConvBnRelu2d"),
)
def convbn2d_module_loader(expr):
    module = expr.inputs[0].owner
    if hasattr(module.bn, "param_dim"):
        assert module.bn.param_dim == "dim_1c11"
        delattr(module.bn, "param_dim")
    if not hasattr(module.conv, "padding_mode"):
        module.conv.padding_mode = "zeros"


@register_opdef_loader(BatchNorm)
def bn_opdef_loader(expr):
    # mge 1.6
    if not hasattr(expr, "version") and len(expr.outputs) != 6:
        assert len(expr.outputs) == 5
        output = expr.outputs[-1]
        oup = TensorNode(expr, shape=(0,), dtype=None, qparams=output._qparams,)
        expr.outputs.insert(4, oup)


@register_functional_loader(
    ("megengine.functional.tensor", "ones"), ("megengine.functional.tensor", "zeros")
)
def tensor_gen_func_loader(expr):
    if hasattr(expr, "version") and expr.version == "1.7.0":
        expr.set_args_kwargs(expr.args[0], dtype=expr.args[1], device=expr.args[2])
    if not hasattr(expr, "version"):
        # compatiable for version 1.6
        shape = expr.args[0] if len(expr.args) > 0 else expr.kwargs["shape"]

        if len(expr.args) > 1:
            dtype = expr.args[1]
        elif "dtype" in expr.kwargs:
            dtype = expr.kwargs["dtype"]
        else:
            dtype = "float32"

        if len(expr.args) > 2:
            device = expr.args[2]
        elif "device" in expr.kwargs:
            device = expr.kwargs["device"]
        else:
            device = None
        expr.set_args_kwargs(shape, dtype=dtype, device=device)


@register_functional_loader(("megengine.functional.nn", "pad"))
def pad_func_loader(expr):
    if "pad_witdth" in expr.kwargs:
        kwargs = expr.kwargs
        kwargs["pad_width"] = kwargs.pop("pad_witdth")
        expr.set_args_kwargs(*expr.args, **kwargs)


@register_functional_loader(("megengine.functional.nn", "batch_norm"))
def bn_func_loader(expr):
    kwargs = expr.kwargs
    if "compute_mode" in kwargs:
        assert kwargs["compute_mode"] == "default"
        kwargs.pop("compute_mode")
    if "param_dim" in kwargs:
        assert kwargs["param_dim"] == "dim_1c11"
        kwargs.pop("param_dim")
    expr.set_args_kwargs(*expr.args, **kwargs)


@register_functional_loader(("megengine.functional.math", "matmul"))
def matmul_func_loader(expr):
    args = expr.args
    if len(args) == 6:
        assert args[5] == "default"
        expr.set_args_kwargs(*args[0:5])


@register_module_loader(
    ("megengine.module.conv", "Conv1d"),
    ("megengine.module.conv", "Conv2d"),
    ("megengine.module.conv", "ConvRelu2d"),
    ("megengine.module.qat.conv", "Conv2d"),
    ("megengine.module.qat.conv", "ConvRelu2d"),
    ("megengine.module.quantized.conv", "Conv2d"),
    ("megengine.module.quantized.conv", "ConvRelu2d"),
)
def conv2d_module_loader(expr):
    module = expr.inputs[0].owner
    if not hasattr(module, "padding_mode"):
        module.padding_mode = "zeros"


@register_module_loader(
    ("megengine.module.quantized.conv_bn", "ConvBn2d"),
    ("megengine.module.quantized.conv_bn", "ConvBnRelu2d"),
)
def quantized_convbn2d_module_loader(expr):
    module = expr.inputs[0].owner
    if not hasattr(module, "padding_mode"):
        module.padding_mode = "zeros"


@register_functional_loader(("megengine.functional.elemwise", "square"))
def square_func_loader(expr):
    import pkg_resources as pkg

    if not hasattr(expr, "version") or pkg.parse_version(
        expr.version
    ) <= pkg.parse_version("1.11.1"):
        if expr.inputs[0].dtype != np.float32:
            orig_oup = expr.outputs[0]
            oup = TensorNode(expr, shape=orig_oup.shape, dtype=expr.inputs[0].dtype,)
            expr.return_val = (oup,)
            astype_expr = CallMethod(oup, "astype")
            astype_expr.set_args_kwargs(oup, "float32")
            orig_oup.expr = astype_expr
            astype_expr.return_val = (orig_oup,)


@register_functional_loader(("megengine.functional.math", "topk"))
def topk_loader(expr):
    import pkg_resources as pkg

    if not hasattr(expr, "version") or pkg.parse_version(
        expr.version
    ) <= pkg.parse_version("1.12.0"):

        def origin_topk_signature(
            inp, k, descending=False, kth_only=False, no_sort=False
        ):
            pass

        args, kwargs = _convert_kwargs_to_args(
            origin_topk_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)


@register_functional_loader(("megengine.functional.tensor", "arange"))
def arange_func_loader(expr):
    kwargs = expr.kwargs
    args = expr.args
    if not hasattr(expr, "version"):

        def origin_arange_signature(
            start=0, stop=None, step=1, dtype="float32", device=None
        ):
            pass

        args, kwargs = _convert_kwargs_to_args(
            origin_arange_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)
    if len(args) == 5:
        device = args[-1]
        dtype = args[-2]
        args = args[:-2]

        kwargs["dtype"] = dtype
        kwargs["device"] = device

    expr.set_args_kwargs(*args, **kwargs)


@register_functional_loader(("megengine.functional.tensor", "linspace"))
def linespace_loader(expr):
    args, kwargs = expr.args, expr.kwargs
    if not hasattr(expr, "version"):

        def orig_linspace_signature(start, stop, num, dtype="float32", device=None):
            pass

        args, kwargs = _convert_kwargs_to_args(
            orig_linspace_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)
    if len(args) == 5:
        new_args = args[0:-2]
        new_kwargs = {"dtype": args[-2], "device": args[-1]}
        expr.set_args_kwargs(*new_args, **new_kwargs)


@register_functional_loader(("megengine.functional.tensor", "full"))
def full_func_loader(expr):
    kwargs = expr.kwargs
    args = expr.args
    if not hasattr(expr, "version"):

        def orig_full_signature(shape, value, dtype=None, device=None):
            pass

        args, kwargs = _convert_kwargs_to_args(
            orig_full_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)
    if len(args) == 4:
        device = args[-1]
        dtype = args[-2]
        args = args[: len(args) - 2]

        kwargs["dtype"] = dtype
        kwargs["device"] = device

    expr.set_args_kwargs(*args, **kwargs)


@register_functional_loader(("megengine.functional.nn", "conv_transpose2d"))
def deconv_loader(expr):
    args, kwargs = expr.args, expr.kwargs
    if not hasattr(expr, "version"):

        def orig_conv_transpose2d_signature(
            inp,
            weight,
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            conv_mode="cross_correlation",
            compute_mode="default",
        ):
            pass

        args, kwargs = _convert_kwargs_to_args(
            orig_conv_transpose2d_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)
    if len(args) == 9:
        args = list(args)
        args.insert(4, 0)  # output padding = 0
        expr.set_args_kwargs(*args, **kwargs)


@register_functional_loader(("megengine.functional.quantized", "conv_transpose2d"))
def deconv_loader(expr):
    args, kwargs = expr.args, expr.kwargs
    if not hasattr(expr, "version"):

        def orig_conv_transpose2d_signature(
            inp,
            weight,
            bias=None,
            dtype=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            conv_mode="cross_correlation",
            compute_mode="default",
        ):
            pass

        args, kwargs = _convert_kwargs_to_args(
            orig_conv_transpose2d_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)
    if len(args) == 10:
        args = list(args)
        args.insert(5, 0)  # output padding = 0
        expr.set_args_kwargs(*args, **kwargs)


@register_functional_loader(("megengine.functional.nn", "conv_transpose3d"))
def deconv3d_loader(expr):
    args, kwargs = expr.args, expr.kwargs
    if not hasattr(expr, "version"):

        def origin_conv_transpose3d_signature(
            inp, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
        ):
            pass

        args, kwargs = _convert_kwargs_to_args(
            origin_conv_transpose3d_signature, expr.args, expr.kwargs
        )
        expr.set_args_kwargs(*args, **kwargs)
    if len(args) == 7:
        args = list(args)
        args.insert(4, 0)
        expr.set_args_kwargs(*args, **kwargs)
