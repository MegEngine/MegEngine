import collections
import functools
from collections import namedtuple
from contextlib import contextmanager
from functools import partial
from typing import Iterable

import numpy as np
import tabulate

from .. import Tensor
from .. import functional as F
from .. import get_logger
from .. import module as M
from ..core.tensor.dtype import get_dtype_bit
from ..logger import MegEngineLogFormatter
from .module_utils import set_module_mode_safe

try:
    MegEngineLogFormatter.max_lines = float("inf")
except AttributeError as e:
    raise ValueError("set logger max lines failed")

logger = get_logger(__name__)
logger.setLevel("INFO")

_calc_flops_dict = {}
_calc_receptive_field_dict = {}


def _receptive_field_fallback(module, inputs, outputs):
    if not _receptive_field_enabled:
        return
    assert not hasattr(module, "_rf")
    assert not hasattr(module, "_stride")
    if len(inputs) == 0:
        # TODO: support other dimension
        module._rf = (1, 1)
        module._stride = (1, 1)
        return module._rf, module._stride
    rf, stride = preprocess_receptive_field(module, inputs, outputs)
    module._rf = rf
    module._stride = stride
    return rf, stride


# key tuple, impl_dict, fallback
_iter_list = [
    ("flops_num", _calc_flops_dict, None),
    (
        ("receptive_field", "stride"),
        _calc_receptive_field_dict,
        _receptive_field_fallback,
    ),
]

_receptive_field_enabled = False


def _register_dict(*modules, dict=None):
    def callback(impl):
        for module in modules:
            dict[module] = impl
        return impl

    return callback


def register_flops(*modules):
    return _register_dict(*modules, dict=_calc_flops_dict)


def register_receptive_field(*modules):
    return _register_dict(*modules, dict=_calc_receptive_field_dict)


def enable_receptive_field():
    global _receptive_field_enabled
    _receptive_field_enabled = True


def disable_receptive_field():
    global _receptive_field_enabled
    _receptive_field_enabled = False


@register_flops(M.Conv1d, M.Conv2d, M.Conv3d, M.LocalConv2d, M.DeformableConv2d)
def flops_convNd(module: M.Conv2d, inputs, outputs):
    bias = 1 if module.bias is not None else 0
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    return np.prod(outputs[0].shape) * (
        float(module.in_channels // module.groups) * np.prod(module.kernel_size) + bias
    )


@register_flops(M.ConvTranspose2d)
def flops_convNdTranspose(module: M.Conv2d, inputs, outputs):
    bias = 1 if module.bias is not None else 0
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    return (
        np.prod(inputs[0].shape)
        * (module.out_channels // module.groups * np.prod(module.kernel_size))
        + np.prod(outputs[0].shape) * bias
    )


@register_flops(
    M.batchnorm._BatchNorm, M.SyncBatchNorm, M.GroupNorm, M.LayerNorm, M.InstanceNorm,
)
def flops_norm(module: M.Linear, inputs, outputs):
    return np.prod(inputs[0].shape) * 7


@register_flops(M.AvgPool2d, M.MaxPool2d)
def flops_pool(module: M.AvgPool2d, inputs, outputs):
    kernel_sum = 0
    if isinstance(module.kernel_size, tuple) and len(module.kernel_size) == 2:
        kernel_sum = np.prod(module.kernel_size)
    else:
        kernel_sum = module.kernel_size ** 2
    return np.prod(outputs[0].shape) * kernel_sum


@register_flops(M.AdaptiveAvgPool2d, M.AdaptiveMaxPool2d)
def flops_adaptivePool(module: M.AdaptiveAvgPool2d, inputs, outputs):
    stride_h = np.floor(inputs[0].shape[2] / (inputs[0].shape[2] - 1))
    kernel_h = inputs[0].shape[2] - (inputs[0].shape[2] - 1) * stride_h
    stride_w = np.floor(inputs[0].shape[3] / (inputs[0].shape[3] - 1))
    kernel_w = inputs[0].shape[3] - (inputs[0].shape[3] - 1) * stride_w
    return np.prod(outputs[0].shape) * kernel_h * kernel_w


@register_flops(M.Linear)
def flops_linear(module: M.Linear, inputs, outputs):
    bias = module.out_features if module.bias is not None else 0
    return np.prod(outputs[0].shape) * module.in_features + bias


@register_flops(M.BatchMatMulActivation)
def flops_batchmatmul(module: M.BatchMatMulActivation, inputs, outputs):
    bias = 1 if module.bias is not None else 0
    x = inputs[0]
    w = module.weight
    batch_size = x.shape[0]
    n, p = x.shape[1:]
    _, m = w.shape[1:]
    return n * (p + bias) * m * batch_size


# does not need import qat and quantized module since they inherit from float module.
hook_modules = [
    M.conv._ConvNd,
    M.Linear,
    M.BatchMatMulActivation,
    M.batchnorm._BatchNorm,
    M.LayerNorm,
    M.GroupNorm,
    M.InstanceNorm,
    M.pooling._PoolNd,
    M.adaptive_pooling._AdaptivePoolNd,
]


def register_hook_module(module):
    if isinstance(module, (tuple, list)):
        modules = module
        for module in modules:
            register_hook_module(module)
    elif issubclass(module, M.Module):
        hook_modules.append(module)
    else:
        raise TypeError("the param type should in [list,tuple,M.Module]")


def _mean(inp):
    inp = Tensor(inp).astype(np.float32)
    return F.mean(inp).numpy()


def _std(inp):
    inp = Tensor(inp).astype(np.float32)
    return F.std(inp).numpy()


def dict2table(list_of_dict, header):
    table_data = [header]
    for d in list_of_dict:
        row = []
        for h in header:
            v = ""
            if h in d:
                v = d[h]
            row.append(v)
        table_data.append(row)
    return table_data


def sizeof_fmt(num, suffix="B"):
    if suffix == "B":
        scale = 1024.0
        units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
    else:
        scale = 1000.0
        units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y"]
    for unit in units:
        if abs(num) < scale or unit == units[-1]:
            return "{:3.3f} {}{}".format(num, unit, suffix)
        num /= scale


def preprocess_receptive_field(module, inputs, outputs):
    # TODO: support other dimensions
    pre_rf = (
        max(getattr(i.owner, "_rf", (1, 1))[0] for i in inputs),
        max(getattr(i.owner, "_rf", (1, 1))[1] for i in inputs),
    )
    pre_stride = (
        max(getattr(i.owner, "_stride", (1, 1))[0] for i in inputs),
        max(getattr(i.owner, "_stride", (1, 1))[1] for i in inputs),
    )
    return pre_rf, pre_stride


def get_op_stats(module, inputs, outputs):
    if not isinstance(outputs, tuple) and not isinstance(outputs, list):
        outputs = (outputs,)
    rst = {
        "input_shapes": [i.shape for i in inputs],
        "output_shapes": [o.shape for o in outputs],
    }
    valid_flag = False
    for key, _dict, fallback in _iter_list:
        for _type in _dict:
            if isinstance(module, _type):
                value = _dict[_type](module, inputs, outputs)
                valid_flag = True
                break
        else:
            if fallback is not None:
                value = fallback(module, inputs, outputs)
            continue

        if isinstance(key, tuple):
            assert isinstance(value, tuple)
            for k, v in zip(key, value):
                rst[k] = v
        else:
            rst[key] = value

    if valid_flag:
        return rst
    else:
        return None
    return


def sum_op_stats(flops, bar_length_max=20):
    max_flops_num = max([i["flops_num"] for i in flops] + [0])
    total_flops_num = 0
    for d in flops:
        total_flops_num += int(d["flops_num"])
        d["flops_cum"] = sizeof_fmt(total_flops_num, suffix="OPs")

    for d in flops:
        ratio = d["ratio"] = d["flops_num"] / total_flops_num
        d["percentage"] = "{:.2f}%".format(ratio * 100)
        bar_length = int(d["flops_num"] / max_flops_num * bar_length_max)
        d["bar"] = "#" * bar_length
        d["flops"] = sizeof_fmt(d["flops_num"], suffix="OPs")

    total_flops_str = sizeof_fmt(total_flops_num, suffix="OPs")
    total_var_size = sum(
        sum(s[1] if len(s) > 1 else 0 for s in d["output_shapes"]) for d in flops
    )
    flops.append(
        dict(name="total", flops=total_flops_str, output_shapes=total_var_size)
    )

    return total_flops_num, flops


def print_op_stats(flops):
    header = [
        "name",
        "class_name",
        "input_shapes",
        "output_shapes",
        "flops",
        "flops_cum",
        "percentage",
        "bar",
    ]
    if _receptive_field_enabled:
        header.insert(4, "receptive_field")
        header.insert(5, "stride")
    logger.info("flops stats: \n" + tabulate.tabulate(dict2table(flops, header=header)))


def get_param_stats(param: Tensor):
    nbits = get_dtype_bit(np.dtype(param.dtype).name)
    shape = param.shape
    param_dim = np.prod(param.shape)
    param_size = param_dim * nbits // 8
    return {
        "dtype": np.dtype(param.dtype),
        "shape": shape,
        "mean": "{:.3g}".format(_mean(param)),
        "std": "{:.3g}".format(_std(param)),
        "param_dim": param_dim,
        "nbits": nbits,
        "size": param_size,
    }


def sum_param_stats(params, bar_length_max=20):
    max_size = max([d["size"] for d in params] + [0])
    total_param_dims, total_param_size = 0, 0
    for d in params:
        total_param_dims += int(d["param_dim"])
        total_param_size += int(d["size"])
        d["size_cum"] = sizeof_fmt(total_param_size)

    for d in params:
        ratio = d["size"] / total_param_size
        d["ratio"] = ratio
        d["percentage"] = "{:.2f}%".format(ratio * 100)
        bar_length = int(d["size"] / max_size * bar_length_max)
        d["size_bar"] = "#" * bar_length
        d["size"] = sizeof_fmt(d["size"])

    param_size = sizeof_fmt(total_param_size)
    params.append(dict(name="total", param_dim=total_param_dims, size=param_size,))

    return total_param_dims, total_param_size, params


def print_param_stats(params):
    header = [
        "name",
        "dtype",
        "shape",
        "mean",
        "std",
        "param_dim",
        "nbits",
        "size",
        "size_cum",
        "percentage",
        "size_bar",
    ]
    logger.info(
        "param stats: \n" + tabulate.tabulate(dict2table(params, header=header))
    )


def get_activation_stats(output: Tensor, has_input=False):
    out_shape = output.shape
    activations_dtype = np.dtype(output.dtype)
    nbits = get_dtype_bit(activations_dtype.name)
    act_dim = np.prod(out_shape)
    act_size = act_dim * nbits // 8
    activation_stats = {
        "dtype": activations_dtype,
        "shape": out_shape,
        "act_dim": act_dim,
        "nbits": nbits,
        "size": act_size,
    }
    if has_input:
        activation_stats["mean"] = "{:.3g}".format(_mean(output))
        activation_stats["std"] = "{:.3g}".format(_std(output))
    return activation_stats


def sum_activations_stats(activations, bar_length_max=20):
    max_act_size = max([i["size"] for i in activations] + [0])
    total_act_dims, total_act_size = 0, 0
    for d in activations:
        total_act_size += int(d["size"])
        total_act_dims += int(d["act_dim"])
        d["size_cum"] = sizeof_fmt(total_act_size)

    for d in activations:
        ratio = d["ratio"] = d["size"] / total_act_size
        d["percentage"] = "{:.2f}%".format(ratio * 100)
        bar_length = int(d["size"] / max_act_size * bar_length_max)
        d["size_bar"] = "#" * bar_length
        d["size"] = sizeof_fmt(d["size"])

    act_size = sizeof_fmt(total_act_size)
    activations.append(dict(name="total", act_dim=total_act_dims, size=act_size,))

    return total_act_dims, total_act_size, activations


def print_activations_stats(activations, has_input=False):
    header = [
        "name",
        "class_name",
        "dtype",
        "shape",
        "nbits",
        "act_dim",
        "size",
        "size_cum",
        "percentage",
        "size_bar",
    ]
    if has_input:
        header.insert(4, "mean")
        header.insert(5, "std")
    logger.info(
        "activations stats: \n"
        + tabulate.tabulate(dict2table(activations, header=header))
    )


def print_summary(**kwargs):
    data = [["item", "value"]]
    data.extend(list(kwargs.items()))
    logger.info("summary\n" + tabulate.tabulate(data))


def module_stats(
    model: M.Module,
    inputs: Iterable[np.ndarray] = None,
    input_shapes: list = None,
    cal_params: bool = True,
    cal_flops: bool = True,
    cal_activations: bool = True,
    logging_to_stdout: bool = True,
    bar_length_max: int = 20,
):
    r"""Calculate and print ``model``'s statistics by adding hook and record Module's inputs outputs size.

    Args:
        model: model that need to get stats info.
        inputs: user defined input data for running model and calculating stats, alternative with input_shapes.
        input_shapes: shapes to generate random inputs for running model and calculating stats, alternative with inputs.
        cal_params: whether calculate and record params size.
        cal_flops: whether calculate and record op flops.
        cal_activations: whether calculate and record op activations.
        logging_to_stdout: whether print all calculated statistic details.
        bar_length_max: size of bar indicating max flops or parameter size in net stats.
    """
    has_inputs = False
    if inputs is not None:
        has_inputs = True
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]

        def load_tensor(x):
            if isinstance(x, np.ndarray):
                return Tensor(x)
            elif isinstance(x, collections.abc.Mapping):
                return {k: load_tensor(v) for k, v in x.items()}
            elif isinstance(x, tuple) and hasattr(x, "_fields"):  # nametuple
                return type(x)(*(load_tensor(value) for value in x))
            elif isinstance(x, collections.abc.Sequence):
                return [load_tensor(v) for v in x]
            else:
                return Tensor(x, dtype=np.float32)

        inputs = load_tensor(inputs)

    else:
        if input_shapes:
            if not isinstance(input_shapes[0], tuple):
                input_shapes = [input_shapes]
            inputs = [F.zeros(in_size, dtype=np.float32) for in_size in input_shapes]
        else:
            logger.error(
                "Inputs or input_shapes is required for running model and calculating stats.",
                exc_info=True,
            )
            return
    if not cal_activations:
        log_activations = False

    disable_receptive_field()
    recorded_parameters = set()

    def module_stats_hook(module, inputs, outputs, name=""):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        if cal_flops:
            flops_stats = get_op_stats(module, inputs, outputs)
            if flops_stats is not None:
                flops_stats["name"] = name
                flops_stats["class_name"] = class_name
                flops.append(flops_stats)

        if cal_params:
            if (
                hasattr(module, "weight")
                and (module.weight is not None)
                and module.weight not in recorded_parameters
            ):
                w = module.weight
                param_stats = get_param_stats(w)
                param_stats["name"] = name + "-w"
                params.append(param_stats)
                recorded_parameters.add(w)

            if (
                hasattr(module, "bias")
                and module.bias is not None
                and module.bias not in recorded_parameters
            ):
                b = module.bias
                param_stats = get_param_stats(b)
                param_stats["name"] = name + "-b"
                params.append(param_stats)
                recorded_parameters.add(b)

        if cal_activations:
            if not isinstance(outputs, (tuple, list)):
                output = outputs
            else:
                output = outputs[0]
            activation_stats = get_activation_stats(output, has_inputs)
            activation_stats["name"] = name
            activation_stats["class_name"] = class_name
            activations.append(activation_stats)

    params = []
    flops = []
    hooks = []
    activations = []
    total_stats = namedtuple(
        "total_stats", ["param_size", "param_dims", "flops", "act_size", "act_dims"]
    )
    stats_details = namedtuple("module_stats", ["params", "flops", "activations"])

    module_to_name = dict()
    for (name, module) in model.named_modules():
        if isinstance(module, tuple(hook_modules)):
            hooks.append(
                module.register_forward_hook(partial(module_stats_hook, name=name))
            )
            module_to_name[module] = name

    @contextmanager
    def param_stat_context():
        def wrapper(fun):
            @functools.wraps(fun)
            def param_access_record(module, item):
                member = fun(module, item)
                if (
                    item in ["weight", "bias"]
                    and member is not None
                    and member not in recorded_parameters
                ):
                    name = module_to_name[module]
                    if item == "weight":
                        suffix = "-w"
                    elif item == "bias":
                        suffix = "-b"

                    param_name = name + suffix
                    param_stats = get_param_stats(member)
                    param_stats["name"] = param_name
                    params.append(param_stats)
                    recorded_parameters.add(member)

                return member

            return param_access_record

        origin_get_attr = object.__getattribute__
        try:
            M.Module.__getattribute__ = wrapper(origin_get_attr)
            yield
        finally:
            M.Module.__getattribute__ = origin_get_attr

    with set_module_mode_safe(model, training=False) as model, param_stat_context():
        model(*inputs)

    for h in hooks:
        h.remove()

    extra_info = {
        "#params": len(params),
    }
    (
        total_flops,
        total_param_dims,
        total_param_size,
        total_act_dims,
        total_act_size,
    ) = (0, 0, 0, 0, 0)

    if cal_params:
        total_param_dims, total_param_size, params = sum_param_stats(
            params, bar_length_max
        )
        extra_info["total_param_dims"] = sizeof_fmt(total_param_dims, suffix="")
        extra_info["total_param_size"] = sizeof_fmt(total_param_size)
        if logging_to_stdout:
            print_param_stats(params)

    if cal_flops:
        total_flops, flops = sum_op_stats(flops, bar_length_max)
        extra_info["total_flops"] = sizeof_fmt(total_flops, suffix="OPs")
        if logging_to_stdout:
            print_op_stats(flops)

    if cal_activations:
        total_act_dims, total_act_size, activations = sum_activations_stats(
            activations, bar_length_max
        )
        extra_info["total_act_dims"] = sizeof_fmt(total_act_dims, suffix="")
        extra_info["total_act_size"] = sizeof_fmt(total_act_size)
        if logging_to_stdout:
            print_activations_stats(activations, has_inputs)

    if cal_flops and cal_params and total_param_size != 0:
        extra_info["flops/param_size"] = "{:3.3f}".format(
            total_flops / total_param_size
        )

    print_summary(**extra_info)

    return (
        total_stats(
            param_size=total_param_size,
            param_dims=total_param_dims,
            flops=total_flops,
            act_size=total_act_size,
            act_dims=total_act_dims,
        ),
        stats_details(params=params, flops=flops, activations=activations),
    )
