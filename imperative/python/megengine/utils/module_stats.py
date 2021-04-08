# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import contextlib
from functools import partial

import numpy as np
import tabulate

import megengine as mge
import megengine.module as m
import megengine.module.qat as qatm
import megengine.module.quantized as qm
from megengine.core.tensor.dtype import get_dtype_bit
from megengine.functional.tensor import zeros

try:
    mge.logger.MegEngineLogFormatter.max_lines = float("inf")
except AttributeError as e:
    raise ValueError("set logger max lines failed")

logger = mge.get_logger(__name__)
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


@register_flops(
    m.Conv1d, m.Conv2d, m.Conv3d, m.ConvTranspose2d, m.LocalConv2d, m.DeformableConv2d
)
def flops_convNd(module: m.Conv2d, inputs, outputs):
    bias = 1 if module.bias is not None else 0
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    return np.prod(outputs[0].shape) * (
        module.in_channels // module.groups * np.prod(module.kernel_size) + bias
    )


@register_flops(m.Linear)
def flops_linear(module: m.Linear, inputs, outputs):
    bias = module.out_features if module.bias is not None else 0
    return np.prod(outputs[0].shape) * module.in_features + bias


@register_flops(m.BatchMatMulActivation)
def flops_batchmatmul(module: m.BatchMatMulActivation, inputs, outputs):
    bias = 1 if module.bias is not None else 0
    x = inputs[0]
    w = module.weight
    batch_size = x.shape[0]
    n, p = x.shape[1:]
    _, m = w.shape[1:]
    return n * (p + bias) * m * batch_size


# does not need import qat and quantized module since they inherit from float module.
hook_modules = (
    m.conv._ConvNd,
    m.Linear,
    m.BatchMatMulActivation,
)


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
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "{:3.3f} {}{}".format(num, unit, suffix)
        num /= 1024.0
    sign_str = "-" if num < 0 else ""
    return "{}{:.1f} {}{}".format(sign_str, num, "Yi", suffix)


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


def print_op_stats(flops, bar_length_max=20):
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

    total_flops_str = sizeof_fmt(total_flops_num, suffix="OPs")
    total_var_size = sum(
        sum(s[1] if len(s) > 1 else 0 for s in d["output_shapes"]) for d in flops
    )
    flops.append(
        dict(name="total", flops=total_flops_str, output_shapes=total_var_size)
    )

    logger.info("flops stats: \n" + tabulate.tabulate(dict2table(flops, header=header)))

    return total_flops_num


def get_param_stats(param: np.ndarray):
    nbits = get_dtype_bit(param.dtype.name)
    shape = param.shape
    param_dim = np.prod(param.shape)
    param_size = param_dim * nbits // 8
    return {
        "dtype": param.dtype,
        "shape": shape,
        "mean": "{:.3g}".format(param.mean()),
        "std": "{:.3g}".format(param.std()),
        "param_dim": param_dim,
        "nbits": nbits,
        "size": param_size,
    }


def print_param_stats(params, bar_length_max=20):
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

    header = [
        "name",
        "dtype",
        "shape",
        "mean",
        "std",
        "param_dim",
        "bits",
        "size",
        "size_cum",
        "percentage",
        "size_bar",
    ]

    logger.info(
        "param stats: \n" + tabulate.tabulate(dict2table(params, header=header))
    )

    return total_param_dims, total_param_size


def print_summary(**kwargs):
    data = [["item", "value"]]
    data.extend(list(kwargs.items()))
    logger.info("summary\n" + tabulate.tabulate(data))


def module_stats(
    model: m.Module,
    input_size: int,
    bar_length_max: int = 20,
    log_params: bool = True,
    log_flops: bool = True,
):
    r"""
    Calculate and print ``model``'s statistics by adding hook and record Module's inputs outputs size.

    :param model: model that need to get stats info.
    :param input_size: size of input for running model and calculating stats.
    :param bar_length_max: size of bar indicating max flops or parameter size in net stats.
    :param log_params: whether print and record params size.
    :param log_flops: whether print and record op flops.
    """
    disable_receptive_field()

    def module_stats_hook(module, inputs, outputs, name=""):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]

        flops_stats = get_op_stats(module, inputs, outputs)
        if flops_stats is not None:
            flops_stats["name"] = name
            flops_stats["class_name"] = class_name
            flops.append(flops_stats)

        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight
            param_stats = get_param_stats(w.numpy())
            param_stats["name"] = name + "-w"
            params.append(param_stats)

        if hasattr(module, "bias") and module.bias is not None:
            b = module.bias
            param_stats = get_param_stats(b.numpy())
            param_stats["name"] = name + "-b"
            params.append(param_stats)

    @contextlib.contextmanager
    def adjust_stats(module, training=False):
        """Adjust module to training/eval mode temporarily.

        Args:
            module (M.Module): used module.
            training (bool): training mode. True for train mode, False fro eval mode.
        """

        def recursive_backup_stats(module, mode):
            for m in module.modules():
                # save prev status to _prev_training
                m._prev_training = m.training
                m.train(mode, recursive=False)

        def recursive_recover_stats(module):
            for m in module.modules():
                # recover prev status and delete attribute
                m.training = m._prev_training
                delattr(m, "_prev_training")

        recursive_backup_stats(module, mode=training)
        yield module
        recursive_recover_stats(module)

    # multiple inputs to the network
    if not isinstance(input_size[0], tuple):
        input_size = [input_size]

    params = []
    flops = []
    hooks = []

    for (name, module) in model.named_modules():
        if isinstance(module, hook_modules):
            hooks.append(
                module.register_forward_hook(partial(module_stats_hook, name=name))
            )

    inputs = [zeros(in_size, dtype=np.float32) for in_size in input_size]
    with adjust_stats(model, training=False) as model:
        model(*inputs)

    for h in hooks:
        h.remove()

    extra_info = {
        "#params": len(params),
    }
    total_flops, total_param_dims, total_param_size = 0, 0, 0
    if log_params:
        total_param_dims, total_param_size = print_param_stats(params, bar_length_max)
        extra_info["total_param_dims"] = sizeof_fmt(total_param_dims)
        extra_info["total_param_size"] = sizeof_fmt(total_param_size)
    if log_flops:
        total_flops = print_op_stats(flops, bar_length_max)
        extra_info["total_flops"] = sizeof_fmt(total_flops, suffix="OPs")
    if log_params and log_flops:
        extra_info["flops/param_size"] = "{:3.3f}".format(
            total_flops / total_param_size
        )

    print_summary(**extra_info)

    return total_param_size, total_flops
