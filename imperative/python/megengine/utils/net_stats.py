# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from functools import partial

import numpy as np
import tabulate

import megengine as mge
import megengine.core.tensor.dtype as dtype
import megengine.module as m
import megengine.module.qat as qatm
import megengine.module.quantized as qm
from megengine.functional.tensor import zeros

try:
    mge.logger.MegEngineLogFormatter.max_lines = float("inf")
except AttributeError as e:
    raise ValueError("set logger max lines failed")

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


CALC_FLOPS = {}


def _register_modules(*modules):
    def callback(impl):
        for module in modules:
            CALC_FLOPS[module] = impl
        return impl

    return callback


@_register_modules(
    m.Conv2d,
    m.ConvTranspose2d,
    m.LocalConv2d,
    qm.Conv2d,
    qm.ConvRelu2d,
    qm.ConvBn2d,
    qm.ConvBnRelu2d,
    qatm.Conv2d,
    qatm.ConvRelu2d,
    qatm.ConvBn2d,
    qatm.ConvBnRelu2d,
)
def count_convNd(module, input, output):
    bias = 1 if module.bias is not None else 0
    group = module.groups
    ic = input[0].shape[1]
    oc = output[0].shape[1]
    goc = oc // group
    gic = ic // group
    N = output[0].shape[0]
    HW = np.prod(output[0].shape[2:])
    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    return N * HW * goc * (gic * np.prod(module.kernel_size) + bias)


@_register_modules(m.ConvTranspose2d)
def count_deconvNd(module, input, output):
    return np.prod(input[0].shape) * output[0].shape[1] * np.prod(module.kernel_size)


@_register_modules(m.Linear, qatm.Linear, qm.Linear)
def count_linear(module, input, output):
    return np.prod(output[0].shape) * module.in_features


# does not need import qat and quantized module since they inherit from float module.
hook_modules = (
    m.Conv2d,
    m.ConvTranspose2d,
    m.LocalConv2d,
    m.BatchNorm2d,
    m.Linear,
)


def net_stats(model, input_size, bar_length_max=20, log_params=True, log_flops=True):
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

    def get_byteswidth(tensor):
        if dtype.is_quantize(tensor.dtype):
            return 1
        # elif dtype.is_bfloat16(tensor.dtype):
        #      return 2
        else:
            return 4

    def print_flops_stats(flops):
        flops_list = [i["flops_num"] for i in flops]
        max_flops_num = max(flops_list + [0])
        # calc total flops and set flops_cum
        total_flops_num = 0
        for d in flops:
            total_flops_num += int(d["flops_num"])
            d["flops_cum"] = sizeof_fmt(total_flops_num, suffix="OPs")

        for i in flops:
            f = i["flops_num"]
            i["flops"] = sizeof_fmt(f, suffix="OPs")
            r = i["ratio"] = f / total_flops_num
            i["percentage"] = "{:.2f}%".format(r * 100)
            bar_length = int(f / max_flops_num * bar_length_max)
            i["bar"] = "#" * bar_length

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

        total_flops_str = sizeof_fmt(total_flops_num, suffix="OPs")
        total_var_size = sum(sum(s[1] for s in i["output_shapes"]) for i in flops)
        flops.append(
            dict(name="total", flops=total_flops_str, output_shapes=total_var_size)
        )

        logger.info(
            "flops stats: \n" + tabulate.tabulate(dict2table(flops, header=header))
        )

        return total_flops_num

    def print_params_stats(params):
        total_param_dims, total_param_size = 0, 0
        for d in params:
            total_param_dims += int(d["param_dim"])
            total_param_size += int(d["size"])
            d["size"] = sizeof_fmt(d["size"])
            d["size_cum"] = sizeof_fmt(total_param_size)

        for d in params:
            ratio = d["param_dim"] / total_param_dims
            d["ratio"] = ratio
            d["percentage"] = "{:.2f}%".format(ratio * 100)

        # construct bar
        max_ratio = max([d["ratio"] for d in params])
        for d in params:
            bar_length = int(d["ratio"] / max_ratio * bar_length_max)
            d["size_bar"] = "#" * bar_length

        param_size = sizeof_fmt(total_param_size)
        params.append(dict(name="total", param_dim=total_param_dims, size=param_size,))

        header = [
            "name",
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

        return total_param_size

    def net_stats_hook(module, input, output, name=""):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]

        flops_fun = CALC_FLOPS.get(type(module))
        if callable(flops_fun):
            flops_num = flops_fun(module, input, output)

            if not isinstance(output, (list, tuple)):
                output = [output]

            flops.append(
                dict(
                    name=name,
                    class_name=class_name,
                    input_shapes=[i.shape for i in input],
                    output_shapes=[o.shape for o in output],
                    flops_num=flops_num,
                    flops_cum=0,
                )
            )

        if hasattr(module, "weight") and module.weight is not None:
            w = module.weight
            value = w.numpy()
            param_dim = np.prod(w.shape)
            param_bytes = get_byteswidth(w)
            params.append(
                dict(
                    name=name + "-w",
                    shape=w.shape,
                    param_dim=param_dim,
                    bits=param_bytes * 8,
                    size=param_dim * param_bytes,
                    size_cum=0,
                    mean="{:.2g}".format(value.mean()),
                    std="{:.2g}".format(value.std()),
                )
            )

        if hasattr(module, "bias") and module.bias is not None:
            b = module.bias
            value = b.numpy()
            param_dim = np.prod(b.shape)
            param_bytes = get_byteswidth(b)
            params.append(
                dict(
                    name=name + "-b",
                    shape=b.shape,
                    param_dim=param_dim,
                    bits=param_bytes * 8,
                    size=param_dim * param_bytes,
                    size_cum=0,
                    mean="{:.2g}".format(value.mean()),
                    std="{:.2g}".format(value.std()),
                )
            )

    # multiple inputs to the network
    if not isinstance(input_size[0], tuple):
        input_size = [input_size]

    params = []
    flops = []
    hooks = []

    for (name, module) in model.named_modules():
        if isinstance(module, hook_modules):
            hooks.append(
                module.register_forward_hook(partial(net_stats_hook, name=name))
            )

    inputs = [zeros(in_size, dtype=np.float32) for in_size in input_size]
    model.eval()
    model(*inputs)
    for h in hooks:
        h.remove()

    total_flops, total_params = 0, 0
    if log_params:
        total_params = print_params_stats(params)
    if log_flops:
        total_flops = print_flops_stats(flops)

    return total_params, total_flops
