# -*- coding: utf-8 -*-
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from tabulate import tabulate

import megengine as mge
import megengine.functional as MF
import megengine.module as MM

module_cache = {
    "conv2d": (MM.Conv2d(32, 32, 3, 1, 0), nn.Conv2d(32, 32, 3, 1, 0).cuda()),
    "dw_conv2d": (
        MM.Conv2d(32, 32, 3, 1, 0, groups=32),
        nn.Conv2d(32, 32, 3, 1, 0, groups=32).cuda(),
    ),
    "conv3d": (MM.Conv3d(32, 32, 3, 1, 0), nn.Conv3d(32, 32, 3, 1, 0).cuda()),
    "ConvTranspose2d": (
        MM.ConvTranspose2d(32, 32, 3, 1, 0),
        nn.ConvTranspose2d(32, 32, 3, 1, 0).cuda(),
    ),
    "BatchNorm2d": (MM.BatchNorm2d(64), nn.BatchNorm2d(64).cuda()),
    "Linear": (MM.Linear(1000, 1000), nn.Linear(1000, 1000).cuda()),
}

test_cases = [
    # (mge op, torch op, small inps, large inps, unpack_inps, rep)
    (
        "adaptive_avg_pool2d",
        lambda x: MF.adaptive_avg_pool2d(x, (7, 7)),
        lambda x: TF.adaptive_avg_pool2d(x, (7, 7)),
        [(2, 32, 16, 16)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "adaptive_max_pool2d",
        lambda x: MF.adaptive_max_pool2d(x, (7, 7)),
        lambda x: TF.adaptive_max_pool2d(x, (7, 7)),
        [(2, 32, 16, 16)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    ("argsort", MF.argsort, torch.argsort, [(1000,)], [(1000, 1000),], True, 1000),
    (
        "avg_pool2d",
        lambda x: MF.avg_pool2d(x, 2),
        lambda x: TF.avg_pool2d(x, 2),
        [(2, 32, 16, 16)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "broadcast",
        lambda x: MF.broadcast_to(x, (5,) + x.shape),
        lambda x: torch.broadcast_to(x, (5,) + x.shape),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "batchedmatmul",
        MF.matmul,
        torch.matmul,
        [(8, 64, 32), (8, 32, 64)],
        [(8, 2048, 512), (8, 512, 2048)],
        True,
        1000,
    ),
    (
        "batchnrom2d",
        lambda x: module_cache["BatchNorm2d"][0](x),
        lambda x: module_cache["BatchNorm2d"][1](x),
        [(2, 64, 16, 16)],
        [(64, 64, 128, 128)],
        True,
        1000,
    ),
    (
        "concat",
        MF.concat,
        torch.cat,
        [(20, 100), (50, 100), (30, 100)],
        [(64, 512, 16, 16), (64, 512, 16, 16), (64, 512, 16, 16)],
        False,
        1000,
    ),
    (
        "conv2d",
        lambda x: module_cache["conv2d"][0](x),
        lambda x: module_cache["conv2d"][1](x),
        [(2, 32, 16, 16)],
        [(32, 32, 128, 128)],
        True,
        1000,
    ),
    (
        "conv3d",
        lambda x: module_cache["conv3d"][0](x),
        lambda x: module_cache["conv3d"][1](x),
        [(2, 32, 8, 8, 8)],
        [(32, 32, 16, 16, 16)],
        True,
        1000,
    ),
    (
        "convTranspose2d",
        lambda x: module_cache["ConvTranspose2d"][0](x),
        lambda x: module_cache["ConvTranspose2d"][1](x),
        [(2, 32, 16, 16)],
        [(32, 32, 128, 128)],
        True,
        1000,
    ),
    (
        "dropout",
        lambda x: MF.dropout(x, 0.5),
        TF.dropout,
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "dw_conv2d",
        lambda x: module_cache["dw_conv2d"][0](x),
        lambda x: module_cache["dw_conv2d"][1](x),
        [(2, 32, 16, 16)],
        [(32, 32, 128, 128)],
        True,
        1000,
    ),
    (
        "elemwise.unary",
        MF.log,
        torch.log,
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "elemwise.binary",
        MF.add,
        torch.add,
        [(100, 100), (100, 100)],
        [(64, 512, 16, 16), (64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "expand_dims",
        lambda x: MF.expand_dims(x, 0),
        lambda x: torch.unsqueeze(x, 0),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    ("gelu", MF.gelu, TF.gelu, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    ("hswish", MF.hswish, TF.hardswish, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    (
        "hsigmoid",
        MF.hsigmoid,
        TF.hardsigmoid,
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    ("isinf", MF.isinf, torch.isinf, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    (
        "indeixngMultiAxisVec",
        lambda x: x[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lambda x: x[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [(10, 10, 10, 10)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "logsigmoid",
        MF.logsigmoid,
        TF.logsigmoid,
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "leaky_relu",
        lambda x: MF.leaky_relu(x, 0.5),
        lambda x: TF.leaky_relu(x, 0.5),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "linear",
        lambda x: module_cache["Linear"][0](x),
        lambda x: module_cache["Linear"][1](x),
        [(10, 1000)],
        [(64, 128, 1000)],
        True,
        1000,
    ),
    ("matinv", MF.matinv, torch.inverse, [(10, 10)], [(30, 30)], True, 1000),
    (
        "matmul",
        MF.matmul,
        torch.matmul,
        [(64, 32), (32, 64)],
        [(2048, 1024), (1024, 2048)],
        True,
        1000,
    ),
    (
        "max_pool2d",
        lambda x: MF.max_pool2d(x, 2),
        lambda x: TF.max_pool2d(x, 2),
        [(2, 32, 16, 16)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "normal",
        lambda x: mge.random.normal(0, 1, x.shape),
        lambda x: torch.randn(x.shape, device="cuda"),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "prelu",
        MF.prelu,
        TF.prelu,
        [(100, 100), (1,)],
        [(64, 512, 16, 16), (1,)],
        True,
        1000,
    ),
    (
        "reduce.max",
        lambda x: MF.max(x, 0),
        lambda x: torch.max(x, 0),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "reduce.mean",
        lambda x: MF.mean(x, 0),
        lambda x: torch.mean(x, 0),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "reduce.mean",
        lambda x: MF.mean(x, 0),
        lambda x: torch.mean(x, 0),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    ("relu", MF.relu, TF.relu, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    ("relu6", MF.relu6, TF.relu6, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    (
        "repeat",
        lambda x: MF.repeat(x, 5),
        lambda x: torch.repeat_interleave(x, 5),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    ("silu", MF.silu, TF.silu, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    (
        "split",
        lambda x: MF.split(x, 5),
        lambda x: torch.split(x, 5),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    ("sigmoid", MF.sigmoid, TF.sigmoid, [(100, 100)], [(64, 512, 16, 16)], True, 1000),
    (
        "softmax",
        lambda x: MF.softmax(x, axis=1),
        lambda x: TF.softmax(x, dim=1),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "softplus",
        MF.softplus,
        TF.softplus,
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "squeeze",
        lambda x: MF.squeeze(x, 0),
        lambda x: torch.squeeze(x, 0),
        [(1, 100, 100)],
        [(1, 64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "stack",
        MF.stack,
        torch.stack,
        [(100, 100), (100, 100)],
        [(64, 512, 16, 16), (64, 512, 16, 16)],
        False,
        10000,
    ),
    (
        "subtensor",
        lambda x: x[0:20, 10:60],
        lambda x: x[0:20, 10:60],
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "topk",
        lambda x: MF.topk(x, 10),
        lambda x: torch.topk(x, 10),
        [(100, 100)],
        [(1000, 1000)],
        True,
        1000,
    ),
    (
        "tile",
        lambda x: MF.tile(x, (2,) * len(x.shape)),
        lambda x: torch.tile(x, (2,) * len(x.shape)),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "transpose",
        lambda x: MF.transpose(x, list(range(len(x.shape)))[::-1]),
        lambda x: torch.permute(x, list(range(len(x.shape)))[::-1]),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "where",
        lambda x: MF.where(x > 0.5, x, x),
        lambda x: torch.where(x > 0.5, x, x),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
    (
        "uniform",
        lambda x: mge.random.uniform(0, 1, x.shape),
        lambda x: torch.rand(x.shape, device="cuda"),
        [(100, 100)],
        [(64, 512, 16, 16)],
        True,
        1000,
    ),
]


def perf_func(func, inps, reps, unpack_inps, is_mge):
    if is_mge:
        mge._full_sync()
        tik = time.time()
        for _ in range(reps):
            if unpack_inps:
                out = func(*inps)
            else:
                out = func(inps)
        mge._full_sync()
    else:
        torch.cuda.synchronize()
        with torch.no_grad():
            tik = time.time()
            for _ in range(reps):
                if unpack_inps:
                    out = func(*inps)
                else:
                    out = func(inps)
        torch.cuda.synchronize()
    return time.time() - tik


def get_avg_time(func, inps, reps, unpack_inps, is_mge):
    # warm up
    for _ in range(2):
        t = perf_func(func, inps, reps, unpack_inps, is_mge)

    times = []
    for _ in range(5):
        t = perf_func(func, inps, reps, unpack_inps, is_mge)
        times.append(t)
    return np.mean(times)


def get_perf_results(mge_func, torch_func, shapes, unpack_inps, reps):
    inps = [np.random.randn(*shape) for shape in shapes]

    inps_mge = [mge.tensor(inp, dtype="float32") for inp in inps]
    avg_time_mge = get_avg_time(mge_func, inps_mge, reps, unpack_inps, True)

    inps_torch = [torch.Tensor(inp).type(torch.float).cuda() for inp in inps]
    avg_time_torch = get_avg_time(torch_func, inps_torch, reps, unpack_inps, False)

    return avg_time_mge, avg_time_torch


if __name__ == "__main__":
    header = [
        "opr_name",
        "time(mge/pytorch; small input)",
        "time(mge/pytorch; large input)",
    ]
    table = []
    for case in test_cases:
        assert len(case) == 7
        name, mge_func, torch_func, small_shapes, large_shapes, unpack_inps, reps = case
        data = []
        data.append(name)
        print("========== op: {}".format(name))

        avg_time_mge, avg_time_torch = get_perf_results(
            mge_func, torch_func, small_shapes, unpack_inps, reps
        )
        print("mge time: {}".format(avg_time_mge))
        print("torch time: {}".format(avg_time_torch))
        data.append("{:.2f}".format(avg_time_mge / avg_time_torch))

        avg_time_mge, avg_time_torch = get_perf_results(
            mge_func, torch_func, large_shapes, unpack_inps, reps
        )
        print("mge time: {}".format(avg_time_mge))
        print("torch time: {}".format(avg_time_torch))
        data.append("{:.2f}".format(avg_time_mge / avg_time_torch))
        table.append(data)
    print(tabulate(table, header, tablefmt="github"))
