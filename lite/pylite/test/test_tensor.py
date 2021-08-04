# -*- coding: utf-8 -*-
# This file is part of MegEngine, a deep learning framework developed by
# Megvii.
#
# Copyright (c) Copyright (c) 2020-2021 Megvii Inc. All rights reserved.

import functools

import numpy as np

from megenginelite import *


def require_cuda(func):
    """a decorator that disables a testcase if cuda is not enabled"""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if LiteGlobal.get_device_count(LiteDeviceType.LITE_CUDA):
            return func(*args, **kwargs)

    return wrapped


def test_tensor_make():
    empty_layout = LiteLayout()
    assert empty_layout.ndim == 0
    assert empty_layout.data_type == int(LiteDataType.LITE_FLOAT)

    empty_tensor = LiteTensor()
    assert empty_tensor.layout.ndim == empty_layout.ndim
    assert empty_tensor.layout.data_type == empty_layout.data_type

    layout = LiteLayout([4, 16])
    layout = LiteLayout(dtype="float32")
    layout = LiteLayout([4, 16], "float32")
    layout = LiteLayout([4, 16], "float16")
    layout = LiteLayout([4, 16], np.float32)
    layout = LiteLayout([4, 16], np.int8)
    layout = LiteLayout([4, 16], LiteDataType.LITE_FLOAT)

    tensor = LiteTensor(layout)
    tensor = LiteTensor(layout, LiteDeviceType.LITE_CPU)
    assert tensor.layout == layout
    assert tensor.device_type == LiteDeviceType.LITE_CPU
    assert tensor.is_continue == True
    assert tensor.is_pinned_host == False
    assert tensor.nbytes == 4 * 16 * 4
    assert tensor.device_id == 0

    tensor = LiteTensor(layout, device_id=1)
    assert tensor.device_id == 1


def test_tensor_set_data():
    layout = LiteLayout([2, 16], "int8")
    tensor = LiteTensor(layout)
    assert tensor.nbytes == 2 * 16

    data = [i for i in range(32)]
    tensor.set_data_by_copy(data)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == i

    arr = np.ones([2, 16], "int8")
    tensor.set_data_by_copy(arr)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == 1

    for i in range(32):
        arr[i // 16][i % 16] = i
    tensor.set_data_by_share(arr)
    real_data = tensor.to_numpy()
    for i in range(32):
        assert real_data[i // 16][i % 16] == i

    arr[0][8] = 100
    arr[1][3] = 20
    real_data = tensor.to_numpy()
    assert real_data[0][8] == 100
    assert real_data[1][3] == 20


def test_fill_zero():
    layout = LiteLayout([4, 8], "int16")
    tensor1 = LiteTensor(layout)
    assert tensor1.nbytes == 4 * 8 * 2

    tensor1.set_data_by_copy([i for i in range(32)])
    real_data = tensor1.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i

    tensor1.fill_zero()
    real_data = tensor1.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == 0


def test_copy_from():
    layout = LiteLayout([4, 8], "int16")
    tensor1 = LiteTensor(layout)
    tensor2 = LiteTensor(layout)
    assert tensor1.nbytes == 4 * 8 * 2
    assert tensor2.nbytes == 4 * 8 * 2

    tensor1.set_data_by_copy([i for i in range(32)])
    tensor2.copy_from(tensor1)
    real_data = tensor2.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i

    tensor1.set_data_by_copy([i + 5 for i in range(32)])
    tensor2.copy_from(tensor1)
    real_data = tensor2.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i + 5


def test_reshape():
    layout = LiteLayout([4, 8], "int16")
    tensor1 = LiteTensor(layout)
    assert tensor1.nbytes == 4 * 8 * 2

    tensor1.set_data_by_copy([i for i in range(32)])
    real_data = tensor1.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i

    tensor1.reshape([8, 4])
    real_data = tensor1.to_numpy()
    for i in range(32):
        assert real_data[i // 4][i % 4] == i


def test_slice():
    layout = LiteLayout([4, 8], "int32")
    tensor1 = LiteTensor(layout)
    assert tensor1.nbytes == 4 * 8 * 4

    tensor1.set_data_by_copy([i for i in range(32)])
    real_data_org = tensor1.to_numpy()
    for i in range(32):
        assert real_data_org[i // 8][i % 8] == i

    tensor2 = tensor1.slice([1, 4], [3, 8])
    assert tensor2.layout.shapes[0] == 2
    assert tensor2.layout.shapes[1] == 4
    assert tensor2.is_continue == False

    real_data = tensor2.to_numpy()
    for i in range(8):
        row = i // 4
        col = i % 4
        assert real_data[row][col] == real_data_org[row + 1][col + 4]


def test_tensor_share_memory():
    layout = LiteLayout([4, 8], "int16")
    tensor1 = LiteTensor(layout)
    tensor2 = LiteTensor(layout)
    assert tensor1.nbytes == 4 * 8 * 2
    assert tensor2.nbytes == 4 * 8 * 2

    tensor1.set_data_by_copy([i for i in range(32)])
    tensor2.share_memory_with(tensor1)
    real_data = tensor2.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i

    tensor1.set_data_by_copy([i + 5 for i in range(32)])
    real_data = tensor2.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i + 5


def test_tensor_share_ctype_memory():
    layout = LiteLayout([4, 8], "int16")
    tensor1 = LiteTensor(layout)
    assert tensor1.nbytes == 4 * 8 * 2

    arr = np.ones([4, 8], "int16")
    for i in range(32):
        arr[i // 8][i % 8] = i
    tensor1.set_data_by_share(arr.ctypes.data, 4 * 8 * 2)
    real_data = tensor1.to_numpy()
    for i in range(32):
        assert real_data[i // 8][i % 8] == i


@require_cuda
def test_tensor_share_ctype_memory_device():
    layout = LiteLayout([4, 8], "int16")
    tensor_cpu = LiteTensor(
        layout=layout, device_type=LiteDeviceType.LITE_CUDA, is_pinned_host=True
    )
    tensor_cuda1 = LiteTensor(layout=layout, device_type=LiteDeviceType.LITE_CUDA)
    tensor_cuda2 = LiteTensor(layout=layout, device_type=LiteDeviceType.LITE_CUDA)
    assert tensor_cpu.nbytes == 4 * 8 * 2
    assert tensor_cuda1.nbytes == 4 * 8 * 2
    assert tensor_cuda2.nbytes == 4 * 8 * 2

    arr = np.ones([4, 8], "int16")
    for i in range(32):
        arr[i // 8][i % 8] = i
    tensor_cpu.set_data_by_share(arr.ctypes.data, 4 * 8 * 2)
    tensor_cuda1.copy_from(tensor_cpu)
    device_mem = tensor_cuda1.get_ctypes_memory()
    tensor_cuda2.set_data_by_share(device_mem, tensor_cuda1.nbytes)
    real_data1 = tensor_cuda1.to_numpy()
    real_data2 = tensor_cuda2.to_numpy()
    for i in range(32):
        assert real_data1[i // 8][i % 8] == i
        assert real_data2[i // 8][i % 8] == i


def test_tensor_share_memory_with():
    layout = LiteLayout([4, 32], "int16")
    tensor = LiteTensor(layout)
    assert tensor.nbytes == 4 * 32 * 2

    arr = np.ones([4, 32], "int16")
    for i in range(128):
        arr[i // 32][i % 32] = i
    tensor.set_data_by_share(arr)
    real_data = tensor.to_numpy()
    for i in range(128):
        assert real_data[i // 32][i % 32] == i

    tensor2 = LiteTensor(layout)
    tensor2.share_memory_with(tensor)
    real_data = tensor.to_numpy()
    real_data2 = tensor2.to_numpy()
    for i in range(128):
        assert real_data[i // 32][i % 32] == i
        assert real_data2[i // 32][i % 32] == i

    arr[1][18] = 5
    arr[3][7] = 345
    real_data = tensor2.to_numpy()
    assert real_data[1][18] == 5
    assert real_data[3][7] == 345


def test_empty_tensor():
    empty_tensor = LiteTensor()
    assert empty_tensor.layout.ndim == 0
    assert empty_tensor.layout.data_type == int(LiteDataType.LITE_FLOAT)
    # check empty tensor to numpy
    data = empty_tensor.to_numpy()


def test_tensor_by_set_copy_with_new_layout():
    layout = LiteLayout([4, 32], "int16")
    tensor = LiteTensor(layout)
    assert tensor.nbytes == 4 * 32 * 2

    arr = np.ones([8, 64], "int32")
    tensor.set_data_by_copy(arr)
    new_layout = tensor.layout
    assert new_layout.ndim == 2
    assert new_layout.shapes[0] == 8
    assert new_layout.shapes[1] == 64

    tensor = LiteTensor(layout)
    tensor.set_data_by_share(arr)
    new_layout = tensor.layout
    assert new_layout.ndim == 2
    assert new_layout.shapes[0] == 8
    assert new_layout.shapes[1] == 64


def test_tensor_concat():
    layout = LiteLayout([4, 32], "int16")
    tensors = []
    arr = np.ones([4, 32], "int16")
    for j in range(4):
        for i in range(128):
            arr[i // 32][i % 32] = j
        tensor = LiteTensor(layout)
        tensor.set_data_by_copy(arr)
        tensors.append(tensor)
    new_tensor = LiteTensorConcat(tensors, 0)

    real_data = new_tensor.to_numpy()
    for j in range(4):
        for i in range(128):
            index = j * 128 + i
            assert real_data[index // 32][index % 32] == j
