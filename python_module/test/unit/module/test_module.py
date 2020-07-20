# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import tempfile
from collections import OrderedDict
from io import BytesIO

import numpy as np
import pytest
from helpers import MLP

import megengine as mge
import megengine._internal as mgb
from megengine.core import Buffer, Parameter, Tensor, tensor
from megengine.module import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    Sequential,
)
from megengine.quantization.quantize import quantize, quantize_qat
from megengine.test import assertTensorClose


class MyModule(Module):
    class InnerModule(Module):
        def __init__(self):
            super().__init__()
            self.bn = BatchNorm2d(4)

        def forward(self, x):
            x = self.bn(x)

    def __init__(self):
        super().__init__()
        self.i = self.InnerModule()
        self.bn = BatchNorm2d(4)
        self.param = Parameter(np.ones(1, dtype=np.float32))
        self.buff = Buffer(np.ones(1, dtype=np.float32))

    def forward(self, x):
        x = self.i(x)
        x = self.bn(x)
        return x


def test_module_api():
    m = MyModule()
    assert list(m.children()) == [m.bn, m.i]
    assert list(m.named_children()) == [("bn", m.bn), ("i", m.i)]
    assert list(m.modules()) == [m, m.bn, m.i, m.i.bn]
    assert list(m.named_modules()) == [
        ("", m),
        ("bn", m.bn),
        ("i", m.i),
        ("i.bn", m.i.bn),
    ]
    assert list(m.named_modules(prefix="x")) == [
        ("x", m),
        ("x.bn", m.bn),
        ("x.i", m.i),
        ("x.i.bn", m.i.bn),
    ]
    assert list(m.buffers()) == [
        m.bn.running_mean,
        m.bn.running_var,
        m.buff,
        m.i.bn.running_mean,
        m.i.bn.running_var,
    ]
    assert list(m.buffers(recursive=False)) == [m.buff]
    assert list(m.named_buffers()) == [
        ("bn.running_mean", m.bn.running_mean),
        ("bn.running_var", m.bn.running_var),
        ("buff", m.buff),
        ("i.bn.running_mean", m.i.bn.running_mean),
        ("i.bn.running_var", m.i.bn.running_var),
    ]
    assert list(m.parameters()) == [
        m.bn.bias,
        m.bn.weight,
        m.i.bn.bias,
        m.i.bn.weight,
        m.param,
    ]
    assert list(m.named_parameters()) == [
        ("bn.bias", m.bn.bias),
        ("bn.weight", m.bn.weight),
        ("i.bn.bias", m.i.bn.bias),
        ("i.bn.weight", m.i.bn.weight),
        ("param", m.param),
    ]
    m.eval()
    assert (
        m.training == False
        and m.bn.training == False
        and m.i.training == False
        and m.i.bn.training == False
    )
    m.bn.train()
    assert m.training == False and m.bn.training == True and m.i.bn.training == False
    m.eval()
    m.i.train()
    assert (
        m.training == False
        and m.bn.training == False
        and m.i.training == True
        and m.i.bn.training == True
    )
    m.eval()
    m.train()
    assert m.training == True and m.bn.training == True and m.i.bn.training == True

    def fn(m):
        m.training = False

    m.apply(fn)
    assert m.bn.training == False and m.i.bn.training == False


def test_module_api_reuse_submodule():
    m = MyModule()
    m.h = m.i  # pylint: disable=attribute-defined-outside-init
    assert list(m.modules()) == [m, m.bn, m.i, m.i.bn]
    assert list(m.named_modules()) == [
        ("", m),
        ("bn", m.bn),
        ("h", m.i),
        ("h.bn", m.i.bn),
    ]


def test_module_api_iterable_stability():
    m = MyModule()
    l = list(m.modules())
    for _ in range(100):
        assert list(m.modules()) == l


class MyModule2(Module):
    class InnerModule(Module):
        def __init__(self):
            super().__init__()
            self.bn = BatchNorm2d(4)
            self.test_bool_key = {True: 1, False: 0}

        def forward(self, x):
            x = self.bn(x)

    def __init__(self):
        super().__init__()
        self.bn = BatchNorm2d(4)
        self.a = [
            BatchNorm2d(4),
            {"x": BatchNorm2d(4), "y": [BatchNorm2d(4), self.InnerModule()], "z": 0},
            (self.InnerModule(),),
        ]

    def forward(self, x):
        return x


def test_expand_structure():
    m = MyModule2()
    assert list(m.named_modules()) == [
        ("", m),
        ("a.0", m.a[0]),
        ("a.1.x", m.a[1]["x"]),
        ("a.1.y.0", m.a[1]["y"][0]),
        ("a.1.y.1", m.a[1]["y"][1]),
        ("a.1.y.1.bn", m.a[1]["y"][1].bn),
        ("a.2.0", m.a[2][0]),
        ("a.2.0.bn", m.a[2][0].bn),
        ("bn", m.bn),
    ]


def test_flatten_others():
    def be_others(obj):
        return not isinstance(obj, (Tensor, Module))

    m = MyModule2()
    assert len(list(m._flatten(with_key=True, predicate=be_others))) == 0


def test_flatten_with_parent():
    m = MyModule2()
    assert list(m.named_modules(with_parent=True)) == [
        ("", m, None),
        ("a.0", m.a[0], m),
        ("a.1.x", m.a[1]["x"], m),
        ("a.1.y.0", m.a[1]["y"][0], m),
        ("a.1.y.1", m.a[1]["y"][1], m),
        ("a.1.y.1.bn", m.a[1]["y"][1].bn, m.a[1]["y"][1]),
        ("a.2.0", m.a[2][0], m),
        ("a.2.0.bn", m.a[2][0].bn, m.a[2][0]),
        ("bn", m.bn, m),
    ]
    assert list(m.modules(with_parent=True)) == [
        (m, None),
        (m.a[0], m),
        (m.a[1]["x"], m),
        (m.a[1]["y"][0], m),
        (m.a[1]["y"][1], m),
        (m.a[1]["y"][1].bn, m.a[1]["y"][1]),
        (m.a[2][0], m),
        (m.a[2][0].bn, m.a[2][0]),
        (m.bn, m),
    ]


class MyModule3(Module):
    class InnerModule(Module):
        def __init__(self):
            super().__init__()
            self.bn = BatchNorm2d(4)

        def forward(self, x):
            x = self.bn(x)

    def __init__(self):
        super().__init__()
        self.bn = BatchNorm2d(4)
        self.seq = Sequential(BatchNorm2d(4), self.InnerModule(),)

    def forward(self, x):
        return x


def test_module_api_with_sequential():
    m = MyModule3()
    assert list(m.named_modules()) == [
        ("", m),
        ("bn", m.bn),
        ("seq", m.seq),
        ("seq.0", m.seq[0]),
        ("seq.1", m.seq[1]),
        ("seq.1.bn", m.seq[1].bn),
    ]


def test_sequential_named_children():
    modules = OrderedDict()
    modules["name0"] = Linear(20, 10)
    modules["name1"] = Linear(10, 5)
    modules["name2"] = Linear(5, 1)
    m = Sequential(modules)
    l = list(m.named_children())
    assert l[0][0] == "name0"
    assert l[1][0] == "name1"
    assert l[2][0] == "name2"


def test_state_dict():
    data_shape = (2, 28)
    data = tensor()
    data.set_value(np.random.random(data_shape))
    mlp = MLP()
    pred0 = mlp(data)

    with BytesIO() as fout:
        mge.save(mlp.state_dict(), fout)
        fout.seek(0)
        state_dict = mge.load(fout)
        state_dict["extra"] = None
        mlp1 = MLP()
        mlp1.load_state_dict(state_dict, strict=False)
        pred1 = mlp1(data)
        assertTensorClose(pred0.numpy(), pred1.numpy(), max_err=5e-6)
        with pytest.raises(KeyError):
            mlp1.load_state_dict(state_dict)
        del state_dict["extra"]
        del state_dict["dense0.bias"]
        with pytest.raises(KeyError):
            mlp1.load_state_dict(state_dict)


class AssertModule(Module):
    def __init__(self):
        super().__init__()
        self.error_tensor_key = {True: tensor(), False: 0}

    def forward(self, x):
        return x


def test_assert_message():
    m = AssertModule()
    with pytest.raises(
        AssertionError, match="keys for Tensor and Module must be str, error key: True"
    ):
        list(m._flatten())


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv1 = Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv1.weight = self.conv0.weight

    def forward(self, inputs):
        pass


def test_shared_param():
    net = Simple()
    assert net.conv0.weight is net.conv1.weight
    data = tensor(np.random.random((1, 1, 8, 8)).astype(np.float32))
    assertTensorClose(net.conv0(data).numpy(), net.conv1(data).numpy())
    with BytesIO() as f:
        mge.save(net, f)
        f.seek(0)
        net1 = mge.load(f)
    assert net1.conv0.weight is net1.conv1.weight
    assertTensorClose(net1.conv0(data).numpy(), net1.conv1(data).numpy())

    with BytesIO() as f:
        mge.save(net.conv0, f)
        f.seek(0)
        conv0 = mge.load(f)

    with BytesIO() as f:
        mge.save(net.conv1, f)
        f.seek(0)
        conv1 = mge.load(f)

    assert conv0.weight is not conv1.weight
    assertTensorClose(conv0(data).numpy(), conv1(data).numpy())


def test_pickle_module():
    data_shape = (2, 28)
    data = tensor()
    data.set_value(np.random.random(data_shape))
    mlp = MLP()
    # pickle before forward
    with BytesIO() as fout:
        mge.save(mlp, fout)
        fout.seek(0)
        mlp1 = mge.load(fout)
        pred0 = mlp1(data)

    pred1 = mlp(data)

    # pickle after forward
    with BytesIO() as fout:
        mge.save(mlp, fout)
        fout.seek(0)
        mlp1 = mge.load(fout)
        pred2 = mlp1(data)

    assertTensorClose(pred0.numpy(), pred1.numpy(), max_err=5e-6)
    assertTensorClose(pred0.numpy(), pred2.numpy(), max_err=5e-6)


def test_dump_model():
    data_shape = (2, 28)
    data = tensor()
    data.set_value(np.random.random(data_shape))
    mlp = MLP()
    pred = mlp(data)
    with tempfile.NamedTemporaryFile() as f:
        mge.dump(pred, f.name)


def test_load_quantized():
    data_shape = (2, 28)
    data = tensor(np.random.random(data_shape), dtype="float32")
    data = data.astype(mgb.dtype.qint8(0.1))
    mlp = MLP()
    quantize_qat(mlp)
    quantize(mlp)
    mlp.dense0.weight = Parameter(
        mlp.dense0.weight.astype(mgb.dtype.qint8(0.001)).numpy()
    )
    mlp.dense1.weight = Parameter(
        mlp.dense1.weight.astype(mgb.dtype.qint8(0.0002)).numpy()
    )
    mlp.eval()
    pred0 = mlp(data)

    with BytesIO() as fout:
        mge.save(mlp.state_dict(), fout)
        fout.seek(0)
        checkpoint = mge.load(fout)
        # change mlp weight.
        mlp.dense0.weight = Parameter(
            mlp.dense0.weight.astype(mgb.dtype.qint8(0.00001)).numpy()
        )
        mlp.dense1.weight = Parameter(
            mlp.dense1.weight.astype(mgb.dtype.qint8(0.2)).numpy()
        )
        mlp.load_state_dict(checkpoint)
        pred1 = mlp(data)

    assertTensorClose(
        pred0.astype("float32").numpy(), pred1.astype("float32").numpy(), max_err=5e-6
    )
