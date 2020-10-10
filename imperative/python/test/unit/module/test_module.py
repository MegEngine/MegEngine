# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import os
import tempfile
from collections import OrderedDict
from io import BytesIO

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine import Parameter, Tensor, tensor
from megengine.module import (
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    Sequential,
    Softmax,
)
from megengine.quantization.quantize import quantize, quantize_qat


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
    np.testing.assert_allclose(x.numpy(), y, rtol=1e-6)


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
    compare_fn: the function to compare the result and expected, use np.testing.assert_allclose if None.
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


class MyModule(Module):
    class InnerModule(Module):
        def __init__(self):
            super().__init__()
            self.bn = BatchNorm2d(4)

        def forward(self, x):
            return self.bn(x)

    def __init__(self):
        super().__init__()
        self.i = self.InnerModule()
        self.bn = BatchNorm2d(4)
        self.param = Parameter(np.ones(1, dtype=np.float32))
        self.buff = Tensor(np.ones(1, dtype=np.float32))

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


def test_module_api_hooks():
    net = MyModule()
    pre_hook_num = 0
    post_hook_num = 0
    hooks = []

    def pre_hook(module, inputs):
        nonlocal pre_hook_num
        pre_hook_num += 1
        modified_inputs = tuple(inp + 1 for inp in inputs)
        return modified_inputs

    def post_hook(module, inputs, outputs):
        nonlocal post_hook_num
        post_hook_num += 1
        outputs += 1
        return outputs

    net.apply(lambda module: hooks.append(module.register_forward_pre_hook(pre_hook)))
    net.apply(lambda module: hooks.append(module.register_forward_hook(post_hook)))

    shape = (1, 4, 1, 1)
    x = tensor(np.zeros(shape, dtype=np.float32))
    y = net(x)

    assert pre_hook_num == 4
    assert post_hook_num == 4
    mean1 = Parameter(np.zeros(shape), dtype=np.float32)
    bn1 = F.batch_norm(
        x + 3, mean1, Parameter(np.ones(shape), dtype=np.float32), training=True
    )
    np.testing.assert_allclose(
        net.i.bn.running_mean.numpy(), mean1.numpy(),
    )
    mean2 = Parameter(np.zeros(shape), dtype=np.float32)
    bn2 = F.batch_norm(
        bn1 + 3, mean2, Parameter(np.ones(shape), dtype=np.float32), training=True
    )
    np.testing.assert_allclose(
        net.bn.running_mean.numpy(), mean2.numpy(),
    )
    np.testing.assert_allclose((bn2 + 2).numpy(), y.numpy())

    assert len(hooks) == 8
    for handler in hooks:
        handler.remove()
    y = net(x)
    assert pre_hook_num == 4
    assert post_hook_num == 4


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
    data = tensor(np.random.random(data_shape))
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
        np.testing.assert_allclose(pred0.numpy(), pred1.numpy(), atol=5e-6)
        with pytest.raises(KeyError):
            mlp1.load_state_dict(state_dict)
        del state_dict["extra"]
        del state_dict["dense0.bias"]
        with pytest.raises(KeyError):
            mlp1.load_state_dict(state_dict)


class AssertModule(Module):
    def __init__(self):
        super().__init__()
        self.error_tensor_key = {True: tensor([]), False: 0}

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
    np.testing.assert_allclose(net.conv0(data).numpy(), net.conv1(data).numpy())
    with BytesIO() as f:
        mge.save(net, f)
        f.seek(0)
        net1 = mge.load(f)
    assert net1.conv0.weight is net1.conv1.weight
    np.testing.assert_allclose(net1.conv0(data).numpy(), net1.conv1(data).numpy())

    with BytesIO() as f:
        mge.save(net.conv0, f)
        f.seek(0)
        conv0 = mge.load(f)

    with BytesIO() as f:
        mge.save(net.conv1, f)
        f.seek(0)
        conv1 = mge.load(f)

    assert conv0.weight is not conv1.weight
    np.testing.assert_allclose(conv0(data).numpy(), conv1(data).numpy())


def test_pickle_module():
    data_shape = (2, 28)
    data = tensor(np.random.random(data_shape))
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

    np.testing.assert_allclose(pred0.numpy(), pred1.numpy(), atol=5e-6)
    np.testing.assert_allclose(pred0.numpy(), pred2.numpy(), atol=5e-6)


@pytest.mark.skip(reason="under development")
def test_dump_model():
    data_shape = (2, 28)
    data = Tensor(np.random.random(data_shape))
    mlp = MLP()
    pred = mlp(data)
    f = tempfile.NamedTemporaryFile(delete=False)
    f_name = f.name
    try:
        mge.dump(pred, f_name)
    finally:
        f.close()
        os.unlink(f_name)


def test_load_quantized():
    from megengine.core.tensor import dtype

    data_shape = (2, 28)
    data = tensor(np.random.random(data_shape), dtype="float32")
    data = data.astype(dtype.qint8(0.1))
    mlp = MLP()
    quantize_qat(mlp)
    quantize(mlp)
    mlp.dense0.weight = Parameter(mlp.dense0.weight.astype(dtype.qint8(0.001)).numpy())
    mlp.dense1.weight = Parameter(mlp.dense1.weight.astype(dtype.qint8(0.0002)).numpy())
    mlp.eval()
    pred0 = mlp(data)

    with BytesIO() as fout:
        mge.save(mlp.state_dict(), fout)
        fout.seek(0)
        checkpoint = mge.load(fout)
        # change mlp weight.
        mlp.dense0.weight = Parameter(
            mlp.dense0.weight.astype(dtype.qint8(0.00001)).numpy()
        )
        mlp.dense1.weight = Parameter(
            mlp.dense1.weight.astype(dtype.qint8(0.2)).numpy()
        )
        mlp.load_state_dict(checkpoint)
        pred1 = mlp(data)

    np.testing.assert_allclose(
        pred0.astype("float32").numpy(), pred1.astype("float32").numpy(), atol=5e-6
    )


def test_repr_basic():
    # test whether __repr__ can output correct information
    class ConvModel(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(3, 128, 3, stride=2, bias=False)
            self.conv2 = Conv2d(3, 128, 3, padding=1, bias=False)
            self.conv3 = Conv2d(3, 128, 3, dilation=2, bias=False)
            self.bn1 = BatchNorm2d(128)
            self.bn2 = BatchNorm1d(128)
            self.dropout = Dropout(drop_prob=0.1)
            self.softmax = Softmax(axis=100)
            self.pooling = MaxPool2d(kernel_size=2, padding=0)
            self.submodule1 = Sequential(Dropout(drop_prob=0.1), Softmax(axis=100),)
            self.fc1 = Linear(512, 1024)

        def forward(self, inputs):
            pass

    ground_truth = (
        "ConvModel(\n"
        "  (conv1): Conv2d(3, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)\n"
        "  (conv2): Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1), bias=False)\n"
        "  (conv3): Conv2d(3, 128, kernel_size=(3, 3), dilation=(2, 2), bias=False)\n"
        "  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)\n"
        "  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)\n"
        "  (dropout): Dropout(drop_prob=0.1)\n  (softmax): Softmax(axis=100)\n"
        "  (pooling): MaxPool2d(kernel_size=2, stride=2, padding=0)\n"
        "  (submodule1): Sequential(\n"
        "    (0): Dropout(drop_prob=0.1)\n"
        "    (1): Softmax(axis=100)\n  )\n"
        "  (fc1): Linear(in_features=512, out_features=1024, bias=True)\n"
        ")"
    )
    net = ConvModel()
    output = net.__repr__()
    assert output == ground_truth


def test_repr_module_reassign():
    # test whether __repr__ can deal with module reassign
    class ConvModel1(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(3, 128, 3, bias=False)
            self.conv2 = Conv2d(3, 128, 3, padding=1, bias=False)
            self.conv1 = Conv2d(3, 256, 3, dilation=2, bias=False)

        def forward(self, inputs):
            pass

    ground_truth = (
        "ConvModel1(\n"
        "  (conv1): Conv2d(3, 256, kernel_size=(3, 3), dilation=(2, 2), bias=False)\n"
        "  (conv2): Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1), bias=False)\n"
        ")"
    )
    net = ConvModel1()
    output = net.__repr__()
    assert output == ground_truth


def test_repr_module_rereference():
    # test whether __repr__ can deal with module re-reference
    class ConvModel2(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(3, 128, 3, bias=False)
            self.conv2 = self.conv1
            self.conv3 = self.conv1

        def forward(self, inputs):
            pass

    ground_truth = (
        "ConvModel2(\n"
        "  (conv1): Conv2d(3, 128, kernel_size=(3, 3), bias=False)\n"
        "  (conv2): Conv2d(3, 128, kernel_size=(3, 3), bias=False)\n"
        "  (conv3): Conv2d(3, 128, kernel_size=(3, 3), bias=False)\n"
        ")"
    )
    net = ConvModel2()
    output = net.__repr__()
    assert output == ground_truth


def test_repr_module_delete():
    # test whether __repr__ can deal with module delete
    class ConvModel3(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(3, 128, 3, bias=False)
            self.softmax = Softmax(100)

        def forward(self, inputs):
            pass

    ground_truth = (
        "ConvModel3(\n"
        "  (conv1): Conv2d(3, 128, kernel_size=(3, 3), bias=False)\n"
        ")"
    )
    net = ConvModel3()
    del net.softmax
    output = net.__repr__()
    assert output == ground_truth
