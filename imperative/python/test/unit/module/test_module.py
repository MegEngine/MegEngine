# -*- coding: utf-8 -*-
from collections import OrderedDict
from io import BytesIO

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
from megengine import Parameter, Tensor, tensor
from megengine.device import get_device_count
from megengine.module import (
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    GroupNorm,
    InstanceNorm,
    Linear,
    MaxPool2d,
    Module,
    Sequential,
    Softmax,
)
from megengine.module.module import _access_structure
from megengine.quantization.quantize import quantize, quantize_qat
from megengine.traced_module import TracedModule, trace_module
from megengine.utils.module_utils import get_expand_structure, set_expand_structure


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


@pytest.mark.parametrize("test_traced_module", [True, False])
def test_module_api(test_traced_module):
    m = MyModule()
    if test_traced_module:
        buff = m.buff
        param = m.param
        m = trace_module(m, Tensor(np.random.random((1, 4, 16, 16))))
        assert "buff" not in m.__dict__
        assert "param" not in m.__dict__
        m.buff = buff
        m.param = param

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
    assert list(m.tensors()) == [
        m.bn.bias,
        m.bn.running_mean,
        m.bn.running_var,
        m.bn.weight,
        m.buff,
        m.i.bn.bias,
        m.i.bn.running_mean,
        m.i.bn.running_var,
        m.i.bn.weight,
        m.param,
    ]
    assert list(m.named_tensors()) == [
        ("bn.bias", m.bn.bias),
        ("bn.running_mean", m.bn.running_mean),
        ("bn.running_var", m.bn.running_var),
        ("bn.weight", m.bn.weight),
        ("buff", m.buff),
        ("i.bn.bias", m.i.bn.bias),
        ("i.bn.running_mean", m.i.bn.running_mean),
        ("i.bn.running_var", m.i.bn.running_var),
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


@pytest.mark.parametrize("test_traced_module", [True, False])
def test_module_api_reuse_submodule(test_traced_module):
    m = MyModule()
    if test_traced_module:
        m = trace_module(m, Tensor(np.random.random((1, 4, 16, 16))))
    m.h = m.i  # pylint: disable=attribute-defined-outside-init
    assert list(m.modules()) == [m, m.bn, m.i, m.i.bn]
    assert list(m.named_modules()) == [
        ("", m),
        ("bn", m.bn),
        ("h", m.i),
        ("h.bn", m.i.bn),
    ]


@pytest.mark.parametrize("test_traced_module", [True, False])
def test_module_api_iterable_stability(test_traced_module):
    m = MyModule()
    if test_traced_module:
        m = trace_module(m, Tensor(np.random.random((1, 4, 16, 16))))
    l = list(m.modules())
    for _ in range(100):
        assert list(m.modules()) == l


@pytest.mark.parametrize("test_traced_module", [True, False])
def test_module_api_hooks(test_traced_module):
    net = MyModule()
    if test_traced_module:
        net = trace_module(net, Tensor(np.zeros((1, 4, 1, 1))))
    pre_hook_num = 0
    post_hook_num = 0
    hooks = []

    def pre_hook(_, inputs):
        nonlocal pre_hook_num
        pre_hook_num += 1
        modified_inputs = tuple(inp + 1 for inp in inputs)
        return modified_inputs

    def post_hook(_, __, outputs):
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
    rst = [
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
    assert list(m.named_modules()) == rst

    for item in rst[1:]:
        assert get_expand_structure(m, item[0]) == item[1]

    for item in reversed(rst[1:]):
        if _access_structure(m, item[0], lambda p, k, o: isinstance(p, tuple)):
            continue
        set_expand_structure(m, item[0], "TEST_VALUE")
        assert get_expand_structure(m, item[0]) == "TEST_VALUE"


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
    with pytest.raises(
        AssertionError, match="keys for Tensor and Module must be str, error key: True"
    ):
        m = AssertModule()
        list(m._flatten())


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv1 = Conv2d(1, 1, kernel_size=3, bias=False)
        self.conv1.weight = self.conv0.weight

    def forward(self, inputs):
        x = self.conv0(inputs)
        y = self.conv1(inputs)
        return x + y


@pytest.mark.parametrize("test_traced_module", [True, False])
def test_shared_param(test_traced_module):
    net = Simple()
    if test_traced_module:
        net = trace_module(net, tensor(np.random.random((1, 1, 8, 8))))
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


class Simple2(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(1, 1, kernel_size=3, bias=False)
        self.conv0 = Conv1d(1, 1, kernel_size=3, bias=False)
        self.conv1.weight = self.conv0.weight

    def forward(self, inputs):
        pass


def test_shared_param_1d():
    net = Simple2()
    assert net.conv0.weight is net.conv1.weight
    data = tensor(np.random.random((1, 1, 8)).astype(np.float32))
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


@pytest.mark.parametrize("test_traced_module", [True, False])
def test_pickle_module(test_traced_module):
    data_shape = (2, 28)
    data = tensor(np.random.random(data_shape))
    mlp = MLP()
    pred_gt = mlp(data)
    if test_traced_module:
        mlp = trace_module(mlp, data)
    # pickle before forward
    with BytesIO() as fout:
        mge.save(mlp, fout)
        fout.seek(0)
        mlp1 = mge.load(fout)
        if test_traced_module:
            assert type(mlp1) == TracedModule
        pred0 = mlp1(data)

    pred1 = mlp(data)

    # pickle after forward
    with BytesIO() as fout:
        mge.save(mlp, fout)
        fout.seek(0)
        mlp1 = mge.load(fout)
        if test_traced_module:
            assert type(mlp1) == TracedModule
        pred2 = mlp1(data)

    np.testing.assert_allclose(pred_gt.numpy(), pred1.numpy(), atol=5e-6)
    np.testing.assert_allclose(pred0.numpy(), pred1.numpy(), atol=5e-6)
    np.testing.assert_allclose(pred0.numpy(), pred2.numpy(), atol=5e-6)


def test_repr_basic():
    # test whether __repr__ can output correct information
    class ConvModel(Module):
        def __init__(self):
            super().__init__()
            self.conv1 = Conv2d(3, 128, 3, padding=1, bias=False)
            self.conv2 = Conv2d(3, 128, 3, dilation=2, bias=False)
            self.bn1 = BatchNorm1d(128)
            self.bn2 = BatchNorm2d(128)
            self.pooling = MaxPool2d(kernel_size=2, padding=0)
            modules = OrderedDict()
            modules["depthwise"] = Conv2d(256, 256, 3, 1, 0, groups=256, bias=False,)
            modules["pointwise"] = Conv2d(
                256, 256, kernel_size=1, stride=1, padding=0, bias=True,
            )
            self.submodule1 = Sequential(modules)
            self.list1 = [Dropout(drop_prob=0.1), [Softmax(axis=100)]]
            self.tuple1 = (
                Dropout(drop_prob=0.1),
                (Softmax(axis=100), Dropout(drop_prob=0.2)),
            )
            self.dict1 = {"Dropout": Dropout(drop_prob=0.1)}
            self.fc1 = Linear(512, 1024)

        def forward(self, inputs):
            pass

    ground_truth = (
        "ConvModel(\n"
        "  (conv1): Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1), bias=False)\n"
        "  (conv2): Conv2d(3, 128, kernel_size=(3, 3), dilation=(2, 2), bias=False)\n"
        "  (bn1): BatchNorm1d(128, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)\n"
        "  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.9, affine=True, track_running_stats=True)\n"
        "  (pooling): MaxPool2d(kernel_size=2, stride=2, padding=0)\n"
        "  (submodule1): Sequential(\n"
        "    (depthwise): Conv2d(256, 256, kernel_size=(3, 3), groups=256, bias=False)\n"
        "    (pointwise): Conv2d(256, 256, kernel_size=(1, 1))\n"
        "  )\n"
        "  (list1.0): Dropout(drop_prob=0.1)\n"
        "  (list1.1.0): Softmax(axis=100)\n"
        "  (tuple1.0): Dropout(drop_prob=0.1)\n"
        "  (tuple1.1.0): Softmax(axis=100)\n"
        "  (tuple1.1.1): Dropout(drop_prob=0.2)\n"
        "  (dict1.Dropout): Dropout(drop_prob=0.1)\n"
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


def test_repr_module_reset_attr():
    class ResetAttrModule(Module):
        def __init__(self, flag):
            super().__init__()
            if flag:
                self.a = None
                self.a = Linear(3, 5)
            else:
                self.a = Linear(3, 5)
                self.a = None

        def forward(self, x):
            if self.a:
                x = self.a(x)
            return x

    ground_truth = [
        (
            "ResetAttrModule(\n"
            "  (a): Linear(in_features=3, out_features=5, bias=True)\n"
            ")"
        ),
        ("ResetAttrModule()"),
    ]

    m0 = ResetAttrModule(True)
    m1 = ResetAttrModule(False)
    output = [m0.__repr__(), m1.__repr__()]
    assert output == ground_truth


def test_module_compatible():
    class Empty(Module):
        def forward(self):
            pass

    empty_module = Empty()
    old_attributes = set(
        [
            "_modules",
            "name",
            "training",
            "quantize_disabled",
            "_forward_pre_hooks",
            "_forward_hooks",
            "_name",
            "_short_name",
        ]
    )
    current_attributes = set(empty_module.__dict__.keys())
    assert (
        old_attributes == current_attributes
    ), "Add or delete attributes in Module class may break compatibility of pickle serialization"


@pytest.mark.skip(reason="pytest aborted")
@pytest.mark.parametrize("affine", [True, False])
def test_grou_norm(affine):
    num_groups = 256
    num_channels = 256
    weight_np = np.random.uniform(-0.5, 0.5, (num_channels))
    bias_np = np.random.uniform(-0.5, 0.5, (num_channels))

    class OriginGroupNormFunc(Module):
        def __init__(self, eps=1e-5, affine=True, **kwargs):
            super().__init__(**kwargs)
            assert num_channels % num_groups == 0
            self.num_groups = num_groups
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = Parameter(weight_np)
                self.bias = Parameter(bias_np)
            else:
                self.weight = None
                self.bias = None

        def forward(self, x):
            N, C, H, W = x.shape
            x = x.reshape(N, self.num_groups, -1)
            mean = x.mean(axis=2, keepdims=True)
            var = (x * x).mean(axis=2, keepdims=True) - mean * mean
            x = (x - mean) / F.sqrt(var + self.eps)
            x = x.reshape(N, C, H, W)
            if self.affine:
                x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(
                    1, -1, 1, 1
                )
            return x

    inp = np.random.uniform(-0.5, 0.5, (2, num_channels, 10, 16)).astype("float32")
    mge_inp = Tensor(inp)
    mge_m = GroupNorm(num_groups, num_channels, affine=affine)
    mge_m.weight = Parameter(weight_np)
    mge_m.bias = Parameter(bias_np)
    ori_inp = Tensor(inp)
    ori_m = OriginGroupNormFunc(affine=affine)

    mge_gm = mge.autodiff.GradManager().attach((*mge_m.parameters(), mge_inp))
    ori_gm = mge.autodiff.GradManager().attach((*ori_m.parameters(), ori_inp))
    dy = Tensor(np.random.uniform(-0.5, 0.5, inp.shape))
    for i in range(2):
        with mge_gm:
            mge_output = mge_m(mge_inp)

            mge_gm.backward(mge_output, dy)

        with ori_gm:
            ori_output = ori_m(ori_inp)

            ori_gm.backward(ori_output, dy)

        np.testing.assert_allclose(mge_output.numpy(), ori_output.numpy(), atol=1e-05)
        np.testing.assert_allclose(
            ori_inp.grad.numpy(), mge_inp.grad.numpy(), atol=1e-05
        )
        if affine == True:
            np.testing.assert_allclose(
                mge_m.weight.grad.numpy(), ori_m.weight.grad.numpy(), atol=1e-05
            )
            np.testing.assert_allclose(
                mge_m.bias.grad.numpy(), ori_m.bias.grad.numpy(), atol=1e-05
            )


@pytest.mark.parametrize("affine", [True, False])
def test_instance_norm(affine):
    num_channels = 4
    weight_np = np.random.uniform(-0.5, 0.5, (num_channels))
    bias_np = np.random.uniform(-0.5, 0.5, (num_channels))

    class OriginInstanceNormFunc(Module):
        def __init__(self, eps=1e-5, affine=True, **kwargs):
            super().__init__(**kwargs)
            self.num_channels = num_channels
            self.eps = eps
            self.affine = affine
            if self.affine:
                self.weight = Parameter(weight_np)
                self.bias = Parameter(bias_np)
            else:
                self.weight = None
                self.bias = None

        def forward(self, x):
            N, C, H, W = x.shape
            x = x.reshape(N, self.num_channels, -1)
            mean = x.mean(axis=2, keepdims=True)
            var = (x * x).mean(axis=2, keepdims=True) - mean * mean
            x = (x - mean) / F.sqrt(var + self.eps)
            x = x.reshape(N, C, H, W)
            if self.affine:
                x = self.weight.reshape(1, -1, 1, 1) * x + self.bias.reshape(
                    1, -1, 1, 1
                )
            return x

    inp = np.random.uniform(-0.5, 0.5, (2, num_channels, 10, 16)).astype("float32")
    mge_inp = Tensor(inp)
    mge_m = InstanceNorm(num_channels, affine=affine)
    mge_m.weight = Parameter(weight_np)
    mge_m.bias = Parameter(bias_np)

    ori_inp = Tensor(inp)
    ori_m = OriginInstanceNormFunc(affine=affine)

    mge_im = mge.autodiff.GradManager().attach((*mge_m.parameters(), mge_inp))
    ori_im = mge.autodiff.GradManager().attach((*ori_m.parameters(), ori_inp))
    dy = Tensor(np.random.uniform(-0.5, 0.5, inp.shape))

    for i in range(2):
        with mge_im:
            mge_output = mge_m(mge_inp)

            mge_im.backward(mge_output, dy)

        with ori_im:
            ori_output = ori_m(ori_inp)

            ori_im.backward(ori_output, dy)

        np.testing.assert_allclose(mge_output.numpy(), ori_output.numpy(), atol=1e-05)
        np.testing.assert_allclose(
            ori_inp.grad.numpy(), mge_inp.grad.numpy(), atol=1e-04
        )
        if affine == True:
            np.testing.assert_allclose(
                mge_m.weight.grad.numpy(), ori_m.weight.grad.numpy(), atol=1e-04
            )
            np.testing.assert_allclose(
                mge_m.bias.grad.numpy(), ori_m.bias.grad.numpy(), atol=1e-04
            )
