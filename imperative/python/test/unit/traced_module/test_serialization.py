import pickle
from collections import defaultdict
from functools import wraps
from tempfile import TemporaryFile

import numpy as np

import megengine.functional as F
import megengine.module as M
import megengine.traced_module.expr as Expr
import megengine.traced_module.serialization as S
from megengine import Tensor
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops import builtin
from megengine.core.ops.builtin import Elemwise
from megengine.module import Module
from megengine.traced_module import trace_module
from megengine.traced_module.expr import CallMethod, Constant
from megengine.traced_module.node import TensorNode
from megengine.traced_module.serialization import (
    register_functional_loader,
    register_module_loader,
    register_opdef_loader,
    register_tensor_method_loader,
)
from megengine.traced_module.utils import _convert_kwargs_to_args


def _check_id(traced_module):
    _total_ids = traced_module.graph._total_ids
    node_ids = [n._id for n in traced_module.graph.nodes().as_list()]
    assert len(set(node_ids)) == len(node_ids)
    assert max(node_ids) + 1 == _total_ids[0]

    expr_ids = [n._id for n in traced_module.graph.exprs().as_list()]
    assert len(set(expr_ids)) == len(expr_ids)
    assert max(expr_ids) + 1 == _total_ids[1]


def _check_name(flatened_module):
    node_names = [n._name for n in flatened_module.graph.nodes().as_list()]
    assert len(set(node_names)) == len(node_names)


def _check_expr_users(traced_module):
    node_user = defaultdict(list)
    for expr in traced_module.graph._exprs:
        for node in expr.inputs:
            node_user[node].append(expr)
        if isinstance(expr, CallMethod) and expr.graph:
            _check_expr_users(expr.inputs[0].owner)

    for node in traced_module.graph.nodes(False):
        node.users.sort(key=lambda m: m._id)
        node_user[node].sort(key=lambda m: m._id)
        assert node.users == node_user[node]


class MyBlock(Module):
    def __init__(self, in_channels, channels):
        super(MyBlock, self).__init__()
        self.conv1 = M.Conv2d(in_channels, channels, 3, 1, padding=1, bias=False)
        self.bn1 = M.BatchNorm2d(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x) + 1
        return x


class MyModule(Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.block0 = MyBlock(8, 4)
        self.block1 = MyBlock(4, 2)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        return x


def test_dump_and_load():
    module = MyModule()
    x = Tensor(np.ones((1, 8, 14, 14)))
    expect = module(x)
    traced_module = trace_module(module, x)
    np.testing.assert_array_equal(expect, traced_module(x))
    obj = pickle.dumps(traced_module)
    new_tm = pickle.loads(obj)
    _check_id(new_tm)
    _check_expr_users(new_tm)
    traced_module.graph._reset_ids()
    old_nodes = traced_module.graph.nodes().as_list()
    new_nodes = new_tm.graph.nodes().as_list()
    old_exprs = traced_module.graph.exprs().as_list()
    new_exprs = new_tm.graph.exprs().as_list()
    assert len(old_nodes) == len(new_nodes)
    for i, j in zip(old_nodes, new_nodes):
        assert i._name == j._name
        assert i._qualname == j._qualname
        assert i._id == j._id
    assert len(old_exprs) == len(new_exprs)
    for i, j in zip(old_exprs, new_exprs):
        assert i._id == j._id

    np.testing.assert_array_equal(expect, traced_module(x))


def test_opdef_loader():
    class MyModule1(Module):
        def forward(self, x, y):
            op = Elemwise("ADD")
            return apply(op, x, y)[0]

    m = MyModule1()
    x = Tensor(np.ones((20)))
    y = Tensor(np.ones((20)))
    traced_module = trace_module(m, x, y)
    orig_loader_dict = S.OPDEF_LOADER
    S.OPDEF_LOADER = {}

    @register_opdef_loader(Elemwise)
    def add_opdef_loader(expr):
        if expr.opdef_state["mode"] == "ADD":
            expr.opdef_state["mode"] = "MUL"
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

    obj = pickle.dumps(traced_module)
    new_module = pickle.loads(obj)
    _check_id(new_module)
    _check_expr_users(new_module)
    _check_name(new_module.flatten())
    assert (
        isinstance(new_module.graph._exprs[0], CallMethod)
        and new_module.graph._exprs[1].opdef.mode == "MUL"
        and len(new_module.graph._exprs) == 2
    )
    result = new_module(x, y)
    np.testing.assert_equal(result.numpy(), x.numpy())
    S.OPDEF_LOADER = orig_loader_dict


def test_functional_loader():
    class MyModule2(Module):
        def forward(self, x, y):
            return F.conv2d(x, y)

    m = MyModule2()
    x = Tensor(np.random.random((1, 3, 32, 32)))
    y = Tensor(np.random.random((3, 3, 3, 3)))
    traced_module = trace_module(m, x, y)
    orig_loader_dict = S.FUNCTIONAL_LOADER
    S.FUNCTIONAL_LOADER = {}

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

    obj = pickle.dumps(traced_module)
    new_module = pickle.loads(obj)
    _check_expr_users(new_module)
    _check_id(new_module)
    result = new_module(x, y)
    gt = m(x, y)
    assert (
        isinstance(new_module.graph._exprs[0], CallMethod)
        and len(new_module.graph._exprs) == 2
    )
    np.testing.assert_equal(result.numpy(), gt.numpy())
    S.FUNCTIONAL_LOADER = orig_loader_dict


def test_tensor_method_loader():
    class MyModule3(Module):
        def forward(self, x):
            return x + 1

    m = MyModule3()
    x = Tensor(np.ones((20)))
    traced_module = trace_module(m, x)
    orig_loader_dict = S.TENSORMETHOD_LOADER
    S.TENSORMETHOD_LOADER = {}

    @register_tensor_method_loader("__add__")
    def add_loader(expr):
        args = list(expr.args)
        if not isinstance(args[1], TensorNode):
            args[1] = Tensor(args[1])
            node = Constant(args[1], "const").outputs[0]

            astype_expr = CallMethod(node, "astype")
            oup = TensorNode(
                astype_expr, shape=node.shape, dtype=node.dtype, qparams=node.qparams,
            )
            astype_expr.set_args_kwargs(node, expr.inputs[0].dtype)
            astype_expr.return_val = (oup,)

            add_expr = CallMethod(oup, "__add__")
            add_expr.set_args_kwargs(oup, oup)
            oup1 = TensorNode(
                add_expr, shape=oup.shape, dtype=oup.dtype, qparams=node.qparams,
            )
            add_expr.return_val = oup1
            args[1] = oup1
            expr.set_args_kwargs(*args)

    obj = pickle.dumps(traced_module)
    new_module = pickle.loads(obj)
    _check_expr_users(new_module)
    _check_id(new_module)
    result = new_module(x)
    gt = m(x)
    assert (
        isinstance(new_module.graph._exprs[0], Constant)
        and len(new_module.graph._exprs) == 4
    )
    np.testing.assert_equal(result.numpy(), (x + 2).numpy())
    S.TENSORMETHOD_LOADER = orig_loader_dict


def test_module_loader():
    class MyModule4(Module):
        def __init__(self):
            super().__init__()
            self.conv = M.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.conv(x)

    m = MyModule4()
    x = Tensor(np.random.random((1, 3, 32, 32)))
    traced_module = trace_module(m, x)
    orig_loader_dict = S.MODULE_LOADER
    S.MODULE_LOADER = {}

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

    obj = pickle.dumps(traced_module)
    new_module = pickle.loads(obj)
    result = new_module(x)
    gt = m(x)
    assert (
        isinstance(new_module.graph._exprs[1], CallMethod)
        and len(new_module.graph._exprs) == 3
    )
    np.testing.assert_equal(result.numpy(), gt.numpy())
    S.MODULE_LOADER = orig_loader_dict


def test_shared_module():
    class MyModule(M.Module):
        def __init__(self):
            super().__init__()
            self.a = M.Elemwise("ADD")
            self.b = self.a

        def forward(self, x, y):
            z = self.a(x, y)
            z = self.b(z, y)
            return z

    x = Tensor(1)
    y = Tensor(2)
    m = MyModule()
    tm = trace_module(m, x, y)
    obj = pickle.dumps(tm)
    load_tm = pickle.loads(obj)
    _check_expr_users(load_tm)
    _check_name(load_tm.flatten())
    _check_id(load_tm)
    assert load_tm.a is load_tm.b


def test_convert_kwargs_to_args():
    def func(a, b, c=4, *, d, e=3, f=4):
        pass

    args = (1,)
    kwargs = {"b": 1, "d": 6}
    new_args, new_kwargs = _convert_kwargs_to_args(func, args, kwargs)
    assert new_args == (1, 1, 4)
    assert new_kwargs == {"d": 6, "e": 3, "f": 4}

    args = (1,)
    kwargs = {"d": 6}
    new_args, new_kwargs = _convert_kwargs_to_args(func, args, kwargs, is_bounded=True)
    assert new_args == (1, 4)
    assert new_kwargs == {"d": 6, "e": 3, "f": 4}

    def func1(a, b, c, d, e, *, f):
        pass

    args = ()
    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}
    new_args, new_kwargs = _convert_kwargs_to_args(func1, args, kwargs)
    assert new_args == (1, 2, 3, 4, 5)
    assert new_kwargs == {"f": 6}


def test_opdef_serialization():
    with TemporaryFile() as f:
        x = builtin.Elemwise(mode="Add")
        pickle.dump(x, f)
        f.seek(0)
        load_x = pickle.load(f)
        assert x == load_x

    with TemporaryFile() as f:
        x = builtin.Convolution(stride_h=9, compute_mode="float32")
        x.strategy = (
            builtin.Convolution.Strategy.PROFILE
            | builtin.Convolution.Strategy.HEURISTIC
            | builtin.Convolution.Strategy.REPRODUCIBLE
        )
        pickle.dump(x, f)
        f.seek(0)
        load_x = pickle.load(f)
        assert x.strategy == load_x.strategy
        assert x == load_x


def test_square_function_compat():
    @wraps(F.elemwise.square)
    def origin_square(x):
        return F.pow(x, 2)

    new_square = F.elemwise.square
    F.elemwise.square = origin_square
    current_version = Expr.__version__
    Expr.__version__ = "1.11.1"

    class old_square(M.Module):
        def forward(self, x):
            x = F.relu(x)
            x = F.elemwise.square(x)
            return x * 2

    m = trace_module(old_square(), Tensor([1, 2, 4, 6]))
    float_m = trace_module(old_square(), Tensor([1.0, 2.0, 4.0, 6.0]))

    # dump old version square
    obj = pickle.dumps(m)
    f_obj = pickle.dumps(float_m)

    # load in new version
    F.elemwise.square = new_square
    Expr.__version__ = current_version
    new_m = pickle.loads(obj)
    new_float_m = pickle.loads(f_obj)
    assert len(new_m.graph._exprs) == 4 and len(new_float_m.graph._exprs) == 3
    assert new_m(Tensor([1, 2, 4, 6])).dtype == np.float32
