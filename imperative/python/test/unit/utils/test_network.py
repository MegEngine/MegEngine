import io

import numpy as np

import megengine.core.tensor.megbrain_graph as G
import megengine.functional as F
import megengine.module as M
import megengine.utils.network_node as N
from megengine.jit.tracing import trace
from megengine.tensor import Tensor
from megengine.utils.comp_graph_tools import GraphInference
from megengine.utils.network import Network as Net
from megengine.utils.network import as_oprnode
from megengine.utils.network_node import Host2DeviceCopy, VarNode


def test_replace_var():

    a = Tensor([1, 2])
    b = Tensor([3, 4])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(a, b):
        return (a + b) * 2

    fwd(a, b)
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model, arg_names=["a", "b"], output_names="o", optimize_for_inference=False
    )
    orig_model.seek(0)

    graph = Net.load(orig_model)
    vara = graph.var_filter.name("a").as_unique()
    varb = graph.var_filter.name("b").as_unique()

    out = F.mul(vara, varb)
    out = F.relu(out)

    opnode = list(graph.opr_filter.has_input(vara))
    repl_dict = {opnode[0].outputs[0]: out}
    graph.replace_vars(repl_dict)

    modified_model = io.BytesIO()
    graph.dump(modified_model)
    modified_model.seek(0)
    load_graph = GraphInference(modified_model)

    out = load_graph.run(a, b)
    np.testing.assert_equal(out["o"], [6, 16])


def test_replace_opr():

    a = Tensor([1, 2])
    b = Tensor([3, 4])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(a, b):
        return (a + b) * 2

    fwd(a, b)
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model, arg_names=["a", "b"], output_names="o", optimize_for_inference=False
    )
    orig_model.seek(0)

    graph = Net.load(orig_model)
    vara = graph.var_filter.name("a").as_unique()
    varb = graph.var_filter.name("b").as_unique()

    out1 = F.sub(vara, varb)
    out1 = F.relu(out1)
    out1 = graph.add_dep_oprs(out1)
    orig_opr = graph.opr_filter.has_input(vara).as_unique()

    repl_dict = {orig_opr: out1[0].owner}
    graph.replace_oprs(repl_dict)
    modified_model1 = io.BytesIO()
    graph.dump(modified_model1)
    modified_model1.seek(0)

    load_graph = GraphInference(modified_model1)
    out = load_graph.run(a, b)
    np.testing.assert_equal(out["o"], [0, 0])


def test_modify_params():

    a = Tensor([1, 2])
    b = Tensor([3, 4])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(a, b):
        return (a + b) * 2

    fwd(a, b)
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model, arg_names=["a", "b"], output_names="o", optimize_for_inference=False
    )
    orig_model.seek(0)

    graph = Net.load(orig_model)
    param_const = graph.params_filter.as_unique()
    param_const.set_value(3)

    modified_model = io.BytesIO()
    graph.dump(modified_model)
    modified_model.seek(0)
    load_graph = GraphInference(modified_model)

    out = load_graph.run(a, b)
    np.testing.assert_equal(out["o"], [12, 18])


def test_make_const():

    a = Tensor([1, 2])
    b = Tensor([3, 4])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(a, b):
        return (a + b) * 2

    fwd(a, b)
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model, arg_names=["a", "b"], output_names="o", optimize_for_inference=False
    )
    orig_model.seek(0)

    graph = Net.load(orig_model)
    const_b = graph.make_const(np.array([0.0, 0.0]), name="b")
    varb = graph.var_filter.name("b").as_unique()

    repl_dict = {varb: const_b}
    graph.replace_vars(repl_dict)

    modified_model = io.BytesIO()
    graph.dump(modified_model)
    modified_model.seek(0)
    load_graph = GraphInference(modified_model)

    out = load_graph.run(a)
    np.testing.assert_equal(out["o"], [2, 4])


def test_add_input():

    a = Tensor([1, 2])
    b = Tensor([3, 4])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(a, b):
        return (a + b) * 2

    fwd(a, b)
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model, arg_names=["a", "b"], output_names="o", optimize_for_inference=False
    )
    orig_model.seek(0)

    graph = Net.load(orig_model)
    inp_c = graph.make_input_node((2,), np.int32, name="c")
    varo = graph.var_filter.name("o").as_unique()

    out = F.add(varo, inp_c)
    out.name = "o1"
    graph.remove_output(varo)
    graph.add_output(out)
    modified_model = io.BytesIO()

    graph.dump(modified_model)
    modified_model.seek(0)
    load_graph = GraphInference(modified_model)

    out = load_graph.run(a, b, a)
    np.testing.assert_equal(out["o1"], ((a + b) * 2 + a).numpy())


def test_add_output():

    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])

    @trace(symbolic=True, capture_as_const=True)
    def fwd(a, b):
        return (a + b) * 2

    fwd(a, b)
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model, arg_names=["a", "b"], output_names="o", optimize_for_inference=False
    )
    orig_model.seek(0)

    net = Net.load(orig_model)
    var_a = net.var_filter.name("a").as_unique()
    var_b = net.var_filter.name("b").as_unique()

    y = F.add(var_a, var_b)
    y = F.sigmoid(y)

    y.name = "o1"
    net.add_output(y)

    modified_model = io.BytesIO()
    net.dump(modified_model)
    modified_model.seek(0)

    g = GraphInference(modified_model)
    out = g.run(a.numpy(), b.numpy())

    np.testing.assert_equal(out["o"], ((a + b) * 2).numpy())
    np.testing.assert_equal(out["o1"], (F.sigmoid((a + b))).numpy())


def test_query():
    class Model(M.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = M.Conv2d(3, 32, 3)
            self.conv2 = M.Conv2d(32, 32, 3)
            self.conv3 = M.Conv2d(32, 32, 3)

        def forward(self, data):
            x = self.conv1(data)
            x = self.conv2(x)
            x = self.conv3(x)
            return x

    n = Model()

    @trace(symbolic=True, capture_as_const=True)
    def fwd(data):
        return n(data)

    fwd(Tensor(np.random.random((1, 3, 224, 224))))
    orig_model = io.BytesIO()
    fwd.dump(
        orig_model,
        arg_names=["data"],
        output_names="o",
        keep_opr_name=True,
        keep_var_name=True,
        optimize_for_inference=False,
    )
    orig_model.seek(0)

    graph = Net.load(orig_model)

    r = graph.data_providers_filter.as_count()
    assert r == 1

    opr = graph.get_opr_by_type(Host2DeviceCopy)
    assert isinstance(opr, Host2DeviceCopy)

    r1 = graph.params_filter.as_count()
    assert r1 == 6

    r2 = graph.opr_filter.type(N.ConvolutionForward).as_count()
    assert r2 == 3

    r3 = graph.opr_filter.not_type(N.ConvolutionForward).as_count()
    assert r3 == len(graph.all_oprs) - r2

    var = graph.var_filter.name("data").as_unique()
    r4 = graph.opr_filter.has_input(var).as_count()
    assert r4 == 1

    r5 = graph.opr_filter.name("data").as_count()
    assert r5 == 1

    opr = graph.get_opr_by_name("data")
    assert isinstance(opr, Host2DeviceCopy)

    var = graph.get_var_by_name("data")
    assert isinstance(var, VarNode)

    r6 = graph.var_filter.name("*bias").as_count()
    assert r6 == 3


def test_optimize_for_inference():
    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return F.exp(x)

    orig_model = io.BytesIO()
    f(Tensor(5.0))
    f.dump(orig_model, optimize_for_inference=False)
    orig_model.seek(0)

    optimize_model = io.BytesIO()
    net = Net.load(orig_model)
    net.dump(optimize_model, enable_io16xc32=True)
    optimize_model.seek(0)

    res = G.load_graph(optimize_model)
    computing_input = res.output_vars_list[0].owner.inputs[0]
    assert computing_input.dtype == np.float16


def test_reset_batchsize():
    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return F.exp(x)

    orig_model = io.BytesIO()
    f(Tensor(np.random.random((3, 3, 224, 224))))
    f.dump(orig_model, optimize_for_inference=False)
    orig_model.seek(0)

    modified_model = io.BytesIO()
    net = Net.load(orig_model)
    net.reset_batch_size(1)
    net.dump(modified_model, optimize_for_inference=False)
    modified_model.seek(0)

    net1 = Net.load(modified_model)
    assert net1.data_providers_filter.as_unique().shape[0] == 1


def test_modify_opr_name():
    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        return F.exp(x)

    orig_model = io.BytesIO()
    f(Tensor(np.random.random((3, 3, 224, 224))))
    f.dump(orig_model, arg_names=["a"], optimize_for_inference=False)
    orig_model.seek(0)

    modified_model = io.BytesIO()
    net = Net.load(orig_model)
    net.modify_opr_names("net")
    net.modify_opr_names(lambda x: "net1." + x)
    net.dump(modified_model, optimize_for_inference=False)
    modified_model.seek(0)

    net1 = Net.load(modified_model)
    assert net1.data_providers_filter.as_unique().name == "net1.net.a"
