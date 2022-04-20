import io
import pickle

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.utils.comp_graph_tools as cgtools
from megengine.core._trace_option import set_symbolic_shape
from megengine.jit import trace
from megengine.traced_module import trace_module


class Main(M.Module):
    def forward(self, x):
        return x["data"]


class PreProcess(M.Module):
    def __init__(self):
        super().__init__()
        self.A = F.zeros((1,))
        self.I = F.ones((1,))
        self.bb_out = mge.tensor(
            np.array([[[0, 0], [160, 0], [160, 48], [0, 48]]], dtype="float32")
        )

    def forward(self, data, quad):
        """
        data: (1, 3, 48, 160)
        quad: (1, 4, 2)
        """
        N = quad.shape[0]
        dst = F.repeat(self.bb_out, N, axis=0).reshape(-1, 4, 2)
        I = F.broadcast_to(self.I, quad.shape)
        A = F.broadcast_to(self.A, (N, 8, 8))
        A[:, 0:4, 0:2] = quad
        A[:, 4:8, 5:6] = I[:, :, 0:1]
        A[:, 0:4, 6:8] = -quad * dst[:, :, 0:1]
        A[:, 4:8, 3:5] = quad
        A[:, 0:4, 2:3] = I[:, :, 0:1]
        A[:, 4:8, 6:8] = -quad * dst[:, :, 1:2]
        B = dst.transpose(0, 2, 1).reshape(-1, 8, 1)
        M = F.concat([F.matmul(F.matinv(A), B)[:, :, 0], I[:, 0:1, 0]], axis=1).reshape(
            -1, 3, 3
        )
        new_data = F.warp_perspective(data, M, (48, 160))  # (N, 3, 48, 160)
        return {"data": new_data}


class Net(M.Module):
    def __init__(self, traced_module):
        super().__init__()
        self.pre_process = PreProcess()
        self.traced_module = traced_module

    def forward(self, data, quad):
        x = self.pre_process(data, quad)
        x = self.traced_module(x)
        return x


def test_preprocess():
    saved = set_symbolic_shape(True)
    batch_size = 2
    module = Main()
    data = mge.tensor(
        np.random.randint(0, 256, size=(batch_size, 3, 48, 160)), dtype=np.float32
    )
    traced_module = trace_module(module, {"data": data})
    obj = pickle.dumps(traced_module)
    traced_module = pickle.loads(obj)
    module = Net(traced_module)
    module.eval()
    quad = mge.tensor(np.random.normal(size=(batch_size, 4, 2)), dtype=np.float32)
    expect = module(data, quad)
    traced_module = trace_module(module, data, quad)
    actual = traced_module(data, quad)
    for i, j in zip(expect, actual):
        np.testing.assert_array_equal(i, j)
    func = trace(traced_module, capture_as_const=True)
    actual = func(data, quad)
    for i, j in zip(expect, actual):
        np.testing.assert_array_equal(i, j)
    model = io.BytesIO()
    func.dump(model, arg_names=("data", "quad"))
    model.seek(0)
    infer_cg = cgtools.GraphInference(model)
    actual = list(
        infer_cg.run(inp_dict={"data": data.numpy(), "quad": quad.numpy()}).values()
    )[0]
    np.testing.assert_allclose(expect, actual)

    set_symbolic_shape(saved)
