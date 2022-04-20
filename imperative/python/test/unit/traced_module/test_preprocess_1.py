import io
import pickle

import numpy as np

import megengine.functional as F
import megengine.module as M
import megengine.utils.comp_graph_tools as cgtools
from megengine.core._trace_option import set_symbolic_shape
from megengine.jit import trace
from megengine.traced_module import trace_module


class Main(M.Module):
    def forward(self, x):
        return x


class PreProcess(M.Module):
    def __init__(self):
        super().__init__()
        self.I = F.ones((1,))
        self.M = F.zeros((1,))

    def forward(self, data, idx, roi):
        N, H, W, C = data.shape
        xmax = roi[:, 1, 0]
        xmin = roi[:, 0, 0]
        ymax = roi[:, 1, 1]
        ymin = roi[:, 0, 1]
        scale = F.maximum((xmax - xmin) / W, (ymax - ymin) / H)
        I = F.broadcast_to(self.I, (N,))
        M = F.broadcast_to(self.M, (N, 3, 3))
        M[:, 0, 0] = scale
        M[:, 0, 2] = xmin
        M[:, 1, 1] = scale
        M[:, 1, 2] = ymin
        M[:, 2, 2] = I
        resized = (
            F.warp_perspective(
                data, M, (H, W), mat_idx=idx, border_mode="CONSTANT", format="NHWC"
            )
            .transpose(0, 3, 1, 2)
            .astype(np.float32)
        )
        return resized


class Net(M.Module):
    def __init__(self, traced_module):
        super().__init__()
        self.pre_process = PreProcess()
        self.traced_module = traced_module

    def forward(self, data, idx, roi):
        x = self.pre_process(data, idx, roi)
        x = self.traced_module(x)
        return x


def test_preprocess():
    saved = set_symbolic_shape(True)
    module = Main()
    data = F.ones((1, 14, 8, 8), dtype=np.uint8)
    traced_module = trace_module(module, data)
    obj = pickle.dumps(traced_module)
    traced_module = pickle.loads(obj)
    module = Net(traced_module)
    module.eval()
    idx = F.zeros((1,), dtype=np.int32)
    roi = F.ones((1, 2, 2), dtype=np.float32)
    y = module(data, idx, roi)
    traced_module = trace_module(module, data, idx, roi)
    np.testing.assert_array_equal(traced_module(data, idx, roi), y)
    func = trace(traced_module, capture_as_const=True)
    np.testing.assert_array_equal(func(data, idx, roi), y)
    model = io.BytesIO()
    func.dump(model, arg_names=("data", "idx", "roi"))
    model.seek(0)
    infer_cg = cgtools.GraphInference(model)
    np.testing.assert_allclose(
        list(
            infer_cg.run(
                inp_dict={"data": data.numpy(), "idx": idx.numpy(), "roi": roi.numpy()}
            ).values()
        )[0],
        y,
        atol=1e-6,
    )

    set_symbolic_shape(saved)
