import numpy as np
import pytest

import megengine as mge
import megengine.autodiff as autodiff
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine import Parameter, Tensor, amp
from megengine.core._config import set_auto_format_convert
from megengine.core._trace_option import use_symbolic_shape


class MyModule(M.Module):
    class InnerModule(M.Module):
        def __init__(self):
            super().__init__()
            self.bn = M.BatchNorm2d(4)

        def forward(self, x):
            return self.bn(x)

    def __init__(self):
        super().__init__()
        self.i = self.InnerModule()
        self.conv = M.Conv2d(4, 4, 4, groups=2)
        self.bn = M.BatchNorm2d(4)
        self.param = Parameter(np.ones((1, 3, 1, 1), dtype=np.float32))
        self.buff = Tensor(np.ones((1, 3, 1, 1), dtype=np.float32))

    def forward(self, x):
        x = self.i(x)
        x = self.bn(x)
        return x


@pytest.mark.parametrize("is_inplace", [False, True])
def test_convert_module(is_inplace):
    m = MyModule()
    expected_shape = {
        "i.bn.weight": (1, 4, 1, 1),
        "i.bn.bias": (1, 4, 1, 1),
        "i.bn.running_mean": (1, 4, 1, 1),
        "i.bn.running_var": (1, 4, 1, 1),
        "conv.weight": (2, 2, 2, 4, 4),
        "conv.bias": (1, 4, 1, 1),
        "bn.weight": (1, 4, 1, 1),
        "bn.bias": (1, 4, 1, 1),
        "bn.running_mean": (1, 4, 1, 1),
        "bn.running_var": (1, 4, 1, 1),
        "param": (1, 3, 1, 1),
        "buff": (1, 3, 1, 1),
    }
    m = amp.convert_module_format(m, is_inplace)
    for name, param in m.named_tensors():
        assert param.format == "nhwc"
        if use_symbolic_shape():
            np.testing.assert_array_equal(
                param.shape.numpy(), expected_shape[name], name
            )
        else:
            assert param.shape == expected_shape[name], name


class Module(M.Module):
    def __init__(self):
        super().__init__()
        self.conv = M.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = M.BatchNorm2d(16)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out


def test_format_remained():
    m = Module()

    m = amp.convert_module_format(m)

    gm = autodiff.GradManager().attach(m.parameters())
    opt = optim.SGD(m.parameters(), lr=0.01)
    scaler = amp.GradScaler()

    image = mge.tensor(np.random.normal(size=(1, 3, 224, 224)), dtype="float32")
    label = mge.tensor(np.ones((1, 224, 224)), dtype="int32")

    image = amp.convert_tensor_format(image)

    @amp.autocast(enabled=True)
    def train_step(image):
        with gm:
            logits = m(image)
            loss = F.nn.cross_entropy(logits, label)
            scaler.backward(gm, loss)
        opt.step().clear_grad()
        return logits

    for _ in range(5):
        res = train_step(image)
        assert res.format == "nhwc"
