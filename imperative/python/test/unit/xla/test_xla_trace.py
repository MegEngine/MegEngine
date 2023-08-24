import platform

import numpy as np
import pytest

import megengine
import megengine.autodiff as autodiff
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine import is_cuda_available, tensor
from megengine.jit import partial_trace, xla_trace


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_inplace():
    def func(x, y):
        x += 1
        y += 1

    xla_func = partial_trace(func, backend="xla")
    xla_func(tensor(1), tensor(2))

    a1 = megengine.tensor(1)
    a2 = megengine.tensor(2)
    xla_func(a1, a2)
    np.testing.assert_allclose(a1, 2)
    np.testing.assert_allclose(a2, 3)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_xla_trace_shape_change():
    def func(x, y):
        return x + y

    xla_func = partial_trace(func, backend="xla")
    a = np.random.randn(1, 3, 3, 3)
    b = np.random.randn(1, 3, 3, 3)
    rst0 = xla_func(tensor(a), tensor(b))
    rst1 = xla_func(tensor(1.0), tensor(2.0))  # fallback to python function
    rst2 = xla_func(tensor(a), tensor(b))  # exec in xla

    assert not rst1._is_external_value()
    assert rst2._is_external_value()


class ConvNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(3, 6, 5)
        self.bn1 = M.BatchNorm2d(6)
        self.conv2 = M.Conv2d(6, 16, 5)
        self.bn2 = M.BatchNorm2d(16)
        self.fc1 = M.Linear(16 * 5 * 5, 120)
        self.fc2 = M.Linear(120, 84)
        self.classifier = M.Linear(84, 10)

        self.pool = M.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.bn1(self.conv1(x)))
        x = self.pool(self.bn2(self.conv2(x)))
        x = F.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classifier(x)
        return x


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_xla_trace_training():
    def runner(is_trace):
        np.random.seed(123)
        megengine.random.seed(123)

        model = ConvNet()
        model.train()

        image = np.random.randn(3, 8, 3, 32, 32)
        label = np.random.randint(0, 10, (3, 8,))

        gm = autodiff.GradManager().attach(model.parameters())
        optimizer = optim.AdamW(model.parameters(), lr=0.012)

        def func(model, optimizer, timage, tlabel):
            with gm:
                score = model(timage)
                loss = F.nn.cross_entropy(score, tlabel)
                gm.backward(loss)
                optimizer.step().clear_grad()
            return loss

        if is_trace:
            func = xla_trace(func, without_host=True, capture_as_const=True)

        losses, bn_states = [], []
        for i in range(10):
            timage = megengine.Tensor(image[i % 3])
            tlabel = megengine.Tensor(label[i % 3])
            loss = func(model, optimizer, timage, tlabel)
            losses.append(loss.item())

            bn_states.append(model.bn1.running_mean.numpy().reshape(-1))

            if i == 4:
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.006

        return np.asarray(losses), bn_states

    imp_loss, _ = runner(False)
    xla_loss, xla_bn_states = runner(True)
    np.testing.assert_allclose(imp_loss, xla_loss, atol=5e-4, rtol=1e-3)
    assert np.all(xla_bn_states[-1] != xla_bn_states[-2])


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_training():
    def runner(is_trace):
        np.random.seed(123)
        megengine.random.seed(123)

        model = ConvNet()
        model.train()

        image = np.random.randn(3, 8, 3, 32, 32)
        label = np.random.randint(0, 10, (3, 8,))

        gm = autodiff.GradManager().attach(model.parameters())
        optimizer = optim.AdamW(model.parameters(), lr=0.012)

        if is_trace:
            type(model).forward = partial_trace(
                func=type(model).forward, backend="xla", capture_as_const=True
            )
            type(optimizer)._updates = partial_trace(
                func=type(optimizer)._updates, backend="xla", capture_as_const=True
            )

        losses, bn_states = [], []
        for i in range(10):
            timage = megengine.Tensor(image[i % 3])
            tlabel = megengine.Tensor(label[i % 3])
            with gm:
                score = model(timage)
                loss = F.nn.cross_entropy(score, tlabel)
                gm.backward(loss)
                optimizer.step().clear_grad()

            if i == 4:
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.006
                    pg["weight_decay"] = 0.2
            bn_states.append(model.bn1.running_mean.numpy().reshape(-1))
            losses.append(loss.item())
        return np.asarray(losses), bn_states

    imp_loss, _ = runner(False)
    xla_loss, xla_bn_states = runner(True)
    np.testing.assert_allclose(imp_loss, xla_loss, atol=5e-4, rtol=1e-3)
    assert np.all(xla_bn_states[-1] != xla_bn_states[-2])
