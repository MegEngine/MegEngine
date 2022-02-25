import numpy as np

import megengine as mge
import megengine.autodiff as ad
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim


class Net(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = M.BatchNorm2d(64)
        self.avgpool = M.AvgPool2d(kernel_size=5, stride=5, padding=0)
        self.fc = M.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = F.avg_pool2d(x, 22)
        x = F.flatten(x, 1)
        x = self.fc(x)
        return x


def save_grad_value(net):
    for param in net.parameters():
        param.grad_backup = param.grad.numpy().copy()


def test_clip_grad_norm():
    net = Net()
    x = mge.tensor(np.random.randn(10, 3, 224, 224))
    gm = ad.GradManager().attach(net.parameters())
    opt = optim.SGD(net.parameters(), 1e-3, momentum=0.9)
    with gm:
        loss = net(x).sum()
        gm.backward(loss)
    save_grad_value(net)
    max_norm = 1.0
    original_norm = optim.clip_grad_norm(net.parameters(), max_norm=max_norm, ord=2)
    scale = max_norm / original_norm
    for param in net.parameters():
        np.testing.assert_almost_equal(param.grad.numpy(), param.grad_backup * scale)
    opt.step().clear_grad()


def test_clip_grad_value():
    net = Net()
    x = np.random.randn(10, 3, 224, 224).astype("float32")
    gm = ad.GradManager().attach(net.parameters())
    opt = optim.SGD(net.parameters(), 1e-3, momentum=0.9)
    with gm:
        y = net(mge.tensor(x))
        y = y.mean()
        gm.backward(y)
    save_grad_value(net)
    max_val = 5
    min_val = -2
    optim.clip_grad_value(net.parameters(), lower=min_val, upper=max_val)
    for param in net.parameters():
        np.testing.assert_almost_equal(
            param.grad.numpy(),
            np.maximum(np.minimum(param.grad_backup, max_val), min_val),
        )
    opt.step().clear_grad()
