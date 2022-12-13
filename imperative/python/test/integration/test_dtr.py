import multiprocessing as mp

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
import megengine.tensor as tensor
from megengine.autodiff import GradManager
from megengine.data import DataLoader, RandomSampler, transform
from megengine.data.dataset import CIFAR10


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, M.Linear) or isinstance(m, M.Conv2d):
        M.init.msra_normal_(m.weight)


mean = [125.3, 123.0, 113.9]
std = [63.0, 62.1, 66.7]


class BasicBlock(M.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = M.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = M.BatchNorm2d(planes)
        self.conv2 = M.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = M.BatchNorm2d(planes)
        self.shortcut = M.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = M.Sequential(
                M.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                M.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(M.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = M.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = M.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = M.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return M.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean(3).mean(2)
        out = self.linear(out)
        return out


def run_dtr_drop_copy_dev_tensor():
    mge.dtr.evictee_minimum_size = 128
    mge.dtr.enable()
    x = F.ones((10, 100))
    x._drop()
    x[...] = mge.tensor(x, no_cache=True)
    x.numpy()
    mge.dtr.evictee_minimum_size = 1024 ** 2
    mge.dtr.disable()
    mge._exit(0)


def run_dtr_resnet1202():
    batch_size = 6
    resnet1202 = ResNet(BasicBlock, [200, 200, 200])
    opt = optim.SGD(resnet1202.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    gm = GradManager().attach(resnet1202.parameters())

    def train_func(data, label, *, net, gm):
        net.train()
        with gm:
            pred = net(data)
            loss = F.loss.cross_entropy(pred, label)
            gm.backward(loss)
        return pred, loss

    _, free_mem = mge.device.get_mem_status_bytes()
    tensor_mem = free_mem - (2 ** 30)
    if tensor_mem > 0:
        x = np.ones((1, int(tensor_mem / 4)), dtype=np.float32)
    else:
        x = np.ones((1,), dtype=np.float32)
    t = mge.tensor(x)

    mge.dtr.enable()
    mge.dtr.enable_sqrt_sampling = True

    data = np.random.randn(batch_size, 3, 32, 32).astype("float32")
    label = np.random.randint(0, 10, size=(batch_size,)).astype("int32")
    for _ in range(2):
        opt.clear_grad()
        _, loss = train_func(mge.tensor(data), mge.tensor(label), net=resnet1202, gm=gm)
        opt.step()
        loss.item()

    t.numpy()
    mge.dtr.disable()
    mge._exit(0)


@pytest.mark.require_ngpu(1)
@pytest.mark.isolated_distributed
def test_dtr_resnet1202():
    p = mp.Process(target=run_dtr_resnet1202)
    p.start()
    p.join()
    assert p.exitcode == 0


@pytest.mark.require_ngpu(1)
@pytest.mark.isolated_distributed
def test_dtr_drop_copy_dev_tensor():
    p = mp.Process(target=run_dtr_drop_copy_dev_tensor)
    p.start()
    p.join()
    assert p.exitcode == 0


@pytest.mark.require_ngpu(1)
@pytest.mark.isolated_distributed
def test_dtr_drop_tensor():
    for i in range(50):
        test_dtr_drop_copy_dev_tensor()
