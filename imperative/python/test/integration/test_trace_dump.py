import contextlib
import os
import tempfile

import numpy as np
import pytest

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine import tensor
from megengine.autodiff import GradManager
from megengine.jit import trace
from megengine.optimizer import SGD


@contextlib.contextmanager
def mkstemp():
    fd, path = tempfile.mkstemp()
    try:
        os.close(fd)
        yield path
    finally:
        os.remove(path)


def minibatch_generator(batch_size):
    while True:
        inp_data = np.zeros((batch_size, 2))
        label = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            inp_data[i, :] = np.random.rand(2) * 2 - 1
            label[i] = 1 if np.prod(inp_data[i]) < 0 else 0
        yield {"data": inp_data.astype(np.float32), "label": label.astype(np.int32)}


class XORNet(M.Module):
    def __init__(self):
        self.mid_dim = 14
        self.num_class = 2
        super().__init__()
        self.fc0 = M.Linear(self.num_class, self.mid_dim, bias=True)
        self.bn0 = M.BatchNorm1d(self.mid_dim)
        self.fc1 = M.Linear(self.mid_dim, self.mid_dim, bias=True)
        self.bn1 = M.BatchNorm1d(self.mid_dim)
        self.fc2 = M.Linear(self.mid_dim, self.num_class, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        x = F.tanh(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


def test_xornet_trace_dump():
    net = XORNet()
    opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    gm = GradManager().attach(net.parameters())
    batch_size = 64
    train_dataset = minibatch_generator(batch_size)
    val_dataset = minibatch_generator(batch_size)

    @trace
    def train_fun(data, label):
        with gm:
            net.train()
            pred = net(data)
            loss = F.nn.cross_entropy(pred, label)
            gm.backward(loss)
        return pred, loss

    @trace
    def val_fun(data, label):
        net.eval()
        pred = net(data)
        loss = F.nn.cross_entropy(pred, label)
        return pred, loss

    @trace(symbolic=True, capture_as_const=True)
    def pred_fun(data):
        net.eval()
        pred = net(data)
        pred_normalized = F.softmax(pred)
        return pred_normalized

    train_loss = []
    val_loss = []
    for step, minibatch in enumerate(train_dataset):
        if step > 100:
            break
        data = tensor(minibatch["data"])
        label = tensor(minibatch["label"])
        opt.clear_grad()
        _, loss = train_fun(data, label)
        train_loss.append((step, loss.numpy()))
        if step % 50 == 0:
            minibatch = next(val_dataset)
            _, loss = val_fun(data, label)
            loss = loss.numpy()
            val_loss.append((step, loss))
        opt.step()

    test_data = np.array(
        [
            (0.5, 0.5),
            (0.3, 0.7),
            (0.1, 0.9),
            (-0.5, -0.5),
            (-0.3, -0.7),
            (-0.9, -0.1),
            (0.5, -0.5),
            (0.3, -0.7),
            (0.9, -0.1),
            (-0.5, 0.5),
            (-0.3, 0.7),
            (-0.1, 0.9),
        ]
    )

    data = tensor(test_data.astype(np.float32))
    out = pred_fun(data)

    with mkstemp() as out:
        pred_fun.dump(out, arg_names=["data"], output_names=["label"])


def test_dump_bn_train_mode():
    @trace(symbolic=True, capture_as_const=True)
    def bn_train(data):
        pred = M.BatchNorm2d(10)(data).sum()
        return pred

    data = mge.tensor(np.random.random((10, 10, 10, 10)))
    bn_train(data)
    with pytest.raises(RuntimeError):
        bn_train.dump("test.mge")


class ViT(M.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = M.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.extra = M.Linear(embed_dim, embed_dim)
        self.head = M.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.proj(x)
        x = F.flatten(x, 2).transpose(0, 2, 1)

        x = self.extra(x)
        x = x.mean(axis=1)
        x = self.head(x)
        x = x.sum()
        return x


def test_ViTmode_trace_train():
    model = ViT(embed_dim=384)
    data = mge.random.normal(size=(1, 3, 224, 224))
    optim = SGD(model.parameters(), lr=0.01)
    gm = GradManager()
    gm.attach(model.parameters())

    @trace(symbolic=True)
    def train(d):
        with gm:
            loss = model(d)
            gm.backward(loss)

        optim.step().clear_grad()
        return loss

    @trace(symbolic=True)
    def val(d):
        loss = model(d)
        return loss

    for i in range(3):
        print(f"iter: {i}")
        t = train(data)
        r = val(data)
