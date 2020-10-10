# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import contextlib
import os
import tempfile

import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine import tensor
from megengine.autodiff import GradManager
from megengine.jit import trace


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
        self.fc1 = M.Linear(self.mid_dim, self.mid_dim, bias=True)
        self.fc2 = M.Linear(self.mid_dim, self.num_class, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = F.tanh(x)
        x = self.fc1(x)
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
            loss = loss.numpy()[0]
            val_loss.append((step, loss))
            print("Step: {} loss={}".format(step, loss))
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
    pred_output = out.numpy()
    pred_label = np.argmax(pred_output, 1)

    with np.printoptions(precision=4, suppress=True):
        print("Predicated probability:")
        print(pred_output)

    with mkstemp() as out:
        pred_fun.dump(out, arg_names=["data"], output_names=["label"])
