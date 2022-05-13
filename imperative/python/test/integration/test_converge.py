# -*- coding: utf-8 -*-
import itertools

import numpy as np
import pytest

import megengine as mge
import megengine.autodiff as ad
import megengine.functional as F
import megengine.optimizer as optim
from megengine import Tensor
from megengine.core import set_option
from megengine.module import Linear, Module
from megengine.optimizer import SGD
from megengine.traced_module import trace_module

batch_size = 64
data_shape = (batch_size, 2)
label_shape = (batch_size,)


def minibatch_generator():
    while True:
        inp_data = np.zeros((batch_size, 2))
        label = np.zeros(batch_size, dtype=np.int32)
        for i in range(batch_size):
            # [x0, x1], sampled from U[-1, 1]
            inp_data[i, :] = np.random.rand(2) * 2 - 1
            label[i] = 0 if np.prod(inp_data[i]) < 0 else 1
        yield inp_data.astype(np.float32), label.astype(np.int32)


def calculate_precision(data: np.ndarray, pred: np.ndarray) -> float:
    """ Calculate precision for given data and prediction.

    :type data: [[x, y], ...]
    :param data: Input data
    :type pred: [[x_pred, y_pred], ...]
    :param pred: Network output data
    """
    correct = 0
    assert len(data) == len(pred)
    for inp_data, pred_output in zip(data, pred):
        label = 0 if np.prod(inp_data) < 0 else 1
        pred_label = np.argmax(pred_output)
        if pred_label == label:
            correct += 1
    return float(correct) / len(data)


class XORNet(Module):
    def __init__(self):
        self.mid_layers = 14
        self.num_class = 2
        super().__init__()

        self.fc0 = Linear(self.num_class, self.mid_layers, bias=True)
        self.fc1 = Linear(self.mid_layers, self.mid_layers, bias=True)

        self.fc2 = Linear(self.mid_layers, self.num_class, bias=True)

    def forward(self, x):
        x = self.fc0(x)
        x = F.tanh(x)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


@pytest.mark.parametrize(
    "test_traced_module, with_drop, grad_clip",
    [(False, False, False), (True, True, True)],
)
def test_training_converge(test_traced_module, with_drop, grad_clip):
    if with_drop:
        set_option("enable_drop", 1)
    net = XORNet()
    if test_traced_module:
        inp = Tensor(np.random.random((14, 2)))
        net = trace_module(net, inp)

    opt = SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    gm = ad.GradManager().attach(net.parameters())

    def train(data, label):
        with gm:
            pred = net(data)
            loss = F.nn.cross_entropy(pred, label)
            gm.backward(loss)
            if grad_clip:
                optim.clip_grad_norm(net.parameters(), max_norm=0.2, ord=2.0)
        return loss

    def infer(data):
        return net(data)

    train_dataset = minibatch_generator()
    losses = []

    for data, label in itertools.islice(train_dataset, 1500):
        data = Tensor(data, dtype=np.float32)
        label = Tensor(label, dtype=np.int32)
        opt.clear_grad()
        loss = train(data, label)
        if grad_clip:
            optim.clip_grad_value(net.parameters(), lower=-0.1, upper=0.1)
        opt.step()
        losses.append(loss.numpy())

    assert np.mean(losses[-100:]) < 0.1, "Final training Loss must be low enough"

    ngrid = 10
    x = np.linspace(-1.0, 1.0, ngrid)
    xx, yy = np.meshgrid(x, x)
    xx = xx.reshape((ngrid * ngrid, 1))
    yy = yy.reshape((ngrid * ngrid, 1))
    data = mge.tensor(np.concatenate((xx, yy), axis=1).astype(np.float32))
    pred = infer(data)
    precision = calculate_precision(data.numpy(), pred.numpy())
    assert precision == 1.0, "Test precision must be high enough, get {}".format(
        precision
    )

    if with_drop:
        set_option("enable_drop", 0)
