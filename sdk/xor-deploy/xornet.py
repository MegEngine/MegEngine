import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine.jit import trace


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


@trace(symbolic=True)
def train_fun(data, label, net=None, opt=None):
    net.train()
    pred = net(data)
    loss = F.cross_entropy_with_softmax(pred, label)
    opt.backward(loss)
    return pred, loss


@trace(symbolic=True)
def val_fun(data, label, net=None):
    net.eval()
    pred = net(data)
    loss = F.cross_entropy_with_softmax(pred, label)
    return pred, loss


@trace(symbolic=True)
def pred_fun(data, net=None):
    net.eval()
    pred = net(data)
    pred_normalized = F.softmax(pred)
    return pred_normalized


def main():

    if not mge.is_cuda_available():
        mge.set_default_device("cpux")

    net = XORNet()
    opt = optim.SGD(net.parameters(requires_grad=True), lr=0.01, momentum=0.9)
    batch_size = 64
    train_dataset = minibatch_generator(batch_size)
    val_dataset = minibatch_generator(batch_size)

    data = mge.tensor()
    label = mge.tensor(np.zeros((batch_size,)), dtype=np.int32)
    train_loss = []
    val_loss = []
    for step, minibatch in enumerate(train_dataset):
        if step > 1000:
            break
        data.set_value(minibatch["data"])
        label.set_value(minibatch["label"])
        opt.zero_grad()
        _, loss = train_fun(data, label, net=net, opt=opt)
        train_loss.append((step, loss.numpy()))
        if step % 50 == 0:
            minibatch = next(val_dataset)
            _, loss = val_fun(data, label, net=net)
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

    data.set_value(test_data)
    out = pred_fun(data, net=net)
    pred_output = out.numpy()
    pred_label = np.argmax(pred_output, 1)

    print("Test data")
    print(test_data)

    with np.printoptions(precision=4, suppress=True):
        print("Predicated probability:")
        print(pred_output)

    print("Predicated label")
    print(pred_label)

    model_name = "xornet_deploy.mge"

    if pred_fun.enabled:
        print("Dump model as {}".format(model_name))
        pred_fun.dump(model_name, arg_names=["data"])
    else:
        print("pred_fun must be run with trace enabled in order to dump model")


if __name__ == "__main__":
    main()
