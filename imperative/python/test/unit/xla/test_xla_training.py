import platform
from functools import partial

import numpy as np
import pytest

import megengine
import megengine.autodiff as autodiff
import megengine.functional as F
import megengine.module as M
from megengine import is_cuda_available
from megengine.jit import partial_trace, xla_trace
from megengine.optimizer import SGD, Adadelta, Adagrad, Adam, AdamW


class ConvNet(M.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = M.Conv2d(3, 6, 5, bias=False)
        self.bn1 = M.BatchNorm2d(6)
        self.conv2 = M.Conv2d(6, 16, 5, bias=False)
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
def test_xla_trace_bn_opt_state_update():
    def runner(is_trace):
        np.random.seed(123)
        megengine.random.seed(123)

        model = ConvNet()
        model.train()

        image = np.random.randn(3, 8, 3, 32, 32)
        label = np.random.randint(0, 10, (3, 8,))

        gm = autodiff.GradManager().attach(model.parameters())
        optimizer = AdamW(model.parameters(), lr=0.012)

        def func(model, optimizer, timage, tlabel):
            with gm:
                score = model(timage)
                loss = F.nn.cross_entropy(score, tlabel)
                gm.backward(loss)
                optimizer.step().clear_grad()
            return loss

        if is_trace:
            func = xla_trace(func, without_host=True, capture_as_const=True)

        losses, bn_states, opt_states = [], [], []
        for i in range(6):
            timage = megengine.Tensor(image[i % 3])
            tlabel = megengine.Tensor(label[i % 3])
            loss = func(model, optimizer, timage, tlabel)

            losses.append(loss.item())
            bn_states.append(model.bn1.running_mean.numpy().reshape(-1))
            opt_states.append(
                list(optimizer._state.values())[3]["exp_avg"].numpy().reshape(-1)
            )

            if i == 4:
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.006

        return np.asarray(losses), np.stack(bn_states), np.stack(opt_states)

    imp_loss, imp_bn_states, imp_opt_states = runner(False)
    xla_loss, xla_bn_states, xla_opt_states = runner(True)
    np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5, rtol=1e-3)
    np.testing.assert_allclose(imp_bn_states, xla_bn_states, atol=1e-5)
    np.testing.assert_allclose(imp_opt_states, xla_opt_states, atol=1e-5)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_bn_opt_update():
    def runner(is_trace):
        np.random.seed(123)
        megengine.random.seed(123)

        model = ConvNet()
        model.train()

        image = np.random.randn(3, 8, 3, 32, 32)
        label = np.random.randint(0, 10, (3, 8,))

        gm = autodiff.GradManager().attach(model.parameters())
        optimizer = Adam(model.parameters(), lr=0.012, weight_decay=0.1)

        if is_trace:
            model.forward = partial(
                partial_trace(
                    func=type(model).forward, backend="xla", capture_as_const=True
                ),
                model,
            )
            optimizer._updates = partial(
                partial_trace(
                    func=type(optimizer)._updates, backend="xla", capture_as_const=True
                ),
                optimizer,
            )

        losses, bn_states, opt_states = [], [], []
        for i in range(6):
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
            losses.append(loss.item())
            bn_states.append(model.bn1.running_mean.numpy().reshape(-1))
            opt_states.append(
                list(optimizer._state.values())[7]["exp_avg"].numpy().reshape(-1)
            )

        return np.asarray(losses), np.stack(bn_states), np.stack(opt_states)

    imp_loss, imp_bn_states, imp_opt_states = runner(False)
    xla_loss, xla_bn_states, xla_opt_states = runner(True)
    np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5, rtol=1e-3)
    np.testing.assert_allclose(imp_bn_states, xla_bn_states, atol=1e-5)
    np.testing.assert_allclose(imp_opt_states, xla_opt_states, atol=1e-5)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_xla_trace_optimizer():
    def tester(OptCls, **kwargs):
        def runner(is_trace):
            np.random.seed(123)
            megengine.random.seed(123)

            model = ConvNet()
            model.train()

            image = np.random.randn(3, 8, 3, 32, 32)
            label = np.random.randint(0, 10, (3, 8,))

            gm = autodiff.GradManager().attach(model.parameters())
            optimizer = OptCls(model.parameters(), **kwargs)

            def func(model, optimizer, timage, tlabel):
                with gm:
                    score = model(timage)
                    loss = F.nn.cross_entropy(score, tlabel)
                    gm.backward(loss)
                    optimizer.step().clear_grad()
                return loss

            if is_trace:
                func = xla_trace(func, without_host=True, capture_as_const=True)

            losses, updated_weights = [], []
            for i in range(6):
                timage = megengine.Tensor(image[i % 3])
                tlabel = megengine.Tensor(label[i % 3])
                loss = func(model, optimizer, timage, tlabel)

                losses.append(loss.item())
                updated_weights.append(model.conv1.weight.numpy().reshape(-1))

                if i == 4:
                    for pg in optimizer.param_groups:
                        pg["lr"] = 0.005

            return np.asarray(losses), np.stack(updated_weights)

        imp_loss, imp_weight = runner(False)
        xla_loss, xla_weight = runner(True)
        np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5)
        np.testing.assert_allclose(imp_weight, xla_weight, atol=1e-5)

    tester(SGD, lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.1)
    tester(SGD, lr=0.01, momentum=0.9, nesterov=False, weight_decay=0.1)
    tester(SGD, lr=0.01, momentum=0, nesterov=False, weight_decay=0.1)
    tester(SGD, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0)

    tester(Adadelta, lr=0.01, rho=0.9, eps=1e-5, weight_decay=0.1)
    tester(Adadelta, lr=0.01, rho=0.9, eps=1e-5, weight_decay=0.0)

    tester(Adagrad, lr=0.01, lr_decay=0.9, eps=1e-5, weight_decay=0.1)
    tester(Adagrad, lr=0.01, lr_decay=0.9, eps=1e-5, weight_decay=0.0)
    tester(Adagrad, lr=0.01, lr_decay=0.0, eps=1e-5, weight_decay=0.1)
    tester(Adagrad, lr=0.01, lr_decay=0.0, eps=1e-5, weight_decay=0)

    tester(Adam, lr=0.01, betas=(0.1, 0.01), eps=1e-5, weight_decay=0.1)
    tester(Adam, lr=0.01, betas=(0.1, 0.01), eps=1e-5, weight_decay=0.0)

    tester(AdamW, lr=0.01, betas=(0.1, 0.01), eps=1e-5, weight_decay=0.1)
    tester(AdamW, lr=0.01, betas=(0.1, 0.01), eps=1e-5, weight_decay=0)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_optimizer():
    def tester(OptCls, **kwargs):
        def runner(is_trace):
            np.random.seed(123)
            megengine.random.seed(123)

            model = ConvNet()
            model.train()

            image = np.random.randn(3, 8, 3, 32, 32)
            label = np.random.randint(0, 10, (3, 8,))

            gm = autodiff.GradManager().attach(model.parameters())
            optimizer = OptCls(model.parameters(), **kwargs)

            if is_trace:
                model.forward = partial(
                    partial_trace(
                        func=type(model).forward, backend="xla", capture_as_const=True
                    ),
                    model,
                )
                optimizer._updates = partial(
                    partial_trace(
                        func=type(optimizer)._updates,
                        backend="xla",
                        capture_as_const=True,
                    ),
                    optimizer,
                )

            losses, updated_weights = [], []
            for i in range(6):
                timage = megengine.Tensor(image[i % 3])
                tlabel = megengine.Tensor(label[i % 3])
                with gm:
                    score = model(timage)
                    loss = F.nn.cross_entropy(score, tlabel)
                    gm.backward(loss)
                    optimizer.step().clear_grad()

                losses.append(loss.item())
                updated_weights.append(model.conv1.weight.numpy().reshape(-1))

                if i == 4:
                    for pg in optimizer.param_groups:
                        pg["lr"] = 0.005

            return np.asarray(losses), np.stack(updated_weights)

        imp_loss, imp_weight = runner(False)
        xla_loss, xla_weight = runner(True)
        np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5)
        np.testing.assert_allclose(imp_weight, xla_weight, atol=1e-5)

    tester(SGD, lr=0.01, momentum=0.9, nesterov=True, weight_decay=0.1)
    tester(SGD, lr=0.01, momentum=0.9, nesterov=False, weight_decay=0.1)
    tester(SGD, lr=0.01, momentum=0, nesterov=False, weight_decay=0.1)
    tester(SGD, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0)

    tester(Adadelta, lr=0.01, rho=0.9, eps=1e-5, weight_decay=0.1)
    tester(Adadelta, lr=0.01, rho=0.9, eps=1e-5, weight_decay=0)

    tester(Adagrad, lr=0.01, lr_decay=0.9, eps=1e-5, weight_decay=0.1)
    tester(Adagrad, lr=0.01, lr_decay=0.9, eps=1e-5, weight_decay=0.0)
    tester(Adagrad, lr=0.01, lr_decay=0.0, eps=1e-5, weight_decay=0.1)
    tester(Adagrad, lr=0.01, lr_decay=0.0, eps=1e-5, weight_decay=0)

    tester(Adam, lr=0.01, betas=(0.1, 0.01), eps=1e-5, weight_decay=0.1)
    tester(Adam, lr=0.01, betas=(0.1, 0.01), eps=1e-5, weight_decay=0.0)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_xla_trace_grad():
    def runner(is_trace):
        np.random.seed(123)
        megengine.random.seed(123)

        model = ConvNet()
        model.train()

        image = np.random.randn(3, 8, 3, 32, 32)
        label = np.random.randint(0, 10, (3, 8,))

        side_effect_cnt = 0

        def cb1(_, grad):
            nonlocal side_effect_cnt
            side_effect_cnt += 1
            return grad / 2

        def cb2(_, grad):
            return F.clip(grad, -2e-2, 2e-2)

        gm = autodiff.GradManager().attach(model.parameters(), callbacks=[cb1, cb2])
        optimizer = AdamW(model.parameters(), lr=0.01)

        def func(model, optimizer, timage, tlabel):
            with gm:
                score = model(timage)
                loss = F.nn.cross_entropy(score, tlabel)
                gm.backward(loss)
                optimizer.step().clear_grad()
            return loss

        if is_trace:
            func = xla_trace(func, without_host=True, capture_as_const=True)

        losses, weights, ses = [], [], []
        for i in range(6):
            timage = megengine.Tensor(image[i % 3])
            tlabel = megengine.Tensor(label[i % 3])
            loss = func(model, optimizer, timage, tlabel)

            losses.append(loss.item())
            weights.append(model.conv2.weight.numpy().reshape(-1))
            ses.append(side_effect_cnt)
            if i == 4:
                for pg in optimizer.param_groups:
                    pg["lr"] = 0.008

        return np.asarray(losses), np.stack(weights), np.asarray(ses)

    imp_loss, imp_weights, imp_ses = runner(False)
    xla_loss, xla_weights, xla_ses = runner(True)
    np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5, rtol=1e-3)
    np.testing.assert_array_equal(xla_ses, imp_ses[0])
    np.testing.assert_allclose(imp_weights, xla_weights, atol=1e-5)


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_grad():
    def runner(is_trace, trace_gmcallback=True):
        megengine.random.seed(123)
        np.random.seed(123)

        model = ConvNet()
        model.train()

        image = np.random.randn(3, 8, 3, 32, 32)
        label = np.random.randint(0, 10, (3, 8,))

        side_effect_cnt = 0

        def cb1(_, grad):
            nonlocal side_effect_cnt
            side_effect_cnt += 1
            return grad / 2

        def cb2(_, grad):
            return F.clip(grad, -2e-2, 2e-2)

        gm = autodiff.GradManager().attach(model.parameters(), callbacks=[cb1, cb2])
        optimizer = AdamW(model.parameters(), lr=0.01)

        if is_trace:
            model.forward = partial(
                partial_trace(
                    func=type(model).forward,
                    backend="xla",
                    capture_as_const=True,
                    trace_gmcallback=trace_gmcallback,
                ),
                model,
            )
            optimizer._updates = partial(
                partial_trace(
                    func=type(optimizer)._updates,
                    backend="xla",
                    capture_as_const=True,
                    trace_gmcallback=False,
                ),
                optimizer,
            )

        losses, weights, grads, ses = [], [], [], []
        for i in range(6):
            timage = megengine.Tensor(image[i % 3])
            tlabel = megengine.Tensor(label[i % 3])
            with gm:
                score = model(timage)
                loss = F.loss.cross_entropy(score, tlabel)
                gm.backward(loss)
                grads.append(model.conv1.weight.grad.numpy().reshape(-1))
                optimizer.step().clear_grad()

            losses.append(loss.item())
            weights.append(model.conv2.weight.numpy().reshape(-1))
            ses.append(side_effect_cnt)
        return np.asarray(losses), np.stack(weights), np.stack(grads), np.asarray(ses)

    mge_losses, mge_weights, mge_grads, mge_ses = runner(False)
    xla_losses, xla_weights, xla_grads, xla_ses = runner(True, False)
    xla_cb_losses, xla_cb_weights, xla_cb_grads, xla_cb_ses = runner(True, True)

    np.testing.assert_allclose(mge_losses, xla_losses, atol=1e-5)
    np.testing.assert_allclose(mge_losses, xla_cb_losses, atol=1e-5)
    np.testing.assert_allclose(mge_weights, xla_weights, atol=1e-5)
    np.testing.assert_allclose(mge_weights, xla_cb_weights, atol=1e-5)
    np.testing.assert_allclose(mge_grads, xla_grads, atol=1e-5)
    np.testing.assert_allclose(mge_grads, xla_cb_grads, atol=1e-5)
    np.testing.assert_array_equal(mge_ses, xla_ses)
    np.testing.assert_array_equal(mge_ses[0], xla_cb_ses)
