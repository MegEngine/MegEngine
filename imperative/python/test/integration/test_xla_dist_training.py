import platform
from functools import partial

import numpy as np
import pytest

import megengine
import megengine.autodiff as autodiff
import megengine.functional as F
import megengine.module as M
from megengine import distributed as dist
from megengine import is_cuda_available
from megengine.jit import partial_trace, xla_trace
from megengine.optimizer import AdamW


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
@pytest.mark.isolated_distributed
def test_xla_trace_dist_training():
    @dist.launcher(n_gpus=2, device_type="gpu")
    def worker():
        def runner(is_trace, use_mycb):
            np.random.seed(dist.get_rank() + 123)
            megengine.random.seed(dist.get_rank() + 123)

            model = ConvNet()
            model.train()

            if dist.is_distributed():
                dist.bcast_list_(model.tensors())

            side_effect_cnt = 0

            def mycb(_, grad):
                nonlocal side_effect_cnt
                side_effect_cnt += 1
                return F.clip(grad, -2e-2, 2e-2)

            cblist = (
                [mycb, dist.make_allreduce_cb("mean")]
                if use_mycb
                else [dist.make_allreduce_cb("mean")]
            )
            gm = autodiff.GradManager().attach(model.parameters(), callbacks=cblist)
            optimizer = AdamW(model.parameters(), lr=0.01)

            image = np.random.randn(3, 8, 3, 32, 32)
            label = np.random.randint(0, 10, (3, 8,))

            def func(model, optimizer, timage, tlabel):
                with gm:
                    score = model(timage)
                    loss = F.nn.cross_entropy(score, tlabel)
                    gm.backward(loss)
                    optimizer.step().clear_grad()
                return loss

            if is_trace:
                func = xla_trace(func, without_host=True, capture_as_const=True)

            losses, bn_states, opt_states, weights, ses = [], [], [], [], []
            for i in range(6):
                timage = megengine.Tensor(image[i % 3])
                tlabel = megengine.Tensor(label[i % 3])
                loss = func(model, optimizer, timage, tlabel)

                losses.append(loss.item())
                bn_states.append(model.bn1.running_mean.numpy().reshape(-1))
                opt_states.append(
                    list(optimizer._state.values())[3]["exp_avg"].numpy().reshape(-1)
                )
                weights.append(model.conv2.weight.numpy().reshape(-1))
                ses.append(side_effect_cnt)

                if i == 4:
                    for pg in optimizer.param_groups:
                        pg["lr"] = 0.006

            return (
                np.asarray(losses),
                np.stack(bn_states),
                np.stack(opt_states),
                np.stack(weights),
                np.asarray(ses),
            )

        for use_cb in [True, False]:
            imp_loss, imp_bn_states, imp_opt_states, imp_weights, imp_ses = runner(
                False, use_cb
            )
            xla_loss, xla_bn_states, xla_opt_states, xla_weights, xla_ses = runner(
                True, use_cb
            )
            np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5, rtol=1e-3)
            np.testing.assert_allclose(imp_bn_states, xla_bn_states, atol=1e-5)
            np.testing.assert_allclose(imp_opt_states, xla_opt_states, atol=1e-5)
            np.testing.assert_allclose(imp_weights, xla_weights, atol=1e-5)
            np.testing.assert_array_equal(imp_ses[0], xla_ses)

    worker()


@pytest.mark.skipif(int(platform.python_version_tuple()[1]) < 8, reason="need py38")
@pytest.mark.skipif(platform.system() != "Linux", reason="only support linux now")
@pytest.mark.skipif(not is_cuda_available(), reason="only support cuda now")
def test_partial_trace_dist_training():
    @dist.launcher(n_gpus=2, device_type="gpu")
    def worker():
        def runner(is_trace, trace_gmcallback, use_mycb):
            np.random.seed(dist.get_rank() + 123)
            megengine.random.seed(dist.get_rank() + 123)

            model = ConvNet()
            model.train()

            if dist.is_distributed():
                dist.bcast_list_(model.tensors())

            side_effect_cnt = 0

            def mycb(_, grad):
                nonlocal side_effect_cnt
                side_effect_cnt += 1
                return F.clip(grad, -2e-2, 2e-2)

            cblist = (
                [mycb, dist.make_allreduce_cb("mean")]
                if use_mycb
                else [dist.make_allreduce_cb("mean")]
            )
            gm = autodiff.GradManager().attach(model.parameters(), callbacks=cblist)
            optimizer = AdamW(model.parameters(), lr=0.01)

            image = np.random.randn(3, 8, 3, 32, 32)
            label = np.random.randint(0, 10, (3, 8,))

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
                    ),
                    optimizer,
                )

            losses, bn_states, opt_states, weights, grads, ses = [], [], [], [], [], []
            for i in range(6):
                timage = megengine.Tensor(image[i % 3])
                tlabel = megengine.Tensor(label[i % 3])
                with gm:
                    score = model(timage)
                    loss = F.nn.cross_entropy(score, tlabel)
                    gm.backward(loss)
                    grads.append(model.conv1.weight.grad.numpy().reshape(-1))
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
                weights.append(model.conv2.weight.numpy().reshape(-1))
                ses.append(side_effect_cnt)

            return (
                np.asarray(losses),
                np.stack(bn_states),
                np.stack(opt_states),
                np.stack(weights),
                np.stack(grads),
                np.asarray(ses),
            )

        for use_mycb in [True, False]:
            (
                imp_loss,
                imp_bn_states,
                imp_opt_states,
                imp_weights,
                imp_grads,
                imp_ses,
            ) = runner(False, False, use_mycb)
            (
                xla_loss,
                xla_bn_states,
                xla_opt_states,
                xla_weights,
                xla_grads,
                xla_ses,
            ) = runner(True, False, use_mycb)
            (
                xla_cb_loss,
                xla_cb_bn_states,
                xla_cb_opt_states,
                xla_cb_weights,
                xla_cb_grads,
                xla_cb_ses,
            ) = runner(True, True, use_mycb)

            np.testing.assert_allclose(imp_loss, xla_loss, atol=1e-5, rtol=1e-3)
            np.testing.assert_allclose(imp_loss, xla_cb_loss, atol=1e-5, rtol=1e-3)
            np.testing.assert_allclose(imp_bn_states, xla_bn_states, atol=1e-5)
            np.testing.assert_allclose(imp_bn_states, xla_cb_bn_states, atol=1e-5)
            np.testing.assert_allclose(imp_opt_states, xla_opt_states, atol=1e-5)
            np.testing.assert_allclose(imp_opt_states, xla_cb_opt_states, atol=1e-5)
            np.testing.assert_allclose(imp_weights, xla_weights, atol=1e-5)
            np.testing.assert_allclose(imp_weights, xla_cb_weights, atol=1e-5)
            np.testing.assert_allclose(imp_grads, xla_grads, atol=1e-5)
            np.testing.assert_allclose(imp_grads, xla_cb_grads, atol=1e-5)
            np.testing.assert_array_equal(imp_ses, xla_ses)
            np.testing.assert_array_equal(imp_ses[0], xla_cb_ses)

    worker()
