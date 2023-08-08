import numpy as np
import pytest

import megengine as mge
import megengine.autodiff as ad
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim
from megengine import tensor
from megengine.core._imperative_rt.core2 import apply
from megengine.core.ops.builtin import LAMBUpdate


def lamb_update(
    param_group, step, exp_avg, exp_avg_sq, param, grad, bias_correction, always_adapt
):
    lr = param_group["lr"]
    weight_decay = param_group["weight_decay"]
    eps = param_group["eps"]
    beta0, beta1 = param_group["betas"]

    # since `conver_inputs` is disabled for param updates,
    # scalar should be explicitly tansforred to tensor

    _lr, _neg_lr = map(tensor, (lr, -lr))
    _weight_decay = tensor(weight_decay)
    _eps = tensor(eps)
    _beta0, _beta1 = map(tensor, (beta0, beta1))

    c1, c05, c0 = map(tensor, (1.0, 0.5, 0.0))

    def norm(vec):
        return sum(vec * vec) ** c05

    p_norm = norm(param.flatten())

    # step = step + c1
    step += c1

    # exp_avg = _beta0 * exp_avg + grad * (c1 - _beta0)
    exp_avg *= _beta0
    exp_avg += grad * (c1 - _beta0)

    # exp_avg_sq = _beta1 * exp_avg_sq + (c1 - _beta1) * (grad * grad)
    exp_avg_sq *= _beta1
    exp_avg_sq += (c1 - _beta1) * (grad * grad)

    bias_correction1 = c1 - _beta0 ** step if bias_correction else c1
    bias_correction2 = c1 - _beta1 ** step if bias_correction else c1
    delta = (exp_avg / bias_correction1) / (
        (exp_avg_sq / bias_correction2) ** c05 + _eps
    )
    if weight_decay != 0.0:
        delta += param * _weight_decay

    d_norm = norm(delta.flatten())
    trust_ratio = (
        p_norm / d_norm
        if (always_adapt or weight_decay > 0) and p_norm > c0 and d_norm > c0
        else c1
    )
    new_param = param - _lr * trust_ratio * delta
    return exp_avg, exp_avg_sq, new_param


@pytest.mark.skip(reason="pytest aborted, the same as groupnorm")
def test_lamb():
    op = LAMBUpdate(0.9, 0.999, 1, 1e-3, 0.4, 1e-8, True, False)
    m_t_1 = mge.tensor(np.random.uniform(size=(256, 256)), dtype=np.float32)
    v_t_1 = mge.tensor(np.random.uniform(size=(256, 256)), dtype=np.float32)
    params = mge.tensor(np.random.uniform(size=(256, 256)), dtype=np.float32)
    grad = mge.tensor(np.random.uniform(size=(256, 256)), dtype=np.float16)
    (new_m_t, new_v_t, new_param) = apply(op, m_t_1, v_t_1, params, grad)

    param_group = {
        "betas": (0.9, 0.999),
        "step": 1,
        "lr": 1e-3,
        "weight_decay": 0.4,
        "eps": 1e-8,
    }
    gt_m_t, gt_v_t, gt_new_param = lamb_update(
        param_group, 1, m_t_1, v_t_1, params, grad, True, False
    )
    np.testing.assert_allclose(new_m_t.numpy(), gt_m_t.numpy(), atol=1e-2)
    np.testing.assert_allclose(new_v_t.numpy(), gt_v_t.numpy(), atol=1e-2)
    np.testing.assert_allclose(new_param.numpy(), gt_new_param.numpy(), atol=1e-2)
