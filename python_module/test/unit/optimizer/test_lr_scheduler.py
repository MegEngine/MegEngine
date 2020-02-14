from bisect import bisect_right

import numpy as np
from helpers import MLP

from megengine.optimizer import SGD, MultiStepLR
from megengine.test import assertTensorClose


def test_multi_step_lr():
    mlp = MLP()
    opt = SGD(mlp.parameters(), lr=0.01, momentum=0.9)
    scheduler = MultiStepLR(opt, [3, 6, 8])

    lr = np.array(0.01, dtype=np.float32)
    for i in range(10):
        for group in opt.param_groups:
            assertTensorClose(
                np.array(group["lr"], dtype=np.float32),
                (lr * 0.1 ** bisect_right([3, 6, 8], i)).astype(np.float32),
                max_err=5e-6,
            )
        scheduler.step()
