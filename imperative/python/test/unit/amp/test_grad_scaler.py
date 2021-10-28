# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np

import megengine as mge
from megengine.amp import GradScaler
from megengine.autodiff import GradManager
from megengine.jit import trace


def test_grad_scaler():
    def f():
        gm = GradManager()
        scaler = GradScaler()

        x = mge.tensor(1.0)
        for _ in range(3):
            with gm:
                y = x + 1
                gm.attach(y)
                loss = y + 1
                scaler.backward(gm, loss, unscale_grad=False)
            np.testing.assert_equal(y.grad.numpy(), scaler.scale_factor)
            scaler.unscale(gm.attached_tensors())
            np.testing.assert_equal(y.grad.numpy(), 1)
        # test handle None elements
        scaler.unscale(gm.attached_tensors())

    f()
    trace(f)()
