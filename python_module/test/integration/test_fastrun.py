import numpy as np

import megengine as mge
from megengine.functional.debug_param import set_conv_execution_strategy
from megengine.module.conv import Conv2d


def test_fastrun():
    set_conv_execution_strategy("PROFILE")
    x = Conv2d(1, 1, kernel_size=1, bias=True)
    a = mge.tensor(np.random.randn(1, 1, 1, 1).astype(np.float32))
    a = x(a)
