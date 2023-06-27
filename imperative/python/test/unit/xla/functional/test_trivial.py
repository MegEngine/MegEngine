import numpy as np

import megengine.jit as jit
import megengine.tensor as tensor


def test_get_var_shape():
    def tester(shape):
        x = tensor(np.random.randn(*shape).astype("float32"))

        @jit.trace(without_host=True, use_xla=True)
        def func(x):
            return x.shape

        mge_rst = func(x)
        xla_rst = func(x)
        np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((2, 3, 4, 5))
    tester((1, 2, 3))
    tester((1,))


if __name__ == "__main__":
    test_get_var_shape()
