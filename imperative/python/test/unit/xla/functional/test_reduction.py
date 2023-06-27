import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine.autodiff.grad_manager import GradManager


def test_reduce():
    np.random.seed(123)
    mge.random.seed(123)

    def tester(freduce, inpshape, axes, keepdim, backward, dtype=None, atol=1e-5):
        dtype = dtype or np.float32
        inp = tensor(0.1 * np.random.randn(*inpshape), dtype=dtype)
        doup = tensor(
            0.1 * np.random.randn(*freduce(inp, axis=axes, keepdims=keepdim).shape),
            dtype=dtype,
        )

        gm = GradManager()

        @jit.trace(without_host=True, use_xla=True)
        def func(inp, doup):
            gm.attach([inp])
            with gm:
                oup = freduce(inp, axis=axes, keepdims=keepdim)
                if backward:
                    gm.backward(oup, doup)
                    return [oup, inp.grad]
                else:
                    return [oup]

        mge_rsts = func(inp, doup)
        xla_rsts = func(inp, doup)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=atol)

    tester(F.sum, (2, 4, 8, 16), [1, 2], keepdim=True, backward=True)
    tester(F.mean, (2, 4, 8, 16), [3, 2], keepdim=False, backward=True)
    tester(F.prod, (2, 4, 8, 16), [0, 1, 2, 3], keepdim=False, backward=True)
    tester(F.min, (2, 4, 8, 16), 0, keepdim=True, backward=False)
    tester(F.max, (2, 4, 8, 16), [-2], keepdim=False, backward=False)


if __name__ == "__main__":
    test_reduce()
