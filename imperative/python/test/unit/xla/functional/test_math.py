import numpy as np

import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine.autodiff.grad_manager import GradManager


def test_matmul():
    def tester(lhs_shape, rhs_shape, lhs_transpose, rhs_transpose, dtype=None):
        lhs = tensor(0.1 * np.random.randn(*lhs_shape), dtype=dtype)
        rhs = tensor(0.1 * np.random.randn(*rhs_shape), dtype=dtype)
        out = F.matmul(lhs, rhs, lhs_transpose, rhs_transpose)
        dout = tensor(0.1 * np.random.randn(*out.shape), dtype=dtype)

        gm = GradManager()

        @jit.trace(without_host=True, use_xla=True)
        def func(lhs, rhs, dout):
            gm.attach([lhs, rhs])
            with gm:
                out = F.matmul(lhs, rhs, lhs_transpose, rhs_transpose)
                gm.backward(out, dout)
            return out, lhs.grad, rhs.grad

        mge_rsts = func(lhs, rhs, dout)
        mge_rsts[0].numpy()
        xla_rsts = func(lhs, rhs, dout)

        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=1e-5)

    tester((5,), (5,), False, False)
    tester((4, 5), (5,), False, False)
    tester((5,), (5, 6), False, False)
    tester((5, 4), (5,), True, False)

    tester((4, 5), (5, 6), False, False)
    tester((4, 5), (6, 5), False, True)
    tester((5, 4), (5, 6), True, False)
    tester((5, 4), (6, 5), True, True)

    tester((2, 3, 4, 5), (5, 6), False, False)
    tester((2, 3, 4, 5), (6, 5), False, True)
    tester((2, 1, 5, 4), (5, 6), True, False)
    tester((2, 1, 5, 4), (6, 5), True, True)
    tester((1, 5, 4), (5, 6), True, False)
    tester((1, 5, 4), (6, 5), True, True)

    tester((4, 5), (2, 3, 5, 6), False, False)
    tester((4, 5), (2, 3, 6, 5), False, True)
    tester((5, 4), (2, 1, 5, 6), True, False)
    tester((5, 4), (2, 1, 6, 5), True, True)
    tester((5, 4), (1, 5, 6), True, False)
    tester((5, 4), (1, 6, 5), True, True)

    tester((1, 4, 5), (1, 5, 6), False, False)
    tester((1, 5, 4), (1, 5, 6), True, False)
    tester((3, 4, 5), (3, 5, 6), False, False)
    tester((3, 5, 4), (3, 6, 5), True, True)

    tester((5, 3, 2, 7, 8), (3, 2, 8, 9), False, False)
    tester((5, 1, 2, 7, 8), (1, 2, 9, 8), False, True)
    tester((5, 3, 2, 8, 7), (3, 1, 8, 9), True, False)
    tester((5, 3, 2, 8, 7), (1, 2, 9, 8), True, True)
    tester((5, 3, 2, 8, 7), (1, 8, 9), True, False)
    tester((5, 3, 1, 8, 7), (1, 9, 8), True, True)

    tester((3, 2, 7, 8), (4, 3, 2, 8, 9), False, False)
    tester((3, 1, 7, 8), (4, 3, 1, 9, 8), False, True)
    tester((3, 1, 8, 7), (4, 3, 2, 8, 9), True, False)
    tester((1, 2, 8, 7), (4, 2, 2, 9, 8), True, True)
    tester((1, 8, 7), (4, 3, 2, 8, 9), True, False)
    tester((1, 8, 7), (4, 3, 1, 9, 8), True, True)


if __name__ == "__main__":
    test_matmul()
