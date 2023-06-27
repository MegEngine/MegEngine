import numpy as np

import megengine as mge
import megengine.functional as F
import megengine.jit as jit
import megengine.tensor as tensor
from megengine.autodiff.grad_manager import GradManager


def test_elemwise():
    np.random.seed(123)
    mge.random.seed(123)

    def tester(felemwise, *inp_shapes, backward=True, dtype=None, atol=1e-5):
        dtype = dtype or np.float32
        inps = [
            tensor(0.1 * np.random.randn(*inp_shape), dtype=dtype)
            for inp_shape in inp_shapes
        ]
        doup = tensor(0.1 * np.random.randn(*felemwise(*inps).shape), dtype=dtype)

        gm = GradManager()

        @jit.trace(without_host=True, use_xla=True)
        def func(inps, doup):
            gm.attach(inps)
            with gm:
                oup = felemwise(*inps)
                if backward:
                    gm.backward(oup, doup)
                    return [oup, *[inp.grad for inp in inps]]
                else:
                    return [oup]

        mge_rsts = func(inps, doup)
        xla_rsts = func(inps, doup)
        for mge_rst, xla_rst in zip(mge_rsts, xla_rsts):
            np.testing.assert_allclose(mge_rst.numpy(), xla_rst.numpy(), atol=atol)

    tester(F.neg, (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.abs, (2, 32, 16), dtype=np.float32, atol=1e-5)
    tester(F.tanh, (4, 16, 3, 1), backward=False, dtype=np.float32, atol=1e-5)
    tester(F.exp, (2, 8), dtype=np.float32, atol=1e-5)
    tester(F.sqrt, (32,), dtype=np.float32, atol=1e-5)
    tester(F.log, (8, 8, 16), dtype=np.float32, atol=1e-5)
    tester(F.relu, (1,), dtype=np.float32, atol=1e-5)
    tester(F.gelu, (4, 16, 12, 12), dtype=np.float32, atol=2e-5)

    tester(F.add, (4, 16, 12, 12), (4, 16, 12, 12), dtype=np.float32, atol=1e-5)
    tester(F.sub, (4, 16, 12, 12), (4, 16, 1, 1), dtype=np.float32, atol=1e-5)
    tester(F.mul, (4, 16, 12, 12), (1, 1, 12, 12), dtype=np.float32, atol=1e-5)
    tester(
        F.div,
        (4, 16, 1, 1),
        (4, 16, 12, 12),
        backward=False,
        dtype=np.float32,
        atol=1e-5,
    )
    tester(F.pow, (4, 1, 12, 12), (1, 16, 12, 12), dtype=np.float32, atol=1e-5)

    tester(
        F.equal, (4, 16, 12, 12), (1, 1), backward=False, dtype=np.float32, atol=1e-5
    )
    tester(
        F.not_equal,
        (4, 16, 12, 12),
        (4, 16, 1, 1),
        backward=False,
        dtype=np.float32,
        atol=1e-5,
    )
    tester(
        F.greater,
        (4, 16, 1, 1),
        (4, 16, 12, 12),
        backward=False,
        dtype=np.float32,
        atol=1e-5,
    )
    tester(
        F.greater_equal,
        (16, 1, 1),
        (4, 16, 12, 12),
        backward=False,
        dtype=np.float32,
        atol=1e-5,
    )
    tester(
        F.less,
        (4, 16, 12, 1),
        (4, 16, 12, 12),
        backward=False,
        dtype=np.float32,
        atol=1e-5,
    )
    tester(
        F.less_equal,
        (1, 1, 12, 12),
        (4, 16, 12, 12),
        backward=False,
        dtype=np.float32,
        atol=1e-5,
    )


if __name__ == "__main__":
    test_elemwise()
