import io

import numpy as np

from megengine.core.ops import builtin as ops
from megengine.core.tensor.core import apply
from megengine.core.tensor.raw_tensor import as_raw_tensor
from megengine.jit import exclude_from_trace, trace


def test_trace():
    for symbolic in [False, True]:

        @trace(symbolic=symbolic)
        def f(x):
            op = ops.Elemwise(mode="negate")
            (y,) = apply(op, x)
            return y

        x = as_raw_tensor([1]).numpy()
        y = f.__wrapped__(as_raw_tensor(x)).numpy()

        for i in range(3):
            np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)


def test_exclude_from_trace():
    for symbolic in [False, True]:

        @trace(symbolic=symbolic)
        def f(x):
            neg = ops.Elemwise(mode="negate")
            (x,) = apply(neg, x)
            with exclude_from_trace():
                if i % 2:
                    (x,) = apply(neg, x)
            (x,) = apply(neg, x)
            return x

        x = as_raw_tensor([1]).numpy()

        for i in range(3):
            y = f.__wrapped__(as_raw_tensor(x)).numpy()
            np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)


def test_print_in_trace():
    for symbolic in [False]:  # cannot read value in symbolic mode

        @trace(symbolic=symbolic)
        def f(x):
            nonlocal buf
            neg = ops.Elemwise(mode="negate")
            (x,) = apply(neg, x)
            buf = x.numpy()
            (x,) = apply(neg, x)
            return x

        buf = None
        x = as_raw_tensor([1]).numpy()

        for i in range(3):
            y = f.__wrapped__(as_raw_tensor(x)).numpy()
            z = buf
            buf = None
            np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)
            np.testing.assert_equal(z, buf)


def test_dump():
    @trace(symbolic=True, capture_as_const=True)
    def f(x):
        op = ops.Elemwise(mode="negate")
        (y,) = apply(op, x)
        return y

    x = as_raw_tensor([1]).numpy()
    y = f.__wrapped__(as_raw_tensor(x)).numpy()

    for i in range(3):
        np.testing.assert_equal(f(as_raw_tensor(x)).numpy(), y)

    file = io.BytesIO()
    f.dump(file)
