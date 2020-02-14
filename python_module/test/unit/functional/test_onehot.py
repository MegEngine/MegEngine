import numpy as np
import pytest

import megengine.functional as F
from megengine import tensor
from megengine.test import assertTensorClose


def test_onehot_low_dimension():
    inp = tensor(np.arange(1, 4, dtype=np.int32))
    out = F.one_hot(inp)

    assertTensorClose(
        out.numpy(), np.eye(4, dtype=np.int32)[np.arange(1, 4, dtype=np.int32)]
    )


def test_onehot_high_dimension():
    arr = np.array(
        [[3, 2, 4, 4, 2, 4, 0, 4, 4, 1], [4, 1, 1, 3, 2, 2, 4, 2, 4, 3]], dtype=np.int32
    )

    inp = tensor(arr)
    out = F.one_hot(inp, 10)

    assertTensorClose(out.numpy(), np.eye(10, dtype=np.int32)[arr])
