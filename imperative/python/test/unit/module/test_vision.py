import time

import numpy as np
import pytest

from megengine import Tensor
from megengine.module import (
    AdditiveGaussianNoise,
    AdditiveLaplaceNoise,
    AdditivePoissonNoise,
)


@pytest.mark.parametrize(
    "cls", [AdditiveGaussianNoise, AdditiveLaplaceNoise, AdditivePoissonNoise]
)
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize(
    "shape, format",
    [
        ((128, 3, 160, 160), "default"),
        ((128, 160, 160, 3), "nhwc"),
        ((128, 3, 160, 160), "nchw"),
    ],
)
@pytest.mark.parametrize("seed", [1024, None])
def test_AdditiveNoise(cls, per_channel, shape, format, seed):
    if not per_channel and format == "default":
        return

    input_tensor = Tensor(
        np.random.random(shape), np.float32, device="xpux", format=format
    )

    aug = cls(per_channel=per_channel, seed=seed)
    aug_data = aug(input_tensor)
    if seed is not None:  # fix rng seed
        aug_ref = cls(per_channel=per_channel, seed=seed)
        aug_data_ref = aug_ref(input_tensor)
        np.testing.assert_allclose(aug_data, aug_data_ref)
