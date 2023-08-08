import platform

import numpy as np
import pytest

from megengine import Tensor, is_cuda_available
from megengine.functional import mean, zeros
from megengine.module import (
    AdditiveGaussianNoise,
    AdditiveLaplaceNoise,
    AdditivePoissonNoise,
    Emboss,
    LinearContrast,
    Sharpen,
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


@pytest.mark.parametrize("cls", [Emboss, Sharpen])
@pytest.mark.parametrize(
    "shape, format, dtype",
    [
        ((128, 2, 160, 160), "default", np.uint8),
        ((128, 2, 160, 160), "default", np.float32),
    ],
)
@pytest.mark.parametrize(
    "param1, param2", [(0.5, 0.7), (0.6, 0.8), ((0.6, 0.8), (0.6, 0.8)),],
)
@pytest.mark.parametrize("seed", [1024, None])
def test_blur(cls, shape, format, dtype, param1, param2, seed):
    input_array = np.random.randint(0, 255, size=shape).astype(dtype)
    input_tensor = Tensor(input_array, device="xpux", format=format)

    aug = cls(param1, param2, seed=seed)
    aug_data = aug(input_tensor)
    if seed is not None:  # fix rng seed
        aug_ref = cls(param1, param2, seed=seed)
        aug_data_ref = aug_ref(input_tensor)
        np.testing.assert_allclose(aug_data, aug_data_ref)


@pytest.mark.require_ngpu(1)
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize(
    "shape, format, dtype",
    [
        ((128, 2, 160, 160), "default", np.uint8),
        ((128, 2, 160, 160), "default", np.float32),
    ],
)
@pytest.mark.parametrize("param1", [0.6, 0.8, (0.6, 0.8)])
@pytest.mark.parametrize("seed", [1024, None])
def test_LinearContrast(per_channel, shape, format, dtype, param1, seed):
    input_array = np.random.randint(0, 255, size=shape).astype(dtype)
    input_tensor = Tensor(input_array, device="xpux", format=format)

    aug = LinearContrast(param1, per_channel=per_channel, seed=seed)
    aug_data = aug(input_tensor)
    if seed is not None:  # fix rng seed
        aug_ref = LinearContrast(param1, per_channel=per_channel, seed=seed)
        aug_data_ref = aug_ref(input_tensor)
        np.testing.assert_allclose(aug_data, aug_data_ref)
