import platform

import numpy as np
import pytest

from megengine import Tensor, is_cuda_available
from megengine.functional import clip, mean, zeros
from megengine.module import (
    AdditiveGaussianNoise,
    AdditiveLaplaceNoise,
    AdditivePoissonNoise,
    Cutmix,
    Emboss,
    LinearContrast,
    Mixup,
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


def test_mixup():
    input1 = Tensor(np.random.random((4, 3, 16, 16)), np.float32)
    input2 = Tensor(np.random.random((4, 3, 16, 16)), np.float32)
    label1 = Tensor(np.random.randint(0, 10, size=(4,)))
    label2 = Tensor(np.random.randint(0, 10, size=(4,)))
    M = Mixup(3)
    aug_res = M(input1, label1, input2, label2)
    lamb = M.lamb
    for i in range(input1.shape[0]):
        inputi_1 = input1[i : i + 1]
        inputi_2 = input2[i : i + 1]
        labeli_1 = label1[i : i + 1]
        labeli_2 = label2[i : i + 1]
        aug_inpi_ref = inputi_1 * lamb[i] + inputi_2 * (1 - lamb[i])
        aug_labeli_ref = labeli_1 * lamb[i] + labeli_2 * (1 - lamb[i])
        np.testing.assert_allclose(aug_res[0][i : i + 1], aug_inpi_ref)
        np.testing.assert_allclose(aug_res[1][i : i + 1], aug_labeli_ref)


def test_cutmix():
    input1 = Tensor(np.random.random((2, 2, 16, 16)), np.float32)
    input2 = Tensor(np.random.random((2, 2, 16, 16)), np.float32)
    label1 = Tensor(np.random.randint(0, 10, size=(2,)))
    label2 = Tensor(np.random.randint(0, 10, size=(2,)))
    H, W = input1.shape[-2:]
    M = Cutmix(3)
    aug_res = M(input1, label1, input2, label2)
    cx = M.cx
    cy = M.cy
    cut_h = M.cut_h
    cut_w = M.cut_w
    bbx1 = clip(cx - cut_h // 2, 0, H)
    bbx2 = clip(cx + cut_h // 2, 0, H)
    bby1 = clip(cy - cut_w // 2, 0, W)
    bby2 = clip(cy + cut_w // 2, 0, W)
    for i in range(input1.shape[0]):
        bbx1_i = int(bbx1[i])
        bbx2_i = int(bbx2[i])
        bby1_i = int(bby1[i])
        bby2_i = int(bby2[i])
        input1[i : i + 1, :, bbx1_i:bbx2_i, bby1_i:bby2_i] = input2[
            i : i + 1, :, bbx1_i:bbx2_i, bby1_i:bby2_i
        ]
        np.testing.assert_allclose(aug_res[0][i : i + 1], input1[i : i + 1])
