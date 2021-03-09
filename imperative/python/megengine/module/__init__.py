# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from .activation import LeakyReLU, PReLU, ReLU, Sigmoid, Softmax
from .adaptive_pooling import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from .batch_matmul_activation import BatchMatMulActivation
from .batchnorm import BatchNorm1d, BatchNorm2d, SyncBatchNorm
from .concat import Concat
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvRelu2d,
    ConvTranspose2d,
    DeformableConv2d,
    LocalConv2d,
)
from .conv_bn import ConvBn2d, ConvBnRelu2d
from .deformable_psroi_pooling import DeformablePSROIPooling
from .dropout import Dropout
from .elemwise import Elemwise
from .embedding import Embedding
from .identity import Identity
from .linear import Linear
from .module import Module
from .normalization import GroupNorm, InstanceNorm, LayerNorm
from .pooling import AvgPool2d, MaxPool2d
from .quant_dequant import DequantStub, QuantStub
from .sequential import Sequential
