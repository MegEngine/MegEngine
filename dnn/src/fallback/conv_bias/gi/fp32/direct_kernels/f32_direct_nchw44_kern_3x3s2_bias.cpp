/**
 * \file
 * dnn/src/fallback/conv_bias/gi/fp32/direct_kernels/f32_direct_nchw44_kern_3x3s2_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/fallback/conv_bias/gi/fp32/direct_kernels/f32_direct_nchw44_kern_common_s2.h"
INSTANTIATION_CONV_S2_BIAS(3);
// vim: syntax=cpp.doxygen
