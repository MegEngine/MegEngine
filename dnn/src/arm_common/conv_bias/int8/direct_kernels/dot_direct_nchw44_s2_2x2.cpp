/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw44_s2_2x2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw44_s2.h"
#if MGB_ENABLE_DOT
using namespace megdnn;
using namespace arm_common;

FOR_BIAS(2, 2);

#endif
// vim: syntax=cpp.doxygen
