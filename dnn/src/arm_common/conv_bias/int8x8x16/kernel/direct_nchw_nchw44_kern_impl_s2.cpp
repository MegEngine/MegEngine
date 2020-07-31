/**
 * \file
 * dnn/src/arm_common/conv_bias/int8x8x16/kernel/direct_nchw_nchw44_kern_impl_s2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/conv_bias/int8x8x16/kernel/direct_nchw_nchw44_kern_impl.h"
INSTANCE_CONV(2, 2);
INSTANCE_CONV(3, 2);
INSTANCE_CONV(5, 2);
INSTANCE_CONV(7, 2);

// vim: syntax=cpp.doxygen