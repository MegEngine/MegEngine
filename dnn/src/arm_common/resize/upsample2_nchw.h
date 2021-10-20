/**
 * \file dnn/src/arm_common/resize/upsample2_nchw.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "src/arm_common/resize/opr_impl.h"

namespace megdnn {
namespace arm_common {

void resize_linear_upsample2_nchw_fp32(const ResizeImpl::KernParam<float>& kern_param);

void resize_nearest_upsample2_nchw_fp32(const ResizeImpl::KernParam<float>& kern_param);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void resize_linear_upsample2_nchw_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param);

void resize_nearest_upsample2_nchw_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param);

#endif

}  // namespace arm_common
}  // namespace megdnn
