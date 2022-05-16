/**
 * \file dnn/src/arm_common/resize/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/resize/opr_impl.h"
#include "src/arm_common/handle.h"
#include "src/arm_common/resize/direct_nchwxx.h"
#include "src/arm_common/resize/resize_cv.h"
#include "src/arm_common/resize/upsample2_nchw.h"
#include "src/arm_common/resize/upsample2_nchwxx.h"
#include "src/arm_common/simd_macro/marm_neon.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_resize)

namespace megdnn {
namespace arm_common {

void ResizeImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);

    bool is_contiguous = src.layout.is_contiguous() && dst.layout.is_contiguous();
    bool is_dtype_same = src.layout.dtype == dst.layout.dtype;
    bool is_dtype_fp16 =
            DNN_FLOAT16_SELECT(src.layout.dtype == dtype::Float16(), false);
    bool is_dtype_supported = is_dtype_same && is_dtype_fp16;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    bool is_nchw = param().format == param::Resize::Format::NCHW && is_dtype_fp16;
    bool is_nchw88_fp16 =
            param().format == param::Resize::Format::NCHW88 && is_dtype_fp16;
    bool is_upsample2 = src.layout.shape[2] * 2 == dst.layout.shape[2] &&
                        src.layout.shape[3] * 2 == dst.layout.shape[3];
#endif

    bool is_imode_nearest =
            param().imode == param::Resize::InterpolationMode::INTER_NEAREST;
    bool is_imode_linear =
            param().imode == param::Resize::InterpolationMode::INTER_LINEAR;
    bool is_imode_supported = is_imode_nearest || is_imode_linear;

    bool usable = is_contiguous && is_dtype_supported && is_imode_supported;

    if (param().format == param::Resize::Format::NHWC &&
        (src.layout[3] == 1 || src.layout[3] == 3) && is_nhwc_contig_wc(src.layout)) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(resize_cv_exec(src, dst, param().imode));
    } else if (!usable) {
        fallback::ResizeImpl::exec(src, dst, workspace);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    } else if (is_dtype_fp16) {
        auto kern_param = KernParam<dt_float16>::from_tensors(
                param().format, param().imode, src, dst, workspace);
        if (is_nchw88_fp16) {
            if (is_upsample2) {
                if (is_imode_nearest) {
                    MIDOUT_BEGIN(megdnn_arm_resize, midout_iv(6)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_nearest_upsample2_nchw88_fp16(kern_param));
                    }
                    MIDOUT_END();
                } else {
                    megdnn_assert(is_imode_linear, "invalid imode");
                    MIDOUT_BEGIN(megdnn_arm_resize, midout_iv(7)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_linear_upsample2_nchw88_fp16(kern_param));
                    }
                    MIDOUT_END();
                }
            } else {
                if (is_imode_nearest) {
                    MIDOUT_BEGIN(megdnn_arm_resize, midout_iv(8)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_direct_nearest_nchw88_fp16(kern_param));
                    }
                    MIDOUT_END();
                } else {
                    megdnn_assert(is_imode_linear, "invalid imode");
                    MIDOUT_BEGIN(megdnn_arm_resize, midout_iv(9)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_direct_linear_nchw88_fp16(kern_param));
                    }
                    MIDOUT_END();
                }
            }
        } else if (is_nchw) {
            if (is_upsample2) {
                if (is_imode_nearest) {
                    MIDOUT_BEGIN(megdnn_arm_resize, midout_iv(10)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_nearest_upsample2_nchw_fp16(kern_param));
                    }
                    MIDOUT_END();
                } else {
                    megdnn_assert(is_imode_linear, "invalid imode");
                    MIDOUT_BEGIN(megdnn_arm_resize, midout_iv(11)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_linear_upsample2_nchw_fp16(kern_param));
                    }
                    MIDOUT_END();
                }
            } else {
                fallback::ResizeImpl::exec(src, dst, workspace);
            }
        } else {
            fallback::ResizeImpl::exec(src, dst, workspace);
        }
#endif
    } else {
        fallback::ResizeImpl::exec(src, dst, workspace);
    }
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
