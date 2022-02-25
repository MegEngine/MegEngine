#pragma once
#include "src/arm_common/resize/opr_impl.h"

namespace megdnn {
namespace arm_common {

void resize_linear_upsample2_nchw44_fp32(
        const ResizeImpl::KernParam<float>& kern_param);

void resize_nearest_upsample2_nchw44_fp32(
        const ResizeImpl::KernParam<float>& kern_param);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void resize_linear_upsample2_nchw88_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param);

void resize_nearest_upsample2_nchw88_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param);

#endif

}  // namespace arm_common
}  // namespace megdnn
