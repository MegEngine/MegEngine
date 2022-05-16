#pragma once
#include "src/fallback/resize/gi/helper.h"

namespace megdnn {
namespace fallback {

void resize_linear_upsample2_nchw44_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param);

void resize_nearest_upsample2_nchw44_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param);

}  // namespace fallback
}  // namespace megdnn
