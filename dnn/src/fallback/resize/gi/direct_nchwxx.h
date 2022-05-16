#pragma once
#include "src/fallback/resize/gi/helper.h"
#include "src/fallback/resize/opr_impl.h"

namespace megdnn {
namespace fallback {

void resize_direct_linear_nchw44_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param);

void resize_direct_nearest_nchw44_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param);

}  // namespace fallback
}  // namespace megdnn
