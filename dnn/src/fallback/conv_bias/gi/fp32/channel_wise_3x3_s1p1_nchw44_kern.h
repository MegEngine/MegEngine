#pragma once

#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {
namespace channel_wise_nchw44_float {

template <BiasMode bias_mode, typename Op>
void do_conv_kern_3x3_stride1_padding1(
        const float* src, float* dst, const float* filter, const float* bias, int H,
        int W);

}  // namespace channel_wise_nchw44_float
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
