#pragma once

#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {
namespace channel_wise_nchw44_float {

#define KERN(stride, i)                                                           \
    template <BiasMode bias_mode, typename Op>                                    \
    void do_conv_kern_##stride##_##i##x##i(                                       \
            const float* src, const float* filter, const float* bias, float* dst, \
            const size_t IH, const size_t IW, const size_t OH, const size_t OW,   \
            const size_t PH, const size_t PW);

KERN(stride1, 2)
KERN(stride1, 3)
KERN(stride1, 5)

KERN(stride2, 2)
KERN(stride2, 3)
KERN(stride2, 5)

#undef KERN

}  // namespace channel_wise_nchw44_float
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
