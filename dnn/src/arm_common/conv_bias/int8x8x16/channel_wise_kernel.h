#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace channel_wise_nchw44_8x8x16 {

#define KERN(stride, i)                                                              \
    template <BiasMode bias_mode>                                                    \
    void direct_##stride##_##i##x##i##_int8x8x16(                                    \
            const int8_t* src, const int8_t* filter, const int16_t* bias, void* dst, \
            const size_t IH, const size_t IW, const size_t OH, const size_t OW);

KERN(stride1, 2)
KERN(stride1, 3)
KERN(stride1, 5)

KERN(stride2, 2)
KERN(stride2, 3)
KERN(stride2, 5)

#undef KERN

}  // namespace channel_wise_nchw44_8x8x16
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
