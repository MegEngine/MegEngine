#include "src/arm_common/conv_bias/opr_impl.h"
#if MGB_ENABLE_DOT
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace conv_bias {

#define KERN(stride, i)                                                     \
    template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op> \
    void conv_direct_##stride##_##i##x##i##_int8_dot(                       \
            const int8_t* src, const int8_t* filter, const int32_t* bias,   \
            int32_t* temp, int8_t* dst, const size_t IH, const size_t IW,   \
            const size_t OH, const size_t OW, const Op& op);

KERN(stride1, 2)
KERN(stride1, 3)
KERN(stride1, 5)
KERN(stride1, 7)

KERN(stride2, 2)
KERN(stride2, 3)
KERN(stride2, 5)
KERN(stride2, 7)

#undef KERN

}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
