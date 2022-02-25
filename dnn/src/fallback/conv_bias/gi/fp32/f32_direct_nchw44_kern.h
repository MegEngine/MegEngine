#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/opr_impl.h"
namespace megdnn {
namespace fallback {
namespace conv_bias {

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_direct_fp32_nchw44(
        const float* src, const float* filter, const float* bias, float*, float* dst,
        const int oc, const int ic, const int ih, const int iw, const int oh,
        const int oh_block, const int ow, const Op& op, const int, const int);
template <int stride>
void pack_src_fp32_nchw44(
        float* sptr_base, const float* sptr_origin, const int, const int pw,
        const int pad_right, const int ih, const int iw, const int iw2,
        const int pad_top, const int pad_bottom, const int ic, const int ic_stride);

}  // namespace conv_bias
}  // namespace fallback
}  // namespace megdnn
