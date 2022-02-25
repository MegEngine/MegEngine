#pragma once
#include "megdnn/arch.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/gi/intrinsic_helper.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/elemwise_helper/elemwise_op.h"
namespace megdnn {
namespace fallback {
namespace fp32_direct_nchw_nchw44 {

static inline void pack_weight_fp32_nchw_nchw44(
        const float32_t* in_ptr, float32_t* dst_ptr, const int oc, const int kh,
        const int kw, const int ic) {
    constexpr int oc_step = 4;
    const int filter_oc_stride = kh * kw * ic;
    const int filter_ic_stride = kh * kw * oc_step;
    for (int oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const float32_t* in_ptr_oc = in_ptr + oc_idx * filter_oc_stride;
        float32_t* dst_ptr_oc = dst_ptr + oc_idx * filter_oc_stride;
        for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
            for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
                for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                    GI_FLOAT32_t vsrc = GiLoadFloat32(in_ptr_oc);
                    GiStoreFloat32(dst_ptr_oc + ic_idx * filter_ic_stride, vsrc);
                    in_ptr_oc += oc_step;
                }
                dst_ptr_oc += oc_step;
            }
        }
    }
}
template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_direct_fp32_nchw_nchw44(
        const float32_t* src, const float32_t* filter, const float32_t* bias,
        float32_t*, float32_t* dst, const int oc, const int ic, const int ih,
        const int iw, const int oh, const int oh_block, const int ow, const Op& op,
        const int, const int);
}  // namespace fp32_direct_nchw_nchw44

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
