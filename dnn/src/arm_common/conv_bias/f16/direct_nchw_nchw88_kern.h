#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "src/arm_common/conv_bias/f16/direct_nchw_nchw88_kern_common.h"
#pragma once
namespace megdnn {
namespace arm_common {
namespace fp16_direct_nchw_nchw88 {
//! (OC/8, FH, FW, IC, 8) --> (OC/8, IC, FH, FW, 8)
static inline void pack_weight_fp16_nchw_nchw88(
        const __fp16* in_ptr, __fp16* dst_ptr, const int oc, const int fh, const int fw,
        const int ic) {
    constexpr int oc_step = 8;
    const int ld_ic = fh * fw * oc_step;
    const int ld_oc = ic * fh * fw;

    for (int oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const __fp16* in_ptr_oc = in_ptr + ld_oc * oc_idx;
        __fp16* dst_ptr_oc = dst_ptr + ld_oc * oc_idx;
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            for (int fw_idx = 0; fw_idx < fw; ++fw_idx) {
                for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
                    vst1q_f16(dst_ptr_oc + ic_idx * ld_ic, vld1q_f16(in_ptr_oc));
                    in_ptr_oc += oc_step;
                }
                dst_ptr_oc += oc_step;
            }
        }
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
static void fp16_direct_conv_nchw_nchw88(
        const __fp16* src, const __fp16* filter, const __fp16* bias, __fp16* dst,
        const int oc, const int ic, const int ih, const int iw, const int oh,
        const int oh_block, const int ow, const Op& op) {
    ConvDirectNchwNchw88Fp16<bias_mode, Op, filter_size, stride>::impl(
            src, filter, bias, dst, oc, ic, ih, iw, oh, oh_block, ow, op);
}
}  // namespace fp16_direct_nchw_nchw88
}  // namespace arm_common
}  // namespace megdnn
#endif