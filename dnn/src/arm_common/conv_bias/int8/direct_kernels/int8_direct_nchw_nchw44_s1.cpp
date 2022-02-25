#include "src/arm_common/conv_bias/int8/direct_kernels/int8_direct_nchw_nchw44_common.h"
#include "src/arm_common/conv_bias/int8/direct_nchw_nchw44_kern.h"

namespace megdnn {
namespace arm_common {
namespace int8_direct_nchw_nchw44 {
/**
 * pack {oc / 4, fh, fw, ic, 4(oc)} to {oc / 4, ic, fh ,fw/4, 4(oc)*4(fw)}
 * pack interleave two adjacent row in filter to one row
 * */
template <>
void pack_nchw44_weight_for_nchw_conv<1>(
        const int8_t* src_ptr, int8_t* dst_ptr, const int ic, const int fh,
        const int fw, const int oc) {
    constexpr int oc_step = 4;
    const int fw2 = round_up(fw, 4);
    const int fw_remain = fw2 - fw;
    const int dst_ic_stride = fh * fw2;
    const int oc_step_stride = fh * fw2 * ic * oc_step;
    static const uint8_t transpose_4x4_idx[16] = {0, 4,  1, 5,  2,  6,  3,  7,
                                                  8, 12, 9, 13, 10, 14, 11, 15};
    uint8x16_t tbl_transpose_4x4 = vld1q_u8(&transpose_4x4_idx[0]);
    rep_step(oc_idx, oc, oc_step) {
        int32_t* dst_temp_ptr =
                reinterpret_cast<int32_t*>(dst_ptr + oc_idx * ic * fh * fw2);
        const int32_t* src_temp_ptr =
                reinterpret_cast<const int32_t*>(src_ptr + oc_idx * ic * fh * fw);
        // transpose ic and pad
        rep(fh_idx, fh) {
            rep(fw_idx, fw) {
                rep(ic_idx, ic) {
                    *(dst_temp_ptr + ic_idx * dst_ic_stride) = *src_temp_ptr;
                    src_temp_ptr++;
                }
                dst_temp_ptr++;
            }
            rep(ic_idx, ic) {
                memset(dst_temp_ptr + ic_idx * dst_ic_stride, 0,
                       sizeof(int8_t) * oc_step * fw_remain);
            }
            dst_temp_ptr += fw_remain;
        }
        // transpose fw oc
        int8_t* trans_dst_temp_ptr =
                reinterpret_cast<int8_t*>(dst_ptr + oc_idx * ic * fh * fw2);

        rep_step(idx, oc_step_stride, 16) {
            int8x16_t temp = vld1q_s8(trans_dst_temp_ptr + idx);
            vst1q_s8(trans_dst_temp_ptr + idx, vqtbl1q_s8(temp, tbl_transpose_4x4));
        }
    }
};

/**
 * pack (ic, h, w) to (ic, h, w * 16)
 * pack interleave two adjacent row in src and repeat 4 times, store to one row
 * */
template <>
void pack_nchw_src_for_nchw44_conv<1>(
        const int8_t* sptr_origin, int8_t* sptr_base, const int ic, const int pad_top,
        const int pad_bottom, const int, const int, const int ih, const int iw,
        const int iw2, const int pw, int8_t* temp_ptr) {
    static uint8_t reorder_idx[16] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3};
    uint8x16_t tbl_idx = vld1q_u8(&reorder_idx[0]);

    constexpr int iw_step = 4;
    constexpr int pack_iw_len = 16;
    const int ic_stride = ih * iw;
    const int iw_with_pad = iw + 2 * pw;
    const int iw_with_pad_end = iw_with_pad / iw_step * iw_step;
    rep(ic_idx, ic) {
        const int8_t* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_base, 0,
               sizeof(int8_t) * iw2 * (ih + pad_top + pad_bottom) * pack_iw_len);
        sptr_base += iw2 * pad_top * pack_iw_len;
        rep(ih_idx, ih) {
            memset(temp_ptr, 0, iw_with_pad * sizeof(int8_t));
            memcpy(temp_ptr + pw, sptr, sizeof(int8_t) * iw);
            for (int iw_idx = 0; iw_idx < iw_with_pad_end; iw_idx += iw_step) {
                int8x16_t src[4];
                int8x16_t dst[4];
                src[0] = vld1q_s8(temp_ptr + iw_idx);
                src[1] = vld1q_s8(temp_ptr + iw_idx + 1);
                src[2] = vld1q_s8(temp_ptr + iw_idx + 2);
                src[3] = vld1q_s8(temp_ptr + iw_idx + 3);
                dst[0] = vqtbl1q_s8(src[0], tbl_idx);
                dst[1] = vqtbl1q_s8(src[1], tbl_idx);
                dst[2] = vqtbl1q_s8(src[2], tbl_idx);
                dst[3] = vqtbl1q_s8(src[3], tbl_idx);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 0, dst[0]);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 16, dst[1]);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 32, dst[2]);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 48, dst[3]);
            }
            for (int iw_idx = iw_with_pad_end; iw_idx < iw_with_pad; ++iw_idx) {
                int8x16_t src = vld1q_s8(temp_ptr + iw_idx);
                int8x16_t dst = vqtbl1q_s8(src, tbl_idx);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len, dst);
            }
            sptr_base += iw2 * pack_iw_len;
            sptr += iw;
        }
        sptr_base += iw2 * pad_bottom * pack_iw_len;
    }
}

#define INSTANCE_CONV_KERN_FUN(stride, filter_size, bias_mode, Op) \
    template struct megdnn::arm_common::int8_direct_nchw_nchw44::  \
            ConvDiectStrideInt8NchwNchw44<bias_mode, Op, filter_size, stride>;

#define INSTANCE_OP_PARAM(stride, filter, bias_mode)                               \
    INSTANCE_CONV_KERN_FUN(                                                        \
            stride, filter, bias_mode, TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    INSTANCE_CONV_KERN_FUN(                                                        \
            stride, filter, bias_mode, ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    INSTANCE_CONV_KERN_FUN(                                                        \
            stride, filter, bias_mode, HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)

#define INSTANCE_BIAS_MODE_PARAM(stride, filter)         \
    INSTANCE_OP_PARAM(stride, filter, BiasMode::NO_BIAS) \
    INSTANCE_OP_PARAM(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS)

#define INSTANCE_CONV_KERN(stride, filter) INSTANCE_BIAS_MODE_PARAM(stride, filter)

}  // namespace int8_direct_nchw_nchw44
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
