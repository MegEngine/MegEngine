/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_fuse_nchw44.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/im2col/strategy_base.h"

#if MEGDNN_AARCH64
#include <arm_neon.h>

using namespace megdnn;

namespace {
#define TRANS_AND_STORE(input0, input1, input2, input3)              \
    {                                                                \
        auto tmp01 = vzipq_s32(input0, input1);                      \
        auto tmp23 = vzipq_s32(input2, input3);                      \
        auto dst0 = vzip1q_s64(vreinterpretq_s64_s32(tmp01.val[0]),  \
                               vreinterpretq_s64_s32(tmp23.val[0])); \
        auto dst1 = vzip2q_s64(vreinterpretq_s64_s32(tmp01.val[0]),  \
                               vreinterpretq_s64_s32(tmp23.val[0])); \
        auto dst2 = vzip1q_s64(vreinterpretq_s64_s32(tmp01.val[1]),  \
                               vreinterpretq_s64_s32(tmp23.val[1])); \
        auto dst3 = vzip2q_s64(vreinterpretq_s64_s32(tmp01.val[1]),  \
                               vreinterpretq_s64_s32(tmp23.val[1])); \
        vst1q_s32(dst, vreinterpretq_s32_s64(dst0));                 \
        vst1q_s32(dst + 4, vreinterpretq_s32_s64(dst1));             \
        vst1q_s32(dst + 8, vreinterpretq_s32_s64(dst2));             \
        vst1q_s32(dst + 12, vreinterpretq_s32_s64(dst3));            \
        dst += 16;                                                   \
    }

#define TRANS_AND_STORE_REMAIN(input0, input1, input2, input3, remain) \
    {                                                                  \
        auto tmp01 = vzipq_s32(input0, input1);                        \
        auto tmp23 = vzipq_s32(input2, input3);                        \
        vdst[0] = vzip1q_s64(vreinterpretq_s64_s32(tmp01.val[0]),      \
                             vreinterpretq_s64_s32(tmp23.val[0]));     \
        vdst[1] = vzip2q_s64(vreinterpretq_s64_s32(tmp01.val[0]),      \
                             vreinterpretq_s64_s32(tmp23.val[0]));     \
        vdst[2] = vzip1q_s64(vreinterpretq_s64_s32(tmp01.val[1]),      \
                             vreinterpretq_s64_s32(tmp23.val[1]));     \
        vdst[3] = vzip2q_s64(vreinterpretq_s64_s32(tmp01.val[1]),      \
                             vreinterpretq_s64_s32(tmp23.val[1]));     \
        for (size_t i = 0; i < remain; i++) {                          \
            vst1q_s32(dst + i * 4, vreinterpretq_s32_s64(vdst[i]));    \
        }                                                              \
        dst += 16;                                                     \
    }

void optimize_fuse_im2col_packB(dt_int8* src, size_t ic, size_t iw, size_t ih,
                                size_t curr_iw, size_t curr_ih, dt_int8* dst_ptr) {
    int* src_line0 =
            reinterpret_cast<int*>(src + curr_ih * iw * 4 + curr_iw * 4);
    int* src_line1 =
            reinterpret_cast<int*>(src + (curr_ih + 1) * iw * 4 + curr_iw * 4);
    int* src_line2 =
            reinterpret_cast<int*>(src + (curr_ih + 2) * iw * 4 + curr_iw * 4);
    int* dst = reinterpret_cast<int*>(dst_ptr);
    int32x4_t input[12];
    int remain = 0;
    for (size_t c = 0; c < ic; c++) {
        input[remain] = vld1q_s32(src_line0);
        input[remain + 1] = vld1q_s32(src_line0 + 1);
        input[remain + 2] = vld1q_s32(src_line0 + 2);
        input[remain + 3] = vld1q_s32(src_line1);
        input[remain + 4] = vld1q_s32(src_line1 + 1);
        input[remain + 5] = vld1q_s32(src_line1 + 2);
        input[remain + 6] = vld1q_s32(src_line2);
        input[remain + 7] = vld1q_s32(src_line2 + 1);
        input[remain + 8] = vld1q_s32(src_line2 + 2);
        TRANS_AND_STORE(input[0], input[1], input[2], input[3]);
        TRANS_AND_STORE(input[4], input[5], input[6], input[7]);
        if (remain == 3) {
            TRANS_AND_STORE(input[8], input[9], input[10], input[11]);
            remain = 0;
        } else {
            for (int i = 0; i <= remain; i++) {
                input[i] = input[8 + i];
            }
            remain++;
        }
        src_line0 += ih * iw;
        src_line1 += ih * iw;
        src_line2 += ih * iw;
    }
    //! pad remain to 4
    if (remain > 0) {
        TRANS_AND_STORE(input[0], input[1], input[2], input[3]);
    }
}

void naive_fuse_im2col_packB(dt_int8* src, size_t ic, size_t iw, size_t ih,
                             size_t curr_iw, size_t curr_ih, size_t num_point,
                             size_t ow, dt_int8* dst_ptr) {
    megdnn_assert(num_point <= 4_z,
                  "fuse im2col and packB of 4x4x16 num_point must less than 4");
    int* src_line0 = reinterpret_cast<int*>(src + curr_ih * iw * 4);
    int* src_line1 = reinterpret_cast<int*>(src + (curr_ih + 1) * iw * 4);
    int* src_line2 = reinterpret_cast<int*>(src + (curr_ih + 2) * iw * 4);
    int remain = 0;
    int out[9][4] = {{0}};
    int32x4_t input[12];
    int* dst = reinterpret_cast<int*>(dst_ptr);
    for (size_t c = 0; c < ic; c++) {
        //! Read int buffer out
        size_t index = 0, w = curr_iw, dalta_h = 0;
        while (index < num_point) {
            int* src_next_line0 = src_line0 + dalta_h * iw;
            int* src_next_line1 = src_next_line0 + iw;
            int* src_next_line2 = src_next_line1 + iw;
            for (; index < num_point && w < ow; index++, w++) {
                out[0][index] = src_next_line0[w];
                out[1][index] = src_next_line0[w + 1];
                out[2][index] = src_next_line0[w + 2];
                out[3][index] = src_next_line1[w];
                out[4][index] = src_next_line1[w + 1];
                out[5][index] = src_next_line1[w + 2];
                out[6][index] = src_next_line2[w];
                out[7][index] = src_next_line2[w + 1];
                out[8][index] = src_next_line2[w + 2];
            }
            //! next line
            w = 0;
            dalta_h += 1;
        }
        //! load int vector
        input[remain] = vld1q_s32(out[0]);
        input[remain + 1] = vld1q_s32(out[1]);
        input[remain + 2] = vld1q_s32(out[2]);
        input[remain + 3] = vld1q_s32(out[3]);
        input[remain + 4] = vld1q_s32(out[4]);
        input[remain + 5] = vld1q_s32(out[5]);
        input[remain + 6] = vld1q_s32(out[6]);
        input[remain + 7] = vld1q_s32(out[7]);
        input[remain + 8] = vld1q_s32(out[8]);
        int64x2_t vdst[4];
        TRANS_AND_STORE_REMAIN(input[0], input[1], input[2], input[3], num_point);
        TRANS_AND_STORE_REMAIN(input[4], input[5], input[6], input[7], num_point);
        if (remain == 3) {
            TRANS_AND_STORE_REMAIN(input[8], input[9], input[10], input[11],
                                   num_point);
            remain = 0;
        } else {
            for (int i = 0; i <= remain; i++) {
                input[i] = input[8 + i];
            }
            remain++;
        }
        src_line0 += ih * iw;
        src_line1 += ih * iw;
        src_line2 += ih * iw;
    }
    //! pad remain to 4
    if (remain > 0) {
        int64x2_t vdst[4];
        TRANS_AND_STORE_REMAIN(input[0], input[1], input[2], input[3],
                               num_point);
    }
}
}  // namespace

template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void StrategyFuse4x4x16Nchw44<op_ctype, op_dtype, postprocess_mode>::
        exec_im2col(const WorkspaceBundle& bundle,
                    const WorkspaceBundle& bundle_thread,
                    const StrategyParam& sparam,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernParam,
                    const fallback::MatrixMulImpl::AlgoBase*) {
    size_t ow = param.osz[1];
    size_t ic = param.filter_meta.icpg;
    size_t ih = param.isz[0] + param.filter_meta.padding[0] * 2;
    size_t iw = param.isz[1] + param.filter_meta.padding[1] * 2;
    constexpr static size_t pack_size = 4;
    size_t input_offset =
            ih * iw * ic *
            (sparam.group_id + param.filter_meta.group * sparam.batch_id) *
            sizeof(dt_int8);

    dt_int8* src2 = reinterpret_cast<dt_int8*>(
            reinterpret_cast<uintptr_t>(bundle.get(BUNDLE_PADDING_INDEX)) +
            input_offset);
    bool is_phpwzero = param.filter_meta.padding[0] == 0 &&
                       param.filter_meta.padding[1] == 0;
    if (is_phpwzero) {
        src2 = const_cast<dt_int8*>(
                param.src<dt_int8>(sparam.batch_id, sparam.group_id));
    }
    dt_int8* b_panel =
            reinterpret_cast<dt_int8*>(reinterpret_cast<uintptr_t>(
                    bundle_thread.get(THREAD_BUNDLE_PACKB_INDEX)));
    megdnn_assert(ic % 4 == 0, "nchw44 with ic is not of time 4");
    const int packed_k = (ic * 3 * 3) / pack_size;
    const int ksize4 = round_up<int>(packed_k, 4) * 16 * sizeof(dt_int8);
    size_t out_size = sparam.output_block_size;
    size_t curr_index = sparam.ohw_cur_index;
    size_t curr_ih = curr_index / ow;
    size_t curr_iw = curr_index % ow;
    size_t out_index = 0;
    while (out_index < out_size) {
        for (; curr_iw + 3 < ow && out_index + 3 < out_size;
             curr_iw += 4, out_index += 4) {
            dt_int8* dst = b_panel + (out_index / 4) * ksize4;
            optimize_fuse_im2col_packB(src2, ic / 4, iw, ih, curr_iw, curr_ih,
                                       dst);
        }
        if (curr_iw < ow && out_index < out_size) {
            size_t out_remain = std::min(out_size - out_index, 4_z);
            size_t remain_point_this_line = std::min(ow - curr_iw, out_remain);
            size_t start_point_next_line =
                    (out_remain - remain_point_this_line) % ow;
            size_t pass_lines = (out_remain - remain_point_this_line) / ow;
            dt_int8* dst = b_panel + (out_index / 4) * ksize4;
            naive_fuse_im2col_packB(src2, ic / 4, iw, ih, curr_iw, curr_ih,
                                    out_remain, ow, dst);
            out_index += out_remain;
            curr_iw = start_point_next_line;
            curr_ih += (pass_lines + 1);
        } else {
            curr_iw = 0;
            curr_ih++;
        }
    }
}
#undef TRANS_AND_STORE_REMAIN
#undef TRANS_AND_STORE



namespace megdnn {

template class StrategyFuse4x4x16Nchw44<dt_qint32, dt_qint8,
                                        megdnn::PostprocessMode::QUANTIZED>;
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
