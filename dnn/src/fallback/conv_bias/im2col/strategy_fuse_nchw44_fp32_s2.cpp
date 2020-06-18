/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_fuse_nchw44_fp32_s2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/im2col/strategy_base.h"

#if MEGDNN_AARCH64 || MEGDNN_ARMV7
#include <arm_neon.h>

using namespace megdnn;

namespace {

#define PACKB_ONELINE()                                                      \
    int out_index = 0;                                                       \
    outptr = output_base;                                                    \
    for (; out_index + 11 < block_size; out_index += 12) {                   \
        float32x4x4_t v0 = vld4q_f32(tmp_output);                            \
        float32x4x4_t v1 = vld4q_f32(tmp_output + 16);                       \
        float32x4x4_t v2 = vld4q_f32(tmp_output + 32);                       \
        vst1q_f32(outptr, v0.val[0]);                                        \
        vst1q_f32(outptr + 4, v1.val[0]);                                    \
        vst1q_f32(outptr + 8, v2.val[0]);                                    \
        vst1q_f32(outptr + 12, v0.val[1]);                                   \
        vst1q_f32(outptr + 16, v1.val[1]);                                   \
        vst1q_f32(outptr + 20, v2.val[1]);                                   \
        vst1q_f32(outptr + 24, v0.val[2]);                                   \
        vst1q_f32(outptr + 28, v1.val[2]);                                   \
        vst1q_f32(outptr + 32, v2.val[2]);                                   \
        vst1q_f32(outptr + 36, v0.val[3]);                                   \
        vst1q_f32(outptr + 40, v1.val[3]);                                   \
        vst1q_f32(outptr + 44, v2.val[3]);                                   \
        outptr += ksize12;                                                   \
        tmp_output += 48;                                                    \
    }                                                                        \
                                                                             \
    outptr = output_base4;                                                   \
    for (; out_index + 3 < block_size; out_index += 4) {                     \
        float32x4x4_t v0 = vld4q_f32(tmp_output);                            \
        vst1q_f32(outptr, v0.val[0]);                                        \
        vst1q_f32(outptr + 4, v0.val[1]);                                    \
        vst1q_f32(outptr + 8, v0.val[2]);                                    \
        vst1q_f32(outptr + 12, v0.val[3]);                                   \
        outptr += ksize4;                                                    \
        tmp_output += 16;                                                    \
    }                                                                        \
                                                                             \
    if (out_index < block_size) {                                            \
        float zerobuffer[16] = {0};                                          \
        size_t out_remain = std::min(block_size - out_index, 4);             \
        std::memcpy(zerobuffer, tmp_output, out_remain * sizeof(float) * 4); \
        float32x4x4_t v0 = vld4q_f32(zerobuffer);                            \
        vst1q_f32(outptr, v0.val[0]);                                        \
        vst1q_f32(outptr + 4, v0.val[1]);                                    \
        vst1q_f32(outptr + 8, v0.val[2]);                                    \
        vst1q_f32(outptr + 12, v0.val[3]);                                   \
    }                                                                        \
    output_base += 48;                                                       \
    output_base4 += 16;

#define LOAD_AND_STOR_IM2COL_DST()               \
    float32x4_t v1 = vld1q_f32(&src[index + 4]); \
    float32x4_t v2 = vld1q_f32(&src[index + 8]); \
    vst1q_f32(&output0[i], v0);                  \
    vst1q_f32(&output1[i], v1);                  \
    vst1q_f32(&output2[i], v2);                  \
    i += 4;                                      \
    index += 8;                                  \
    v0 = v2;

void fuse_packb(const float* __restrict src, float* __restrict dst,
                float* __restrict b_panel, const int OW, const int IC,
                const int IH, const int IW, const int cur_index,
                const int block_size) {
    int start_h = cur_index / OW;
    int cur_remain_w = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_remain_w = (cur_index + block_size) % OW;
    bool same_line = start_h == end_h ? true : false;
    size_t newIC = IC / 4;
    float* b_output = b_panel;
    const int packed_k = IC * 3 * 3;
    const int ksize12 = packed_k * 12;
    const int ksize4 = packed_k * 4;
    float* outptr = b_output;
    float* output_base = b_output;
    float* output_base4 = b_output + block_size / 12 * ksize12;
    constexpr int FH = 3;
    constexpr int SH = 2;
    constexpr int SW = 2;
    if (same_line) {
        rep(ic, newIC) {
            rep(fh, FH) {
                float* output02 = dst;
                float* output1 = dst + block_size * 4 + 4;
                size_t i = 0;

                size_t index = 4 * (ic * IH * IW + (start_h * SH + fh) * IW +
                                    cur_remain_w * SW);
                for (int w = cur_remain_w; w < end_remain_w; w++) {
                    vst1q_f32(&output02[i], vld1q_f32(&src[index]));
                    vst1q_f32(&output1[i], vld1q_f32(&src[index + 4]));
                    i += 4;
                    index += 8;
                }
                vst1q_f32(&output02[i], vld1q_f32(&src[index]));
                float* output[3];
                output[0] = output02;
                output[1] = output1;
                output[2] = output02 + 4;
                for (int i = 0; i < 3; i++) {
                    const float* tmp_output = output[i];
                    PACKB_ONELINE();
                }
            }
        }
    } else {
        rep(ic, newIC) {
            rep(fh, FH) {
                float* output0 = dst;
                float* output1 = dst + block_size * 4;
                float* output2 = output1 + block_size * 4;
                size_t i = 0;

                size_t index = 4 * (ic * IH * IW + (start_h * SH + fh) * IW +
                                    (cur_remain_w * SW));
                float32x4_t v0 = vld1q_f32(&src[index]);
                for (int w = cur_remain_w; w < OW; w++) {
                    LOAD_AND_STOR_IM2COL_DST();
                }

                for (int h = start_h + 1; h < end_h; h++) {
                    size_t index = 4 * (ic * IH * IW + (h * SH + fh) * IW);
                    v0 = vld1q_f32(&src[index]);
                    rep(ow, OW) { LOAD_AND_STOR_IM2COL_DST(); }
                }

                index = 4 * (ic * IH * IW + (end_h * SH + fh) * IW);
                v0 = vld1q_f32(&src[index]);
                for (int w = 0; w < end_remain_w; w++) {
                    LOAD_AND_STOR_IM2COL_DST();
                }

                for (int i = 0; i < 3; i++) {
                    const float* tmp_output = output0 + i * block_size * 4;
                    PACKB_ONELINE();
                }
            }
        }
    }
}
#undef PACKB_ONELINE
#undef LOAD_AND_STOR_IM2COL_DST
}  // namespace

template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void StrategyFuseXx12x1Nchw44K3x3S2<op_ctype, op_dtype, postprocess_mode>::
        exec_im2col(const WorkspaceBundle& bundle,
                    const WorkspaceBundle& bundle_thread,
                    const StrategyParam& sparam,
                    const fallback::ConvBiasImpl::NCBKernParam& param,
                    fallback::MatrixMulImpl::KernParam /*matmul_param*/,
                    const fallback::MatrixMulImpl::AlgoBase* /*matmul_algo*/) {
    size_t ow = param.osz[1];
    size_t ic = param.filter_meta.icpg;
    size_t ih = param.isz[0] + param.filter_meta.padding[0] * 2;
    size_t iw = param.isz[1] + param.filter_meta.padding[1] * 2;
    size_t input_offset =
            ih * iw * ic *
            (sparam.group_id + param.filter_meta.group * sparam.batch_id) *
            sizeof(float);

    float* src2 = reinterpret_cast<float*>(
            reinterpret_cast<uintptr_t>(bundle.get(BUNDLE_PADDING_INDEX)) +
            input_offset);
    bool is_phpwzero = param.filter_meta.padding[0] == 0 &&
                       param.filter_meta.padding[1] == 0;
    if (is_phpwzero) {
        src2 = const_cast<float*>(
                param.src<float>(sparam.batch_id, sparam.group_id));
    }
    float* b_panel = reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(
            bundle_thread.get(THREAD_BUNDLE_PACKB_INDEX)));
    megdnn_assert(ic % 4 == 0, "nchw44_dot with ic is not of time 4");

    float* im2col_dst =
            static_cast<float*>(bundle_thread.get(THREAD_BUNDLE_IM2COL_INDEX));
    fuse_packb(src2, im2col_dst, b_panel, ow, ic, ih, iw, sparam.ohw_cur_index,
               sparam.output_block_size);
}

namespace megdnn {

template class StrategyFuseXx12x1Nchw44K3x3S2<float, float,
                                        megdnn::PostprocessMode::FLOAT>;
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
