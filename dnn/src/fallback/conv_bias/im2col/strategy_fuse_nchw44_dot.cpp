/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_fuse_nchw44_dot.cpp
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

#define PACKB_ONELINE()                                                       \
    int out_index = 0;                                                        \
    outptr = output_base;                                                     \
    for (; out_index + 11 < block_size; out_index += 12) {                    \
        std::memcpy(outptr, tmp_output, 48);                                  \
        outptr += ksize12;                                                    \
        tmp_output += 12;                                                     \
    }                                                                         \
                                                                              \
    outptr = output_base4;                                                    \
    for (; out_index + 3 < block_size; out_index += 4) {                      \
        std::memcpy(outptr, tmp_output, 16);                                  \
        outptr += ksize4;                                                     \
        tmp_output += 4;                                                      \
    }                                                                         \
                                                                              \
    if (out_index < block_size) {                                             \
        uint32_t zerobuffer[4] = {0};                                         \
        size_t out_remain = std::min(block_size - out_index, 4);              \
        std::memcpy(outptr, tmp_output, out_remain * sizeof(uint32_t));       \
        outptr += out_remain;                                                 \
        std::memcpy(outptr, zerobuffer, (4 - out_remain) * sizeof(uint32_t)); \
    }                                                                         \
    output_base += 12;                                                        \
    output_base4 += 4;

#define STOR_IM2COL_DST()                   \
    output0[count] = uint32_src[index + 0]; \
    output1[count] = uint32_src[index + 1]; \
    output2[count] = uint32_src[index + 2];

#define LOAD_AND_STOR_IM2COL_DST()                        \
    uint32x4_t v_tmp = vld1q_u32(&uint32_src[index + 4]); \
    uint32x4_t v_o1 = vextq_u32(v_o0, v_tmp, 1);          \
    uint32x4_t v_o2 = vextq_u32(v_o0, v_tmp, 2);          \
    vst1q_u32(&output0[count], v_o0);                     \
    vst1q_u32(&output1[count], v_o1);                     \
    vst1q_u32(&output2[count], v_o2);                     \
    v_o0 = v_tmp;

void fuse_packb(const dt_int8* __restrict src, dt_int8* __restrict dst,
                dt_int8* __restrict b_panel, const int OW, const int IC,
                const int IH, const int IW,
                const int cur_index, const int block_size) {
    int start_h = cur_index / OW;
    int cur_remain_w = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_remain_w = (cur_index + block_size) % OW;
    bool same_line = start_h == end_h ? true : false;
    size_t newIC = IC / 4;
    const uint32_t* uint32_src =
            static_cast<const uint32_t*>(static_cast<const void*>(src));
    uint32_t* output = static_cast<uint32_t*>(static_cast<void*>(dst));
    uint32_t* b_output = static_cast<uint32_t*>(static_cast<void*>(b_panel));
    const int packed_k = newIC * 3 * 3;
    const int ksize12 = packed_k * 12 * sizeof(dt_int8);
    const int ksize4 = packed_k * 4 * sizeof(dt_int8);
    uint32_t* outptr = b_output;
    uint32_t* output_base = b_output;
    uint32_t* output_base4 = b_output + block_size / 12 * ksize12;
    constexpr int FH = 3;
    if (same_line) {
        rep(ic, newIC) {
            rep(fh, FH) {
                size_t count = 0;
                size_t index = 0;
                int w = cur_remain_w;
                index = (ic * IH + (start_h + fh)) * IW + w;
                for (; w + 3 < end_remain_w; w += 4) {
                    vst1q_u32(&output[count], vld1q_u32(&uint32_src[index]));
                    count += 4;
                    index += 4;
                }
                for (; w < end_remain_w; w++) {
                    output[count++] = uint32_src[index++];
                }
                output[count++] = uint32_src[index];
                output[count++] = uint32_src[index + 1];
                for (int i = 0; i < 3; i++) {
                    const uint32_t* tmp_output = output + i;
                    PACKB_ONELINE();
                }
            }
        }
    } else {
        rep(ic, newIC) {
            rep(fh, FH) {
                size_t count = 0;
                size_t index = 0;
                uint32_t* output0 = output;
                uint32_t* output1 = output + block_size;
                uint32_t* output2 = output1 + block_size;
                int w = cur_remain_w;
                index = (ic * IH + (start_h + fh)) * IW + w;
                uint32x4_t v_o0 = vld1q_u32(&uint32_src[index]);
                for ( ; w + 3 < OW; w += 4) {
                    LOAD_AND_STOR_IM2COL_DST();
                    count += 4;
                    index += 4;
                }

                for (; w < OW; w++) {
                    STOR_IM2COL_DST();
                    count++;
                    index++;
                }

                for (int h = start_h + 1; h < end_h; h++) {
                    int ow = 0;
                    index = (ic * IH + (h + fh)) * IW + ow;
                    v_o0 = vld1q_u32(&uint32_src[index]);
                    for (; ow + 3 < OW; ow += 4) {
                        LOAD_AND_STOR_IM2COL_DST();
                        count += 4;
                        index += 4;
                    }

                    for (; ow < OW; ow++) {
                        STOR_IM2COL_DST();
                        count++;
                        index++;
                    }
                }

                index = (ic * IH + (end_h + fh)) * IW;
                w = 0;
                v_o0 = vld1q_u32(&uint32_src[index]);
                for ( ; w + 3 < end_remain_w; w+=4) {
                    LOAD_AND_STOR_IM2COL_DST();
                    count+=4;
                    index+=4;
                }
                for ( ; w < end_remain_w; w++) {
                    STOR_IM2COL_DST();
                    count++;
                    index++;
                }

                for (int k = 0; k < 3; k++) {
                    const uint32_t* tmp_output = output + k * block_size;
                    PACKB_ONELINE();
                }
            }
        }
    }
}
#undef PACKB_ONELINE
#undef STOR_IM2COL_DST
#undef LOAD_AND_STOR_IM2COL_DST
}  // namespace

template <typename op_ctype, typename op_dtype,
          megdnn::PostprocessMode postprocess_mode>
void StrategyFuse8x12x4Nchw44Dot<op_ctype, op_dtype, postprocess_mode>::
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
    megdnn_assert(ic % 4 == 0, "nchw44_dot with ic is not of time 4");

    int8_t* im2col_dst = static_cast<int8_t*>(
            bundle_thread.get(THREAD_BUNDLE_IM2COL_INDEX));

    fuse_packb(src2, im2col_dst, b_panel, ow, ic, ih, iw, sparam.ohw_cur_index,
               sparam.output_block_size);
}

namespace megdnn {

template class StrategyFuse8x12x4Nchw44Dot<dt_qint32, dt_qint8,
                                        megdnn::PostprocessMode::QUANTIZED>;
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
