/**
 * \file dnn/src/fallback/conv_bias/im2col/strategy_fuse_nchw44_dot_s2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/im2col/strategy_base.h"

#if MEGDNN_ARMV7
#include <arm_neon.h>
using namespace megdnn;
namespace {

#define PACKB_ONELINE()                                                       \
    int out_index = 0;                                                        \
    outptr = output_base;                                                     \
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
    output_base += 4;

#define STOR_IM2COL_DST()                   \
    output0[count] = uint32_src[index];     \
    output1[count] = uint32_src[index + 1]; \
    output2[count] = uint32_src[index + 2]; \
    count++;                                \
    index += SW;

#define LOAD_AND_STOR_IM2COL_DST()                              \
    uint32x4x2_t val_01 = vld2q_u32(&uint32_src[index]);        \
    index += 8;                                                 \
    uint32x4_t val_index8 = vdupq_n_u32(uint32_src[index]);     \
    uint32x4_t val_2 = vextq_u32(val_01.val[0], val_index8, 1); \
    vst1q_u32(&output0[count], val_01.val[0]);                  \
    vst1q_u32(&output1[count], val_01.val[1]);                  \
    vst1q_u32(&output2[count], val_2);                          \
    count += 4;

void fuse_packb(const dt_int8* __restrict src, dt_int8* __restrict dst,
                dt_int8* __restrict b_panel, const int OW, const int IC,
                const int IH, const int IW, const int cur_index,
                const int block_size) {
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
    const int ksize4 = packed_k * 4;
    uint32_t* outptr = b_output;
    uint32_t* output_base = b_output;
    constexpr int FH = 3;
    constexpr int SH = 2;
    constexpr int SW = 2;
    if (same_line) {
        rep(ic, newIC) {
            rep(fh, FH) {
                uint32_t* output02 = output;
                uint32_t* output1 = output + block_size + 1;

                size_t count = 0;
                size_t index = 0;
                int w = cur_remain_w;
                index = (ic * IH + (start_h * SH + fh)) * IW + w * SW;
                for (; w + 3 < end_remain_w; w += 4) {
                    uint32x4x2_t val_01 = vld2q_u32(&uint32_src[index]);
                    vst1q_u32(&output02[count], val_01.val[0]);
                    vst1q_u32(&output1[count], val_01.val[1]);
                    count += 4;
                    index += 8;
                }
                for (; w < end_remain_w; w++) {
                    output02[count] = uint32_src[index + 0];
                    output1[count] = uint32_src[index + 1];
                    count++;
                    index += SW;
                }
                output02[count] = uint32_src[index];
                const uint32_t* output_ptr[3];
                output_ptr[0] = output02;
                output_ptr[1] = output1;
                output_ptr[2] = output02 + 1;
                for (int i = 0; i < 3; i++) {
                    const uint32_t* tmp_output = output_ptr[i];
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
                index = (ic * IH + (SH * start_h + fh)) * IW + SW * w;
                for (; w + 3 < OW; w += 4) {
                    LOAD_AND_STOR_IM2COL_DST()
                }

                for (; w < OW; w++) {
                    STOR_IM2COL_DST()
                }

                for (int h = start_h + 1; h < end_h; h++) {
                    int ow = 0;
                    index = (ic * IH + (SH * h + fh)) * IW;
                    for (; ow + 3 < OW; ow += 4) {
                        LOAD_AND_STOR_IM2COL_DST()
                    }

                    for (; ow < OW; ow++) {
                        STOR_IM2COL_DST()
                    }
                }

                index = (ic * IH + (SH * end_h + fh)) * IW;
                w = 0;
                for (; w + 3 < end_remain_w; w += 4) {
                    LOAD_AND_STOR_IM2COL_DST()
                }

                for (; w < end_remain_w; w++) {
                    STOR_IM2COL_DST()
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
void StrategyFuse8x4x4Nchw44DotK3x3S2<op_ctype, op_dtype, postprocess_mode>::
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
    dt_int8* b_panel = reinterpret_cast<dt_int8*>(reinterpret_cast<uintptr_t>(
            bundle_thread.get(THREAD_BUNDLE_PACKB_INDEX)));
    megdnn_assert(ic % 4 == 0, "nchw44dot_dot with ic is not of time 4");

    int8_t* im2col_dst =
            static_cast<int8_t*>(bundle_thread.get(THREAD_BUNDLE_IM2COL_INDEX));

    fuse_packb(src2, im2col_dst, b_panel, ow, ic, ih, iw, sparam.ohw_cur_index,
               sparam.output_block_size);
}

namespace megdnn {

template class StrategyFuse8x4x4Nchw44DotK3x3S2<dt_qint32, dt_qint8,
                                        megdnn::PostprocessMode::QUANTIZED>;
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
