/**
 * \file dnn/src/aarch64/conv_bias/quint8/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/conv_bias/quint8/algos.h"
#include "src/aarch64/conv_bias/quint8/strategy.h"
#include "src/aarch64/matrix_mul/quint8_dot/gemv.h"
#include "src/aarch64/matrix_mul/quint8_dot/strategy.h"
#include "src/arm_common/convolution/img2col_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/matrix_mul/gemm_impl.h"

#include "midout.h"

MIDOUT_DECL(megdnn_aarch64_conv_bias_quint8_gemm)

using namespace megdnn;
using namespace aarch64;
using megdnn::arm_common::HSwishOp;
using megdnn::arm_common::ReluOp;
using megdnn::arm_common::TypeCvtOp;

/* ===================== matrix mul algo ===================== */

bool ConvBiasImpl::AlgoQU8MatrixMul::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    auto&& fm = param.filter_meta;
    return param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
           param.dst_type.enumv() == DTypeEnum::Quantized8Asymm &&
           fm.format == param::ConvBias::Format::NCHW && fm.spatial_ndim == 2 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
           //! As postprocess, the bias is not contigous read, make the
           //! performance bad, so we do not process it in fused kernel
           param.bias_mode != BiasMode::BIAS &&
           //! This algo is only support single thread
           param.nr_threads == 1_z;
}

WorkspaceBundle ConvBiasImpl::AlgoQU8MatrixMul::get_bundle(
        const NCBKernSizeParam& param) {
    UNPACK_CONV_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(N);
    auto IW2 = IH + 2 * PH;
    auto IH2 = IW + 2 * PW;
    bool can_matrix_mul_direct =
            (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0);
    // temp space to store padding-free src (with 16 extra int8)
    // temp space to store unrolled matrix (with 16 extra int8)
    // workspace for matrix mul opr
    size_t part0, part1, part2;
    if (can_matrix_mul_direct) {
        part0 = part1 = 0;
    } else {
        part0 = (IC * IH2 * IW2 + 16) * sizeof(uint8_t);
        part1 = (IC * FH * FW * OH * OW + 16) * sizeof(uint8_t);
    }
    {
        size_t M = OC;
        size_t K = IC * FH * FW;
        size_t N = OH * OW;

#define DISPATCH_GEMM_STRATEGY(_gemm, _gemm_midout_enum, _bias,              \
                               _bias_midout_enum, _nonline,                  \
                               _nonline_midout_enum)                         \
    MIDOUT_BEGIN(megdnn_aarch64_conv_bias_quint8_gemm, 0, _gemm_midout_enum, \
                 _bias_midout_enum, _nonline_midout_enum) {                  \
        matmul::gemm_##_gemm##_##_bias##_##_nonline strategy(                \
                M, N, K, param.filter_type, param.src_type, param.dst_type); \
        part2 = megdnn::matmul::GemmInterleaved<                             \
                        matmul::gemm_##_gemm##_##_bias##_##_nonline>(        \
                        M, N, K, false, false, strategy)                     \
                        .get_workspace_size();                               \
    }                                                                        \
    MIDOUT_END()

        DISPATCH_GEMM_BIAS(u8_8x8, 0)
#undef DISPATCH_GEMM_STRATEGY
    }
    return {nullptr, {part0, part1, part2}};
}

void ConvBiasImpl::AlgoQU8MatrixMul::kimpl(const NCBKernParam& param,
                                           const NCBKernIndex& ncb_index) {
    auto is_xcorr = !param.filter_meta.should_flip;
    UNPACK_CONV_NCB_KERN_SIZES(param);
    auto bundle = get_bundle(param);
    bundle.set(param.workspace_ptr);
    auto IH2 = IH + 2 * PH;
    auto IW2 = IW + 2 * PW;
    size_t group_id = ncb_index.ndrange_id[0];
    uint8_t src_zp = param.src_type.param<dtype::Quantized8Asymm>().zero_point;
    // workspace = tmp..src2
    for (size_t n = 0; n < N; ++n) {
        uint8_t* src = const_cast<uint8_t*>(param.src<uint8_t>(n, group_id));
        uint8_t* filter = const_cast<uint8_t*>(param.filter<uint8_t>(group_id));
        uint8_t* dst = static_cast<uint8_t*>(param.dst<uint8_t>(n, group_id));
        int32_t* bias = const_cast<int32_t*>(param.bias<int32_t>(n, group_id));

        uint8_t *B, *src2;
        if (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0) {
            // special case: 1x1
            B = const_cast<uint8_t*>(src);
        } else {
            src2 = static_cast<uint8_t*>(bundle.get(0));
            // copy src to src2;
            uint8_t* src2_ptr = src2;
            const uint8_t* src_ptr = src;
            rep(ic, IC) {
                if (PH != 0) {
                    std::memset(src2_ptr, src_zp, sizeof(uint8_t) * PH * IW2);
                    src2_ptr += PH * IW2;
                }
                rep(ih, IH) {
                    if (PW != 0)
                        rep(pw, PW) { *(src2_ptr++) = src_zp; }
                    std::memcpy(src2_ptr, src_ptr, sizeof(uint8_t) * IW);
                    src2_ptr += IW;
                    src_ptr += IW;
                    if (PW != 0)
                        rep(pw, PW) { *(src2_ptr++) = src_zp; }
                }
                if (PH != 0) {
                    std::memset(src2_ptr, src_zp, sizeof(uint8_t) * PH * IW2);
                    src2_ptr += PH * IW2;
                }
            }

            B = static_cast<uint8_t*>(bundle.get(1));
            if (SH == 1 && SW == 1) {
                if (is_xcorr)
                    img2col<true>(src2, B, OC, OH, OW, IC, IH2, IW2, FH, FW);
                else
                    img2col<false>(src2, B, OC, OH, OW, IC, IH2, IW2, FH, FW);
            } else {
                if (is_xcorr)
                    img2col_stride<true>(src2, B, OC, OH, OW, IC, IH2, IW2, FH,
                                         FW, SH, SW);
                else
                    img2col_stride<false>(src2, B, OC, OH, OW, IC, IH2, IW2, FH,
                                          FW, SH, SW);
            }
        }
        {
            Workspace workspace(static_cast<dt_byte*>(bundle.get(2)),
                                bundle.get_size(2));
            size_t M = OC;
            size_t K = IC * FH * FW;
            size_t N = OH * OW;

#define DISPATCH_GEMM_STRATEGY(_gemm, _gemm_midout_enum, _bias,              \
                               _bias_midout_enum, _nonline,                  \
                               _nonline_midout_enum)                         \
    MIDOUT_BEGIN(megdnn_aarch64_conv_bias_quint8_gemm, 1, _gemm_midout_enum, \
                 _bias_midout_enum, _nonline_midout_enum) {                  \
        matmul::gemm_##_gemm##_##_bias##_##_nonline strategy(                \
                M, N, K, param.filter_type, param.src_type, param.dst_type); \
        megdnn::matmul::GemmInterleaved<                                     \
                matmul::gemm_##_gemm##_##_bias##_##_nonline>                 \
                gemm_interleaved(M, N, K, false, false, strategy);           \
        gemm_interleaved.execute(filter, K, B, N, dst, N, workspace.raw_ptr, \
                                 bias);                                      \
    }                                                                        \
    MIDOUT_END()

            DISPATCH_GEMM_BIAS(u8_8x8, 0)
#undef DISPATCH_GEMM_STRATEGY
        }
    }
}
// vim: syntax=cpp.doxygen
