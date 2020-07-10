/**
 * \file dnn/src/armv7/conv_bias/int8/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/conv_bias/int8/algos.h"
#include "src/arm_common/convolution/img2col_helper.h"
#include "src/armv7/conv_bias/int8/strategy.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/matrix_mul/gemm_impl.h"

#include "midout.h"

MIDOUT_DECL(megdnn_armv7_conv_bias_int8)

using namespace megdnn;
using namespace armv7;

/* ===================== matrix mul algo ===================== */

bool ConvBiasImpl::AlgoS8MatrixMul::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    auto&& fm = param.filter_meta;
    return param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
           param.dst_type.enumv() == DTypeEnum::QuantizedS8 &&
           fm.format == param::ConvBias::Format::NCHW && fm.spatial_ndim == 2 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
           //! As postprocess, the bias is not contigous read, make the
           //! performance bad, so we do not process it in fused kernel
           param.bias_mode != BiasMode::BIAS &&
           //! This algo is only support single thread
           param.nr_threads == 1_z;
}

WorkspaceBundle ConvBiasImpl::AlgoS8MatrixMul::get_bundle(
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
        part0 = (IC * IH2 * IW2 + 16) * sizeof(int8_t);
        part1 = (IC * FH * FW * OH * OW + 16) * sizeof(int8_t);
    }
    {
        size_t M = OC;
        size_t K = IC * FH * FW;
        size_t N = OH * OW;

#define DISPATCH_GEMM_STRATEGY(_gemm, _gemm_midout_enum, _bias,              \
                               _bias_midout_enum, _nonline,                  \
                               _nonline_midout_enum)                         \
    MIDOUT_BEGIN(megdnn_armv7_conv_bias_int8, 0, _gemm_midout_enum,          \
                 _bias_midout_enum, _nonline_midout_enum) {                  \
        matmul::gemm_##_gemm##_##_bias##_##_nonline strategy(                \
                M, N, K, param.filter_type, param.src_type, param.dst_type); \
        part2 = megdnn::matmul::GemmInterleaved<                             \
                        matmul::gemm_##_gemm##_##_bias##_##_nonline>(        \
                        M, N, K, false, false, strategy)                     \
                        .get_workspace_size();                               \
    }                                                                        \
    MIDOUT_END()

        DISPATCH_GEMM_BIAS(s8_4x2, 0)

#undef DISPATCH_GEMM_STRATEGY
    }
    return {nullptr, {part0, part1, part2}};
}

void ConvBiasImpl::AlgoS8MatrixMul::kimpl(const NCBKernParam& param,
                                          const NCBKernIndex& ncb_index) {
    auto is_xcorr = !param.filter_meta.should_flip;
    UNPACK_CONV_NCB_KERN_SIZES(param);
    auto bundle = get_bundle(param);
    bundle.set(param.workspace_ptr);
    auto IH2 = IH + 2 * PH;
    auto IW2 = IW + 2 * PW;
    size_t group_id = ncb_index.ndrange_id[0];
    // workspace = tmp..src2
    for (size_t n = 0; n < N; ++n) {
        dt_int8* src = const_cast<dt_int8*>(param.src<dt_int8>(n, group_id));
        dt_int8* filter = const_cast<dt_int8*>(param.filter<dt_int8>(group_id));
        dt_int8* dst = static_cast<dt_int8*>(param.dst<dt_int8>(n, group_id));
        dt_int32* bias = const_cast<dt_int32*>(param.bias<dt_int32>(n, group_id));

        dt_int8 *B, *src2;
        if (FH == 1 && FW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0) {
            // special case: 1x1
            B = const_cast<dt_int8*>(src);
        } else {
            src2 = static_cast<dt_int8*>(bundle.get(0));
            // copy src to src2;
            dt_int8* src2_ptr = src2;
            const dt_int8* src_ptr = src;
            rep(ic, IC) {
                if (PH != 0) {
                    std::memset(src2_ptr, 0, sizeof(dt_int8) * PH * IW2);
                    src2_ptr += PH * IW2;
                }
                rep(ih, IH) {
                    if (PW != 0)
                        rep(pw, PW) { *(src2_ptr++) = 0.0f; }
                    std::memcpy(src2_ptr, src_ptr, sizeof(dt_int8) * IW);
                    src2_ptr += IW;
                    src_ptr += IW;
                    if (PW != 0)
                        rep(pw, PW) { *(src2_ptr++) = 0.0f; }
                }
                if (PH != 0) {
                    std::memset(src2_ptr, 0, sizeof(dt_int8) * PH * IW2);
                    src2_ptr += PH * IW2;
                }
            }

            B = static_cast<dt_int8*>(bundle.get(1));
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
    MIDOUT_BEGIN(megdnn_armv7_conv_bias_int8, 1, _gemm_midout_enum,          \
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

            DISPATCH_GEMM_BIAS(s8_4x2, 0)
#undef DISPATCH_GEMM_STRATEGY
        }
    }
}

// vim: syntax=cpp.doxygen
