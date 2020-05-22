/**
 * \file dnn/src/armv7/conv_bias/int8/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/conv_bias/int8/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#include "src/arm_common/conv_bias/matmul_postprocess.h"
#include "src/armv7/matrix_mul/int8/kernel_4x2x16.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace impl {
template <BiasMode bmode, typename Op, int block_m, int block_n>
struct KernCaller;

template <BiasMode bmode, typename Op>
struct KernCaller<bmode, Op, 4, 2> {
    static void run(const dt_int8* packA, const dt_int8* packB, size_t M,
                    size_t N, size_t K, dt_int8* C, size_t LDC, bool is_first_k,
                    Op op, const dt_int32* bias, dt_int32* workspace) {
        megdnn_assert(is_first_k);

        constexpr size_t A_INTERLEAVE = 4;
        constexpr size_t B_INTERLEAVE = 2;
        //! K is packed to times of 4
        K = round_up<size_t>(K, 16);
        const int K4 = K * 4;
        const int K2 = K * 2;

        size_t m = 0;
        for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
            int8_t* output = C + (m * LDC);

            size_t n = 0;
            const dt_int8* cur_packB = packB;
            for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
                matmul_4x2x16::kern_4x2(packA, cur_packB, K, workspace, 2,
                                        is_first_k, 4, 2);
                arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 2, 4,
                                           2>::postprocess(bias, workspace,
                                                           output, LDC, op);
                output += B_INTERLEAVE;
                cur_packB += K2;
            }

            for (; n < N; n += B_INTERLEAVE) {
                matmul_4x2x16::kern_4x2(packA, cur_packB, K, workspace, 2,
                                        is_first_k, 4,
                                        std::min<size_t>(N - n, 2));
#define cb(m, n)                                                             \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 2, 4, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_N(cb, 4, std::min<size_t>(N - n, 2));
#undef cb
                output += B_INTERLEAVE;
                cur_packB += K2;
            }

            packA += K4;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias += A_INTERLEAVE;
            }
        }

        for (; m < M; m += A_INTERLEAVE) {
            int8_t* output = C + (m * LDC);

            size_t n = 0;
            const dt_int8* cur_packB = packB;
            for (; n < N; n += B_INTERLEAVE) {
                matmul_4x2x16::kern_4x2(packA, cur_packB, K, workspace, 2,
                                        is_first_k, std::min<size_t>(M - m, 4),
                                        std::min<size_t>(N - n, 2));
#define cb(m, n)                                                             \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 2, m, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_M(cb, std::min<size_t>(M - m, 4),
                           std::min<size_t>(N - n, 2));
#undef cb

                output += B_INTERLEAVE;
                cur_packB += K2;
            }
            packA += K4;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias += A_INTERLEAVE;
            }
        }
    }
};

}  // namespace impl

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_4x2_nobias_identity)

void gemm_s8_4x2_nobias_identity::pack_A(dt_int8* outptr, const dt_int8* inptr,
                                         int ldin, int y0, int ymax, int k0,
                                         int kmax, bool /*transpose*/) const {
    MEGDNN_MARK_USED_VAR(matmul_4x2x16::gemm_s8_4x2_pack_A_t);
    matmul_4x2x16::gemm_s8_4x2_pack_A_n(outptr, inptr, ldin, y0, ymax, k0,
                                        kmax);
}

void gemm_s8_4x2_nobias_identity::pack_B(dt_int8* out, const dt_int8* in,
                                         int ldin, int x0, int xmax, int k0,
                                         int kmax, bool /*transpose*/) const {
    MEGDNN_MARK_USED_VAR(matmul_4x2x16::gemm_s8_4x2_pack_B_t);
    matmul_4x2x16::gemm_s8_4x2_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
}

size_t gemm_s8_4x2_nobias_identity::get_workspace_size() const {
    return 4 * 2 * sizeof(dt_int32);
}

#define KERN(_bias, _BIAS, _nonline, _OP)                                   \
    void gemm_s8_4x2_##_bias##_##_nonline::kern(                            \
            const dt_int8* packA, const dt_int8* packB, size_t M, size_t N, \
            size_t K, dt_int8* C, size_t LDC, bool is_first_k,              \
            const dt_int32* bias, dt_int32* workspace) const {              \
        float scale_A = A_dtype.param<dtype::QuantizedS8>().scale;          \
        float scale_B = B_dtype.param<dtype::QuantizedS8>().scale;          \
        float scale_C = C_dtype.param<dtype::QuantizedS8>().scale;          \
        DEFINE_OP(_OP);                                                     \
        impl::KernCaller<_BIAS, decltype(op), 4, 2>::run(                   \
                packA, packB, M, N, K, C, LDC, is_first_k, op, bias,        \
                workspace);                                                 \
    }

#define DEFINE_OP(_Op) \
    arm_common::_Op<dt_qint32, dt_qint8> op(scale_A* scale_B, scale_C);

KERN(nobias, BiasMode::NO_BIAS, identity, TypeCvtOp)
KERN(nobias, BiasMode::NO_BIAS, relu, ReluOp)
KERN(nobias, BiasMode::NO_BIAS, hswish, HSwishOp)
#undef DEFINE_OP

#define DEFINE_OP(_Op)                                        \
    arm_common::_Op<dt_qint32, dt_qint8, true> op(scale_A* scale_B, \
                                            scale_A* scale_B, scale_C);
KERN(bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, identity, AddOp)
KERN(bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, relu, FuseAddReluOp)
KERN(bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, hswish, FuseAddHSwishOp)
#undef DEFINE_OP

#undef KERN

// vim: syntax=cpp.doxygen
