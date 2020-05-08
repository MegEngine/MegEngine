/**
 * \file dnn/src/aarch64/conv_bias/int8/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/conv_bias/int8/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#include "src/aarch64/matrix_mul/int8/kernel_4x4x16.h"
#include "src/aarch64/matrix_mul/int8_dot/kernel_8x12x4.h"
#include "src/arm_common/conv_bias/matmul_postprocess.h"

using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

namespace impl {
template <BiasMode bmode, typename Op, int block_m, int block_n>
struct KernCaller;

#if __ARM_FEATURE_DOTPROD
template <BiasMode bmode, typename Op>
struct KernCaller<bmode, Op, 8, 12> {
    static void run(const dt_int8* packA, const dt_int8* packB, size_t M,
                    size_t N, size_t K, dt_int8* C, size_t LDC, bool is_first_k,
                    Op op, const dt_int32* bias, dt_int32* workspace) {
        megdnn_assert(is_first_k);

        constexpr size_t A_INTERLEAVE = 8;
        constexpr size_t B_INTERLEAVE = 12;
        //! K is packed to times of 4
        K = round_up<size_t>(K, 4);
        const int K8 = (K << 3);
        const int K12 = K * 12;
        const int K4 = K * 4;

        size_t m = 0;
        for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
            int8_t* output = C + (m * LDC);

            size_t n = 0;
            const dt_int8* cur_packB = packB;
            for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
                matmul_8x12x4::kern_8x12(packA, cur_packB, K, workspace, 12,
                                         is_first_k);

                arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 8, 12, 8,
                                           12>::postprocess(bias, workspace,
                                                            output, LDC, op);
                output += B_INTERLEAVE;
                cur_packB += K12;
            }

            for (; n < N; n += 4) {
                matmul_8x12x4::kern_8x4(packA, cur_packB, K, workspace, 4,
                                        is_first_k, std::min<size_t>(N - n, 4));

#define cb(m, n)                                                             \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 8, 4, 8, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_N(cb, 8, std::min<size_t>(N - n, 4));
#undef cb
                output += 4;
                cur_packB += K4;
            }
            packA += K8;

            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias += A_INTERLEAVE;
            }
        }

        for (; m < M; m += 4) {
            int8_t* output = C + (m * LDC);
            const dt_int8* cur_packB = packB;
            size_t n = 0;
            for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
                matmul_8x12x4::kern_4x12(packA, cur_packB, K, workspace, 12,
                                         is_first_k,
                                         std::min<size_t>(M - m, 4));
#define cb(m, n)                                                              \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 12, m, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_M_N(cb, std::min<size_t>(M - m, 4), 12);
#undef cb

                output += B_INTERLEAVE;
                cur_packB += K12;
            }

            for (; n < N; n += 4) {
                matmul_8x12x4::kern_4x4(packA, cur_packB, K, workspace, 4,
                                        is_first_k, std::min<size_t>(M - m, 4),
                                        std::min<size_t>(N - n, 4));
#define cb(m, n)                                                             \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 4, m, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_M(cb, std::min<size_t>(M - m, 4),
                           std::min<size_t>(N - n, 4));
#undef cb

                output += 4;
                cur_packB += K4;
            }
            packA += K4;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias += 4;
            }
        }
    }
};

#else

template <BiasMode bmode, typename Op>
struct KernCaller<bmode, Op, 4, 4> {
    static void run(const dt_int8* packA, const dt_int8* packB, size_t M,
                    size_t N, size_t K, dt_int8* C, size_t LDC, bool is_first_k,
                    Op op, const dt_int32* bias, dt_int32* workspace) {
        megdnn_assert(is_first_k);

        constexpr size_t A_INTERLEAVE = 4;
        constexpr size_t B_INTERLEAVE = 4;
        //! K is packed to times of 4
        K = round_up<size_t>(K, 16);
        const int K4 = K * 4;

        size_t m = 0;
        for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
            int8_t* output = C + (m * LDC);

            size_t n = 0;
            const dt_int8* cur_packB = packB;
            for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
                matmul_4x4x16::kern_4x4(packA, cur_packB, K, workspace, 4,
                                        is_first_k);
                arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 4, 4,
                                           4>::postprocess(bias, workspace,
                                                           output, LDC, op);

                output += B_INTERLEAVE;
                cur_packB += K4;
            }

            for (; n < N; n += B_INTERLEAVE) {
                matmul_4x4x16::kern_4x4_remain(packA, cur_packB, K, workspace,
                                               4, is_first_k, 4,
                                               std::min<size_t>(N - n, 4));
#define cb(m, n)                                                             \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 4, 4, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_N(cb, 4, std::min<size_t>(N - n, 4));
#undef cb
                output += B_INTERLEAVE;
                cur_packB += K4;
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
                matmul_4x4x16::kern_4x4_remain(
                        packA, cur_packB, K, workspace, 4, is_first_k,
                        std::min<size_t>(M - m, 4), std::min<size_t>(N - n, 4));

#define cb(m, n)                                                             \
    arm_common::ConvBiasMatmul<bmode, Op, dt_int8, 4, 4, m, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_M(cb, std::min<size_t>(M - m, 4),
                           std::min<size_t>(N - n, 4));
#undef cb
                output += B_INTERLEAVE;
                cur_packB += K4;
            }
            packA += K4;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias += A_INTERLEAVE;
            }
        }
    }
};

#endif

}  // namespace impl
#if !(__ARM_FEATURE_DOTPROD)
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_4x4_nobias_identity)

void gemm_s8_4x4_nobias_identity::pack_A(dt_int8* outptr, const dt_int8* inptr,
                                         int ldin, int y0, int ymax, int k0,
                                         int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x4x16::gemm_s8_4x4_pack_B_n(outptr, inptr, ldin, y0, ymax, k0,
                                            kmax);
    } else {
        matmul_4x4x16::gemm_s8_4x4_pack_A_n(outptr, inptr, ldin, y0, ymax, k0,
                                            kmax);
    }
}

void gemm_s8_4x4_nobias_identity::pack_B(dt_int8* out, const dt_int8* in,
                                         int ldin, int x0, int xmax, int k0,
                                         int kmax, bool transpose) const {
    if (transpose) {
        matmul_4x4x16::gemm_s8_4x4_pack_A_n(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        matmul_4x4x16::gemm_s8_4x4_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

size_t gemm_s8_4x4_nobias_identity::get_workspace_size() const {
    return 4 * 4 * sizeof(dt_int32);
}
#else
MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_s8_8x12_nobias_identity)

void gemm_s8_8x12_nobias_identity::pack_A(dt_int8* outptr, const dt_int8* inptr,
                                          int ldin, int y0, int ymax, int k0,
                                          int kmax, bool transpose) const {
    MEGDNN_MARK_USED_VAR(matmul_8x12x4::gemm_s8_8x12_pack_A_t);
    MEGDNN_MARK_USED_VAR(matmul_8x12x4::gemm_s8_8x12_pack_B_t);
    if (transpose) {
        matmul_8x12x4::gemm_s8_8x12_pack_B_n(outptr, inptr, ldin, y0, ymax, k0,
                                             kmax);
    } else {
        matmul_8x12x4::gemm_s8_8x12_pack_A_n(outptr, inptr, ldin, y0, ymax, k0,
                                             kmax);
    }
}

void gemm_s8_8x12_nobias_identity::pack_B(dt_int8* out, const dt_int8* in,
                                          int ldin, int x0, int xmax, int k0,
                                          int kmax, bool transpose) const {
    if (transpose) {
        matmul_8x12x4::gemm_s8_8x12_pack_A_n(out, in, ldin, x0, xmax, k0, kmax);
    } else {
        matmul_8x12x4::gemm_s8_8x12_pack_B_n(out, in, ldin, x0, xmax, k0, kmax);
    }
}

size_t gemm_s8_8x12_nobias_identity::get_workspace_size() const {
    return 8 * 12 * sizeof(dt_int32);
}

#endif

#define KERN(_block_m, _block_n, _bias, _BIAS, _nonline, _OP)               \
    void gemm_s8_##_block_m##x##_block_n##_##_bias##_##_nonline::kern(      \
            const dt_int8* packA, const dt_int8* packB, size_t M, size_t N, \
            size_t K, dt_int8* C, size_t LDC, bool is_first_k,              \
            const dt_int32* bias, dt_int32* workspace) const {              \
        float scale_A = A_dtype.param<dtype::QuantizedS8>().scale;          \
        float scale_B = B_dtype.param<dtype::QuantizedS8>().scale;          \
        float scale_C = C_dtype.param<dtype::QuantizedS8>().scale;          \
        DEFINE_OP(_OP);                                                     \
        impl::KernCaller<_BIAS, decltype(op), _block_m, _block_n>::run(     \
                packA, packB, M, N, K, C, LDC, is_first_k, op, bias,        \
                workspace);                                                 \
    }

#define DEFINE_OP(_Op) \
    arm_common::_Op<dt_qint32, dt_qint8> op(scale_A* scale_B, scale_C);

#if !(__ARM_FEATURE_DOTPROD)
KERN(4, 4, nobias, BiasMode::NO_BIAS, identity, TypeCvtOp)
KERN(4, 4, nobias, BiasMode::NO_BIAS, relu, ReluOp)
KERN(4, 4, nobias, BiasMode::NO_BIAS, hswish, HSwishOp)
#else
KERN(8, 12, nobias, BiasMode::NO_BIAS, identity, TypeCvtOp)
KERN(8, 12, nobias, BiasMode::NO_BIAS, relu, ReluOp)
KERN(8, 12, nobias, BiasMode::NO_BIAS, hswish, HSwishOp)
#endif
#undef DEFINE_OP

#define DEFINE_OP(_Op)                                        \
    arm_common::_Op<dt_qint32, dt_qint8> op(scale_A* scale_B, \
                                            scale_A* scale_B, scale_C);
#if !(__ARM_FEATURE_DOTPROD)
KERN(4, 4, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, identity, AddOp)
KERN(4, 4, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, relu, FuseAddReluOp)
KERN(4, 4, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, hswish,
     FuseAddHSwishOp)
#else
KERN(8, 12, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, identity, AddOp)
KERN(8, 12, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, relu, FuseAddReluOp)
KERN(8, 12, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, hswish,
     FuseAddHSwishOp)
#endif
#undef DEFINE_OP

#undef KERN

// vim: syntax=cpp.doxygen
