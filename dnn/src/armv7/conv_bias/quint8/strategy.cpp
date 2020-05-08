/**
 * \file dnn/src/armv7/conv_bias/quint8/strategy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/armv7/conv_bias/quint8/strategy.h"
#include "src/armv7/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

#include "src/armv7/matrix_mul/quint8/kernel_4x8x8.h"
#include "src/arm_common/conv_bias/matmul_postprocess.h"

using namespace megdnn;
using namespace armv7;
using namespace armv7::matmul;

namespace impl {
template <BiasMode bmode, typename Op, int block_m, int block_n>
struct KernCaller;

template <BiasMode bmode, typename Op>
struct KernCaller<bmode, Op, 4, 8> {
    static void run(const dt_uint8* packA, const dt_uint8* packB, size_t M,
                    size_t N, size_t K, dt_uint8* C, size_t LDC,
                    bool is_first_k, Op op, const dt_int32* bias,
                    dt_int32* workspace, uint8_t zp_A, uint8_t zp_B) {
        megdnn_assert(is_first_k);

        constexpr size_t A_INTERLEAVE = 4;
        constexpr size_t B_INTERLEAVE = 8;
        //! K is packed to times of 8
        K = round_up<size_t>(K, 8);
        const int K8 = K * 8;
        const int K4 = K * 4;

        size_t m = 0;
        for (; m < M; m += A_INTERLEAVE) {
            uint8_t* output = C + (m * LDC);
            const dt_uint8* cur_packB = packB;
            size_t n = 0;
            for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
                matmul_4x8x8::kern_4x8(packA, cur_packB, K, workspace, 8,
                                       is_first_k, std::min<size_t>(M - m, 4),
                                       zp_A, zp_B);
#define cb(m, n)                                                              \
    arm_common::ConvBiasMatmul<bmode, Op, dt_uint8, 4, 8, m, n>::postprocess( \
            bias, workspace, output, LDC, op);
                DISPATCH_M_N(cb, std::min<size_t>(M - m, 4), 8);
#undef cb

                output += B_INTERLEAVE;
                cur_packB += K8;
            }

            for (; n < N; n += 4) {
                matmul_4x8x8::kern_4x4(packA, cur_packB, K, workspace, 4,
                                       is_first_k, std::min<size_t>(M - m, 4),
                                       std::min<size_t>(N - n, 4), zp_A, zp_B);
#define cb(m, n)                                                              \
    arm_common::ConvBiasMatmul<bmode, Op, dt_uint8, 4, 4, m, n>::postprocess( \
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

}  // namespace impl

MEGDNN_REG_GEMM_STRATEGY_IMPL(gemm_u8_4x8_nobias_identity);
void gemm_u8_4x8_nobias_identity::pack_A(dt_uint8* outptr,
                                         const dt_uint8* inptr, int ldin,
                                         int y0, int ymax, int k0, int kmax,
                                         bool transpose) const {
    uint8_t zA = A_dtype.param<dtype::Quantized8Asymm>().zero_point;
    if (transpose) {
        matmul_4x8x8::gemm_u8_4x8_transpose_pack_A_n(outptr, inptr, ldin, y0,
                                                     ymax, k0, kmax, zA);
    } else {
        matmul_4x8x8::gemm_u8_4x8_pack_A_n(outptr, inptr, ldin, y0, ymax, k0,
                                           kmax, zA);
    }
}

void gemm_u8_4x8_nobias_identity::pack_B(dt_uint8* out, const dt_uint8* in,
                                         int ldin, int x0, int xmax, int k0,
                                         int kmax, bool transpose) const {
    uint8_t zB = B_dtype.param<dtype::Quantized8Asymm>().zero_point;
    if (transpose) {
        matmul_4x8x8::gemm_u8_4x8_transpose_pack_B_n(out, in, ldin, x0, xmax,
                                                     k0, kmax, zB);
    } else {
        matmul_4x8x8::gemm_u8_4x8_pack_B_n(out, in, ldin, x0, xmax, k0, kmax,
                                           zB);
    }
}

size_t gemm_u8_4x8_nobias_identity::get_workspace_size() const {
    return 4 * 8 * sizeof(dt_int32);
}

#define KERN(_block_m, _block_n, _bias, _BIAS, _nonline, _OP)                 \
    void gemm_u8_##_block_m##x##_block_n##_##_bias##_##_nonline::kern(        \
            const dt_uint8* packA, const dt_uint8* packB, size_t M, size_t N, \
            size_t K, dt_uint8* C, size_t LDC, bool is_first_k,               \
            const dt_int32* bias, dt_int32* workspace) const {                \
        float scale_A = A_dtype.param<dtype::Quantized8Asymm>().scale;        \
        uint8_t zp_A = A_dtype.param<dtype::Quantized8Asymm>().zero_point;    \
        float scale_B = B_dtype.param<dtype::Quantized8Asymm>().scale;        \
        uint8_t zp_B = B_dtype.param<dtype::Quantized8Asymm>().zero_point;    \
        float scale_C = C_dtype.param<dtype::Quantized8Asymm>().scale;        \
        uint8_t zp_C = C_dtype.param<dtype::Quantized8Asymm>().zero_point;    \
        DEFINE_OP(_OP);                                                       \
        impl::KernCaller<_BIAS, decltype(op), _block_m, _block_n>::run(       \
                packA, packB, M, N, K, C, LDC, is_first_k, op, bias,          \
                workspace, zp_A, zp_B);                                       \
    }

#define DEFINE_OP(_Op) \
    arm_common::_Op<dt_qint32, dt_quint8> op(scale_A* scale_B, scale_C, zp_C);

KERN(4, 8, nobias, BiasMode::NO_BIAS, identity, TypeCvtOp)
KERN(4, 8, nobias, BiasMode::NO_BIAS, relu, ReluOp)
KERN(4, 8, nobias, BiasMode::NO_BIAS, hswish, HSwishOp)
#undef DEFINE_OP

#define DEFINE_OP(_Op)                                         \
    arm_common::_Op<dt_qint32, dt_quint8> op(scale_A* scale_B, \
                                             scale_A* scale_B, scale_C, zp_C);
KERN(4, 8, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, identity, AddOp)
KERN(4, 8, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, relu, FuseAddReluOp)
KERN(4, 8, bias_channel, BiasMode::BROADCAST_CHANNEL_BIAS, hswish,
     FuseAddHSwishOp)
#undef DEFINE_OP

#undef KERN

// vim: syntax=cpp.doxygen
