/**
 * \file dnn/src/aarch64/matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/aarch64/matrix_mul/algos.h"
#include "src/aarch64/matrix_mul/fp16/strategy.h"
#include "src/aarch64/matrix_mul/fp32/strategy.h"
#include "src/aarch64/matrix_mul/int16/strategy.h"
#include "src/aarch64/matrix_mul/int8/strategy.h"
#include "src/aarch64/matrix_mul/int8_dot/strategy.h"
#include "src/aarch64/matrix_mul/int8x8x16/strategy.h"
#include "src/aarch64/matrix_mul/int4x4x16/strategy.h"
#include "src/aarch64/matrix_mul/quint8/strategy.h"
#include "src/aarch64/matrix_mul/quint8_dot/gemv.h"
#include "src/aarch64/matrix_mul/quint8_dot/strategy.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_impl.h"

#if MGB_ENABLE_CPUINFO
#include "cpuinfo.h"
#endif
#include "midout.h"

MIDOUT_DECL(megdnn_aarch64_matmul_kern)

using namespace megdnn;
using namespace aarch64;

/* ===================== F32K8X12X1 algo ===================== */
bool MatrixMulImpl::AlgoF32K8x12x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT;
}

size_t MatrixMulImpl::AlgoF32K8x12x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoF32K8x12x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::sgemm_8x12 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_8x12>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32K8x12x1::get_kern(
        const KernSizeParam&) const {
    auto f32_kern_8x12 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                     midout_iv("AlgoF32K8x12x1::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<float>(),
                       Bptr = kern_param.B<float>();
            auto Cptr = kern_param.C<float>();
            aarch64::matmul::sgemm_8x12 strategy(M, N, K, A_type, B_type,
                                                 C_type);
            megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_8x12>(
                    M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };

    return f32_kern_8x12;
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF32K8x12x1, megdnn_aarch64_matmul_kern,
                                     "AlgoF32K8x12x1Impl"_hash,
                                     aarch64::matmul::sgemm_8x12, float, float,
                                     AlgoDataType::FLOAT32, DEFAULT);

/* ===================== F32_MK4_8X12X1 algo ===================== */
bool MatrixMulImpl::AlgoF32MK4_8x12x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.M % 4 == 0 && kern_size_param.K % 4 == 0;
}

size_t MatrixMulImpl::AlgoF32MK4_8x12x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoF32MK4_8x12x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::sgemm_mk4_8x12 strategy(M, N, K, A_type, B_type,
                                                 C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_mk4_8x12>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32MK4_8x12x1::get_kern(
        const KernSizeParam&) const {
    auto f32_kern_mk4_8x12 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                     midout_iv("AlgoF32MK4_8x12x1::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<float>(),
                       Bptr = kern_param.B<float>();
            auto Cptr = kern_param.C<float>();
            aarch64::matmul::sgemm_mk4_8x12 strategy(M, N, K, A_type, B_type,
                                                     C_type);
            megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_mk4_8x12>(
                    M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return f32_kern_mk4_8x12;
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF32MK4_8x12x1,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoF32MK4_8x12x1Impl"_hash,
                                     aarch64::matmul::sgemm_mk4_8x12, float,
                                     float, AlgoDataType::FLOAT32, MK4);

/* ===================== F32K4X16X1 algo ===================== */

bool MatrixMulImpl::AlgoF32K4x16x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT;
}

size_t MatrixMulImpl::AlgoF32K4x16x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoF32K4x16x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::sgemm_4x16 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_4x16>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32K4x16x1::get_kern(
        const KernSizeParam&) const {
    auto f32_kern_4x16 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                     midout_iv("AlgoF32K4x16x1::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<float>(),
                       Bptr = kern_param.B<float>();
            auto Cptr = kern_param.C<float>();

            aarch64::matmul::sgemm_4x16 strategy(M, N, K, A_type, B_type,
                                                 C_type);
            megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_4x16>(
                    M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return f32_kern_4x16;
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF32K4x16x1, megdnn_aarch64_matmul_kern,
                                     "AlgoF32K4x16x1Impl"_hash,
                                     aarch64::matmul::sgemm_4x16, float, float,
                                     AlgoDataType::FLOAT32, MK4);

/* ===================== F32MK4_4x16 algo ===================== */

bool MatrixMulImpl::AlgoF32MK4_4x16::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.C_type == dtype::Float32() &&
           kern_size_param.B_type == dtype::Float32() &&
           kern_size_param.A_type == dtype::Float32() &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoF32MK4_4x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoF32MK4_4x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::sgemm_nopack_4x16 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       aarch64::matmul::sgemm_nopack_4x16, false>(M, N, K, trA,
                                                                  trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32MK4_4x16::get_kern(
        const KernSizeParam&) const {
    auto f32_kern_mk4_4x16 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                     midout_iv("AlgoF32MK4_4x16::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<float>(),
                       Bptr = kern_param.B<float>();
            auto Cptr = kern_param.C<float>();

            aarch64::matmul::sgemm_nopack_4x16 strategy(A_type, B_type, C_type);
            megdnn::matmul::GemmInterleaved<aarch64::matmul::sgemm_nopack_4x16,
                                            false>(M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return f32_kern_mk4_4x16;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/* ===================== F16 K8x24x1 algo ===================== */
namespace {
void f16_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern, midout_iv("f16_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_float16>(),
                   Bptr = kern_param.B<dt_float16>();
        auto Cptr = kern_param.C<dt_float16>();

        aarch64::matmul::hgemm_8x24 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::hgemm_8x24>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoF16K8x24x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float16();
}

size_t MatrixMulImpl::AlgoF16K8x24x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoF16K8x24x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::hgemm_8x24 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::hgemm_8x24>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF16K8x24x1::get_kern(
        const KernSizeParam&) const {
    return f16_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF16K8x24x1, megdnn_aarch64_matmul_kern,
                                     "AlogF16K8x24x1Impl"_hash,
                                     aarch64::matmul::hgemm_8x24, dt_float16,
                                     dt_float16, AlgoDataType::FLOAT16,
                                     DEFAULT);
/* ===================== F16_MK8_8x8 algo ===================== */

bool MatrixMulImpl::AlgoF16MK8_8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float16() &&
           kern_size_param.format == param::MatrixMul::Format::MK8 &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoF16MK8_8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoF16MK8_8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_nopack_f16_8x8 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       aarch64::matmul::gemm_nopack_f16_8x8, false>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF16MK8_8x8::get_kern(
        const KernSizeParam&) const {
    auto kern_mk8_8x8 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                     midout_iv("AlgoF16MK8_8x8::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<dt_float16>(),
                       Bptr = kern_param.B<dt_float16>();
            auto Cptr = kern_param.C<dt_float16>();

            aarch64::matmul::gemm_nopack_f16_8x8 strategy(A_type, B_type,
                                                          C_type);
            megdnn::matmul::GemmInterleaved<
                    aarch64::matmul::gemm_nopack_f16_8x8, false>(M, N, K, trA,
                                                                 trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return kern_mk8_8x8;
}

#endif

#if __ARM_FEATURE_DOTPROD
/* ==================== Int8x8x32 K8x12x4 Dotprod algo ==================== */
namespace {
void int8x8x32_k8x12x4_dotprod_kern(
        const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x32_k8x12x4_dotprod_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_s8_8x12 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8_8x12>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32K8x12x4DotProd::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x32(kern_size_param);
}

size_t MatrixMulImpl::AlgoInt8x8x32K8x12x4DotProd::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x32K8x12x4DotProd::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;

        aarch64::matmul::gemm_s8_8x12 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8_8x12>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32K8x12x4DotProd::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_k8x12x4_dotprod_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32K8x12x4DotProd,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x32K8x12x4DotProdImpl"_hash,
                                     aarch64::matmul::gemm_s8_8x12, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     DEFAULT);

/* =================== Int8x8x32 MK4 8X12X4 Dotprod algo =================== */
namespace {
void int8x8x32_mk4_8x12x4_dotprod_kern(
        const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x32_mk4_8x12x4_dotprod_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_mk4_s8_8x12 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_mk4_s8_8x12>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32MK4_8x12x4DotProd::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           (kern_size_param.A_type.enumv() == DTypeEnum::Int8 ||
            kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
           (kern_size_param.C_type.enumv() == DTypeEnum::Int32 ||
            kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32) &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4_DOT &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoInt8x8x32MK4_8x12x4DotProd::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_aarch64_matmul_kern,
            midout_iv("AlgoInt8x8x32MK4_8x12x4DotProd::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;

        aarch64::matmul::gemm_mk4_s8_8x12 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        return megdnn::matmul::GemmInterleaved<
                       aarch64::matmul::gemm_mk4_s8_8x12>(M, N, K, trA, trB,
                                                          strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32MK4_8x12x4DotProd::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_mk4_8x12x4_dotprod_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32MK4_8x12x4DotProd,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x32MK4_8x12x4DotProdImpl"_hash,
                                     aarch64::matmul::gemm_mk4_s8_8x12, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     MK4_DOT);
#else

/* ===================== Int8x8x32 MK4 4x4x16 algo ===================== */
namespace {
void int8x8x32_mk4_4x4x16_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x32_mk4_4x4x16_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        aarch64::matmul::gemm_mk4_s8_4x4 strategy(M, N, K, A_type, B_type,
                                                  C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_mk4_s8_4x4>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32MK4_4x4x16::usable(
        const KernSizeParam& param) const {
    return param.A_type.enumv() == param.B_type.enumv() &&
           (param.A_type.enumv() == DTypeEnum::Int8 ||
            param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
           (param.C_type.enumv() == DTypeEnum::Int32 ||
            param.C_type.enumv() == DTypeEnum::QuantizedS32) &&
           param.compute_mode == Param::ComputeMode::DEFAULT &&
           param.format == param::MatrixMul::Format::MK4 && !param.trA &&
           !param.trB && param.M % 4 == 0 && param.K % 4 == 0;
}

bool MatrixMulImpl::AlgoInt8x8x32MK4_4x4x16::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K > 16;
}

size_t MatrixMulImpl::AlgoInt8x8x32MK4_4x4x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x32MK4_4x4x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_mk4_s8_4x4 strategy(M, N, K, A_type, B_type,
                                                  C_type);
        return megdnn::matmul::GemmInterleaved<
                       aarch64::matmul::gemm_mk4_s8_4x4>(M, N, K, trA, trB,
                                                         strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32MK4_4x4x16::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_mk4_4x4x16_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32MK4_4x4x16,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x32MK4_4x4x16Impl"_hash,
                                     aarch64::matmul::gemm_mk4_s8_4x4, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     MK4);

/* ===================== Int8x8x32 K4x4x16 algo ===================== */
namespace {
void int8x8x32_k4x4x16_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x32_k4x4x16_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_s8_4x4 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8_4x4>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32K4x4x16::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x32(kern_size_param);
}

bool MatrixMulImpl::AlgoInt8x8x32K4x4x16::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K > 16;
}

size_t MatrixMulImpl::AlgoInt8x8x32K4x4x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x32K4x4x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8_4x4 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8_4x4>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32K4x4x16::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_k4x4x16_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32K4x4x16,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x32K4x4x16Impl"_hash,
                                     aarch64::matmul::gemm_s8_4x4, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     DEFAULT);
/* ===================== Int8x8x32 K8x8x8 algo ===================== */
namespace {
void int8x8x32_k8x8x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x32_k8x8x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_s8_8x8 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8_8x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32K8x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x32(kern_size_param);
}

bool MatrixMulImpl::AlgoInt8x8x32K8x8x8::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K <= 16;
}

size_t MatrixMulImpl::AlgoInt8x8x32K8x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x32K8x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8_8x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8_8x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32K8x8x8::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_k8x8x8_kern;
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32K8x8x8,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x32K8x8x8Impl"_hash,
                                     aarch64::matmul::gemm_s8_8x8, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     DEFAULT);
#endif

/* ===================== Int8x8x16 K8x8x8 algo ===================== */
namespace {
void int8x8x16_k8x8x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x16_k8x8x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();

        aarch64::matmul::gemm_s8x8x16_8x8 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8x8x16_8x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16K8x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x16(kern_size_param) &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

bool MatrixMulImpl::AlgoInt8x8x16K8x8x8::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K <= 16;
}

size_t MatrixMulImpl::AlgoInt8x8x16K8x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x16K8x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8x8x16_8x8 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8x8x16_8x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16K8x8x8::get_kern(
        const KernSizeParam&) const {
    return int8x8x16_k8x8x8_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x16K8x8x8,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x16K8x8x8Impl"_hash,
                                     aarch64::matmul::gemm_s8x8x16_8x8, int8_t,
                                     int16_t, AlgoDataType::INT8X8X16, DEFAULT);
/* ===================== Int8x8x16 K4x4x16 algo ===================== */
namespace {
void int8x8x16_k4x4x16_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x16_k4x4x16_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();

        aarch64::matmul::gemm_s8x8x16_4x4 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s8x8x16_4x4>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16K4x4x16::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x16(kern_size_param) &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

bool MatrixMulImpl::AlgoInt8x8x16K4x4x16::preferred(
        const KernSizeParam& kern_size_param) const {
    MEGDNN_MARK_USED_VAR(kern_size_param);
    return true;
}

size_t MatrixMulImpl::AlgoInt8x8x16K4x4x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x16K4x4x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8x8x16_4x4 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8x8x16_4x4>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16K4x4x16::get_kern(
        const KernSizeParam&) const {
    return int8x8x16_k4x4x16_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x16K4x4x16,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x16K4x4x16Impl"_hash,
                                     aarch64::matmul::gemm_s8x8x16_4x4, int8_t,
                                     int16_t, AlgoDataType::INT8X8X16, DEFAULT);

/* ===================== Int8x8x16 K16x12x4 algo ===================== */
namespace {
void int8x8x16_mk4_16x12x4_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x16_mk4_16x12x4_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();

        aarch64::matmul::gemm_s8x8x16_mk4_16x12_a53 strategy(M, N, K, A_type,
                                                             B_type, C_type);
        megdnn::matmul::GemmInterleaved<
                aarch64::matmul::gemm_s8x8x16_mk4_16x12_a53>(M, N, K, trA, trB,
                                                             strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16MK4_16x12x4::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x16(kern_size_param) &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.M % 4 == 0 && kern_size_param.K % 4 == 0;
}

bool MatrixMulImpl::AlgoInt8x8x16MK4_16x12x4::preferred(
        const KernSizeParam&) const {
#if !MGB_ENABLE_CPUINFO
    return false;
#else
    auto arch = cpuinfo_get_current_core()->uarch;
    bool little_core = arch == cpuinfo_uarch_cortex_a53 ||
                       arch == cpuinfo_uarch_cortex_a55;
    return little_core;
#endif
}

size_t MatrixMulImpl::AlgoInt8x8x16MK4_16x12x4::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x16MK4_16x12x4::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8x8x16_mk4_16x12_a53 strategy(M, N, K, A_type,
                                                             B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       matmul::gemm_s8x8x16_mk4_16x12_a53>(M, N, K, trA, trB,
                                                           strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16MK4_16x12x4::get_kern(
        const KernSizeParam&) const {
    return int8x8x16_mk4_16x12x4_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(
        AlgoInt8x8x16MK4_16x12x4, megdnn_aarch64_matmul_kern,
        "AlgoInt8x8x16MK4_16x12x4Impl"_hash,
        aarch64::matmul::gemm_s8x8x16_mk4_16x12_a53, int8_t, int16_t, int16_t,
        AlgoDataType::INT8X8X16, MK4);

/* ===================== Int8x8x16 MK4 4x4x8 algo ===================== */
namespace {
void int8x8x16_mk4_4x4x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x16_mk4_4x4x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();

        aarch64::matmul::gemm_s8x8x16_mk4_4x4_a72 strategy(M, N, K, A_type,
                                                           B_type, C_type);
        megdnn::matmul::GemmInterleaved<
                aarch64::matmul::gemm_s8x8x16_mk4_4x4_a72>(M, N, K, trA, trB,
                                                           strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16MK4_4x4x8::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x16(kern_size_param) &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.M % 4 == 0 && kern_size_param.K % 4 == 0;
}

bool MatrixMulImpl::AlgoInt8x8x16MK4_4x4x8::preferred(
        const KernSizeParam&) const {
#if !MGB_ENABLE_CPUINFO
    return false;
#else
    auto arch = cpuinfo_get_current_core()->uarch;
    bool little_core = arch == cpuinfo_uarch_cortex_a53 ||
                       arch == cpuinfo_uarch_cortex_a55;
    return !little_core;
#endif
}

size_t MatrixMulImpl::AlgoInt8x8x16MK4_4x4x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x16MK4_4x4x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8x8x16_mk4_4x4_a72 strategy(M, N, K, A_type,
                                                           B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       matmul::gemm_s8x8x16_mk4_4x4_a72>(M, N, K, trA, trB,
                                                         strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16MK4_4x4x8::get_kern(
        const KernSizeParam&) const {
    return int8x8x16_mk4_4x4x8_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x16MK4_4x4x8,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x16MK4_4x4x8_Impl"_hash,
                                     aarch64::matmul::gemm_s8x8x16_mk4_4x4_a72,
                                     int8_t, int16_t, AlgoDataType::INT8X8X16,
                                     MK4);

/* ===================== Int16x16x32 K12x8x1 algo ===================== */
namespace {
void int16x16x32_k12x8x1_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int16x16x32_k12x8x1_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int16>(),
                   Bptr = kern_param.B<dt_int16>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_s16_12x8x1 strategy(M, N, K, A_type, B_type,
                                                  C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s16_12x8x1>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt16x16x32K12x8x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode ==
                   param::MatrixMul::ComputeMode::DEFAULT &&
           kern_size_param.A_type.enumv() == DTypeEnum::Int16 &&
           kern_size_param.C_type.enumv() == DTypeEnum::Int32;
}

bool MatrixMulImpl::AlgoInt16x16x32K12x8x1::preferred(
        const KernSizeParam& kern_size_param) const {
    MEGDNN_MARK_USED_VAR(kern_size_param);
    return true;
}

size_t MatrixMulImpl::AlgoInt16x16x32K12x8x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt16x16x32K12x8x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s16_12x8x1 strategy(M, N, K, A_type, B_type,
                                                  C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s16_12x8x1>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt16x16x32K12x8x1::get_kern(
        const KernSizeParam&) const {
    return int16x16x32_k12x8x1_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt16x16x32K12x8x1,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt16x16x32K12x8x1Impl"_hash,
                                     aarch64::matmul::gemm_s16_12x8x1, int16_t,
                                     int32_t, AlgoDataType::INT16X16X32,
                                     DEFAULT);

/* ===================== Int16x16x32MK8_8x8 algo ===================== */

bool MatrixMulImpl::AlgoInt16x16x32MK8_8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.C_type == dtype::Int32() &&
           kern_size_param.B_type == dtype::Int16() &&
           kern_size_param.A_type == dtype::Int16() &&
           kern_size_param.format == param::MatrixMul::Format::MK8 &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoInt16x16x32MK8_8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt16x16x32MK8_8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_nopack_s16_8x8 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       aarch64::matmul::gemm_nopack_s16_8x8, false>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt16x16x32MK8_8x8::get_kern(
        const KernSizeParam&) const {
    auto kern_mk8_8x8 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                     midout_iv("AlgoInt16x16x32MK8_8x8::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<dt_int16>(),
                       Bptr = kern_param.B<dt_int16>();
            auto Cptr = kern_param.C<dt_int32>();

            aarch64::matmul::gemm_nopack_s16_8x8 strategy(A_type, B_type,
                                                          C_type);
            megdnn::matmul::GemmInterleaved<
                    aarch64::matmul::gemm_nopack_s16_8x8, false>(M, N, K, trA,
                                                                 trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return kern_mk8_8x8;
}

#if __ARM_FEATURE_DOTPROD
/* ==================== Quint8 K8x8x4 Dotprod algo ==================== */
namespace {
void quint8_k8x8x4_dotprod_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("quint8_k8x8x4_dotprod_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_uint8>(),
                   Bptr = kern_param.B<dt_uint8>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_u8_8x8 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_u8_8x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoQuint8K8x8x4DotProd::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.B_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32 &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoQuint8K8x8x4DotProd::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoQuint8K8x8x4DotProd::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;

        aarch64::matmul::gemm_u8_8x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_u8_8x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoQuint8K8x8x4DotProd::get_kern(
        const KernSizeParam&) const {
    return quint8_k8x8x4_dotprod_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoQuint8K8x8x4DotProd,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoQuint8K8x8x4DotProdImpl"_hash,
                                     aarch64::matmul::gemm_u8_8x8, uint8_t,
                                     int32_t, AlgoDataType::QUINT8X8X32,
                                     DEFAULT);
/* ===================== Quint8 Gemv DotProd algo ===================== */
namespace {
void quint8_gemv_dotprod_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("quint8_gemv_dotprod_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_uint8>(),
                   Bptr = kern_param.B<dt_uint8>();
        auto Cptr = kern_param.C<dt_int32>();
        auto A_type = kern_param.A_type, B_type = kern_param.B_type;

        aarch64::matmul::gemv_like_quint8(
                Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC,
                A_type.param<dtype::Quantized8Asymm>().zero_point,
                B_type.param<dtype::Quantized8Asymm>().zero_point);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoQuint8GemvDotProd::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.B_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32 &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.N == 1 && kern_size_param.LDB == 1;
}

bool MatrixMulImpl::AlgoQuint8GemvDotProd::preferred(
        const KernSizeParam& kern_size_param) const {
    auto N = kern_size_param.N, LDB = kern_size_param.LDB;
    return (N == 1 && LDB == 1);
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoQuint8GemvDotProd::get_kern(
        const KernSizeParam&) const {
    return quint8_gemv_dotprod_kern;
}
#else

/* ===================== Quint8 K8x8x8 algo ===================== */
namespace {
void quint8_k8x8x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("quint8_gemv_dotprod_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_uint8>(),
                   Bptr = kern_param.B<dt_uint8>();
        auto Cptr = kern_param.C<dt_int32>();

        aarch64::matmul::gemm_u8_8x8 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_u8_8x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoQuint8K8x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.B_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32 &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoQuint8K8x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoQuint8K8x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;

        aarch64::matmul::gemm_u8_8x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_u8_8x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoQuint8K8x8x8::get_kern(
        const KernSizeParam&) const {
    return quint8_k8x8x8_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoQuint8K8x8x8,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoQuint8K8x8x8Impl"_hash,
                                     aarch64::matmul::gemm_u8_8x8, uint8_t,
                                     int32_t, AlgoDataType::QUINT8X8X32,
                                     DEFAULT);
#endif

/* ===================== Int8x8x16 K8x8x8 algo ===================== */
namespace {
void int8x8x16_mk4_8x8x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int8x8x16_mk4_8x8x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();

        aarch64::matmul::gemm_s8x8x16_mk4_8x8x8 strategy(M, N, K, A_type,
                                                             B_type, C_type);
        megdnn::matmul::GemmInterleaved<
                aarch64::matmul::gemm_s8x8x16_mk4_8x8x8>(M, N, K, trA, trB,
                                                             strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16MK4_K8x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x16(kern_size_param) &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.M % 4 == 0 && kern_size_param.K % 4 == 0;
}

bool MatrixMulImpl::AlgoInt8x8x16MK4_K8x8x8::preferred(
        const KernSizeParam&) const {
    return true;
}

size_t MatrixMulImpl::AlgoInt8x8x16MK4_K8x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt8x8x16_MK4_8x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s8x8x16_mk4_8x8x8 strategy(M, N, K, A_type,
                                                             B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       matmul::gemm_s8x8x16_mk4_8x8x8>(M, N, K, trA, trB,
                                                           strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16MK4_K8x8x8::get_kern(
        const KernSizeParam&) const {
    return int8x8x16_mk4_8x8x8_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x16MK4_K8x8x8,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt8x8x16MK4_K8x8x8Impl"_hash,
                                     aarch64::matmul::gemm_s8x8x16_mk4_8x8x8,
                                     int8_t, int16_t, AlgoDataType::INT8X8X16,
                                     MK4);
/* ===================== Int4x4x16 K8x8x8 algo ===================== */
namespace {
void int4x4x16_k8x8x16_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("int4x4x16_k8x8x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();

        aarch64::matmul::gemm_s4x4x16_s4_8x8x8 strategy(M, N, K, A_type, B_type,
                                                   C_type);

        megdnn::matmul::GemmInterleaved<aarch64::matmul::gemm_s4x4x16_s4_8x8x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt4x4x16K8x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS4 &&
           kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS16 &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           (kern_size_param.K & 1) == 0 && (kern_size_param.N & 1) == 0;
}

bool MatrixMulImpl::AlgoInt4x4x16K8x8x8::preferred(
        const KernSizeParam& kern_size_param) const {
    MEGDNN_MARK_USED_VAR(kern_size_param);
    return true;
}

size_t MatrixMulImpl::AlgoInt4x4x16K8x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_aarch64_matmul_kern,
                 midout_iv("AlgoInt4x4x16K8x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        aarch64::matmul::gemm_s4x4x16_s4_8x8x8 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s4x4x16_s4_8x8x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt4x4x16K8x8x8::get_kern(
        const KernSizeParam&) const {
    return int4x4x16_k8x8x16_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt4x4x16K8x8x8,
                                     megdnn_aarch64_matmul_kern,
                                     "AlgoInt4x4x16K8x8x8Impl"_hash,
                                     aarch64::matmul::gemm_s4x4x16_s4_8x8x8,
                                     int8_t, int16_t, AlgoDataType::INT4X4X16,
                                     DEFAULT);
// vim: syntax=cpp.doxygen
