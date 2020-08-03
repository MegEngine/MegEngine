/**
 * \file dnn/src/armv7/matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/armv7/matrix_mul/algos.h"
#include "src/armv7/matrix_mul/fp16/strategy.h"
#include "src/armv7/matrix_mul/fp32/strategy.h"
#include "src/armv7/matrix_mul/int16x16x32/strategy.h"
#include "src/armv7/matrix_mul/int8/strategy.h"
#include "src/armv7/matrix_mul/int8x8x16/strategy.h"
#include "src/armv7/matrix_mul/quint8/strategy.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_impl.h"

#include "midout.h"

MIDOUT_DECL(megdnn_armv7_matmul_kern)

using namespace megdnn;
using namespace armv7;

/* ===================== F32 algo ===================== */

namespace {
void f32_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern, midout_iv("f32_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<float>(), Bptr = kern_param.B<float>();
        auto Cptr = kern_param.C<float>();

        armv7::matmul::sgemm_4x12 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::sgemm_4x12>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace

bool MatrixMulImpl::AlgoF32::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32();
}

size_t MatrixMulImpl::AlgoF32::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoF32::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::sgemm_4x12 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<armv7::matmul::sgemm_4x12>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32::get_kern(
        const KernSizeParam&) const {
    return f32_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF32, megdnn_armv7_matmul_kern,
                                     "AlgoF32Impl"_hash,
                                     armv7::matmul::sgemm_4x12, float, float,
                                     AlgoDataType::FLOAT32, DEFAULT);

/* ===================== F32 algo mk4 K4x12 ===================== */

namespace {
void f32_mk4_pack_4x12_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("f32_mk4_pack_4x12_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<float>(), Bptr = kern_param.B<float>();
        auto Cptr = kern_param.C<float>();

        armv7::matmul::sgemm_mk4_pack_4x12 strategy(M, N, K, A_type, B_type,
                                                    C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::sgemm_mk4_pack_4x12>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace

bool MatrixMulImpl::AlgoF32MK4Pack4x12::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() && !kern_size_param.trA &&
           !kern_size_param.trB && kern_size_param.M % 4 == 0 &&
           kern_size_param.K % 4 == 0 && !kern_size_param.trA &&
           !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoF32MK4Pack4x12::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoF32MK4Pack4x12::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::sgemm_mk4_pack_4x12 strategy(M, N, K, A_type, B_type,
                                                    C_type);
        return megdnn::matmul::GemmInterleaved<
                       armv7::matmul::sgemm_mk4_pack_4x12>(M, N, K, trA, trB,
                                                           strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32MK4Pack4x12::get_kern(
        const KernSizeParam&) const {
    return f32_mk4_pack_4x12_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF32MK4Pack4x12,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoF32MK4Pack4x12"_hash,
                                     armv7::matmul::sgemm_mk4_pack_4x12, float,
                                     float, AlgoDataType::FLOAT32, MK4);

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/* ===================== F16 K4x16x1 algo ===================== */
namespace {
void f16_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern, midout_iv("f16_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_float16>(),
                   Bptr = kern_param.B<dt_float16>();
        auto Cptr = kern_param.C<dt_float16>();

        armv7::matmul::hgemm_4x16 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::hgemm_4x16>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoF16K4x16x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float16();
}

size_t MatrixMulImpl::AlgoF16K4x16x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoF16K4x16x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::hgemm_4x16 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<armv7::matmul::hgemm_4x16>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF16K4x16x1::get_kern(
        const KernSizeParam&) const {
    return f16_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoF16K4x16x1, megdnn_armv7_matmul_kern,
                                     "AlgoF16K4x16x1"_hash,
                                     armv7::matmul::hgemm_4x16, dt_float16,
                                     dt_float16, AlgoDataType::FLOAT16,
                                     DEFAULT);

#endif

/* ===================== Int8x8x32 Kernel 4x2x16 algo ===================== */

namespace {
void kern_int8x8x32_k4x2x16(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int8x8x32_k4x2x16"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_s8_4x2 strategy(M, N, K, kern_param.A_type,
                                            kern_param.B_type,
                                            kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_s8_4x2>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32K4x2x16::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x32(kern_size_param);
}

bool MatrixMulImpl::AlgoInt8x8x32K4x2x16::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K > 32;
}

size_t MatrixMulImpl::AlgoInt8x8x32K4x2x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x32K4x2x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_s8_4x2 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8_4x2>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32K4x2x16::get_kern(
        const KernSizeParam&) const {
    return kern_int8x8x32_k4x2x16;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32K4x2x16,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x32K4x2x16"_hash,
                                     armv7::matmul::gemm_s8_4x2, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     DEFAULT);
/* ===================== Int8x8x32 Kernel 4x8x8 algo ===================== */

namespace {
void kern_int8x8x32_k4x8x8(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int8x8x32_k4x8x8"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_s8_4x8 strategy(M, N, K, kern_param.A_type,
                                            kern_param.B_type,
                                            kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_s8_4x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32K4x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x32(kern_size_param);
}

bool MatrixMulImpl::AlgoInt8x8x32K4x8x8::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K <= 32;
}

size_t MatrixMulImpl::AlgoInt8x8x32K4x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x32K4x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_s8_4x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8_4x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32K4x8x8::get_kern(
        const KernSizeParam&) const {
    return kern_int8x8x32_k4x8x8;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32K4x8x8,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x32K4x8x8"_hash,
                                     armv7::matmul::gemm_s8_4x8, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     DEFAULT);
/* ===================== Quint8 Kernel 4x8x8 algo ===================== */

namespace {
void kern_quint8_k4x8x8(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_quint8_k4x8x8"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_uint8>(), Bptr = kern_param.B<dt_uint8>();
        auto Cptr = kern_param.C<dt_int32>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_u8_4x8 strategy(M, N, K, kern_param.A_type,
                                            kern_param.B_type,
                                            kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_u8_4x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoQuint8K4x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.B_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32 &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoQuint8K4x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoQuint8K4x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_u8_4x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_u8_4x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoQuint8K4x8x8::get_kern(
        const KernSizeParam&) const {
    return kern_quint8_k4x8x8;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoQuint8K4x8x8, megdnn_armv7_matmul_kern,
                                     "AlgoQuint8K4x8x8"_hash,
                                     armv7::matmul::gemm_u8_4x8, uint8_t,
                                     int32_t, AlgoDataType::QUINT8X8X32,
                                     DEFAULT);
/* ===================== Int8x8x16 Kernel 2x4x16 algo ===================== */

namespace {
void kern_int8x8x16_k2x4x16(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int8x8x16_k2x4x16"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_s8x8x16_4x2 strategy(M, N, K, kern_param.A_type,
                                                 kern_param.B_type,
                                                 kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_s8x8x16_4x2>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16K4x2x16::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type == kern_size_param.B_type &&
           kern_size_param.A_type == dtype::Int8() &&
           kern_size_param.C_type == dtype::Int16() &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoInt8x8x16K4x2x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x16K4x2x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_s8x8x16_4x2 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8x8x16_4x2>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16K4x2x16::get_kern(
        const KernSizeParam&) const {
    return kern_int8x8x16_k2x4x16;
}

bool MatrixMulImpl::AlgoInt8x8x16K4x2x16::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K > 128;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x16K4x2x16,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x16K4x2x16"_hash,
                                     armv7::matmul::gemm_s8x8x16_4x2, int8_t,
                                     int16_t, AlgoDataType::INT8X8X16, DEFAULT);
/* ===================== Int8x8x16 Kernel 4x8x8 algo ===================== */

namespace {
void kern_int8x8x16_k4x8x8(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int8x8x16_k4x8x8"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_s8x8x16_4x8 strategy(M, N, K, kern_param.A_type,
                                                 kern_param.B_type,
                                                 kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_s8x8x16_4x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16K4x8x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type == kern_size_param.B_type &&
           kern_size_param.A_type == dtype::Int8() &&
           kern_size_param.C_type == dtype::Int16() &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoInt8x8x16K4x8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x16K4x8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_s8x8x16_4x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8x8x16_4x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16K4x8x8::get_kern(
        const KernSizeParam&) const {
    return kern_int8x8x16_k4x8x8;
}

bool MatrixMulImpl::AlgoInt8x8x16K4x8x8::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K >= 8 && kern_size_param.K <= 128;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x16K4x8x8,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x16K4x8x8"_hash,
                                     armv7::matmul::gemm_s8x8x16_4x8, int8_t,
                                     int16_t, AlgoDataType::INT8X8X16, DEFAULT);

/* =================== Int8x8x16 Kernel MK4 8x8x4 algo ===================*/

namespace {
void kern_int8x8x16_mk4_k8x8x4(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int8x8x16_mk4_k8x8x4"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int16>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_s8x8x16_mk4_8x8 strategy(M, N, K, kern_param.A_type,
                                                     kern_param.B_type,
                                                     kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_s8x8x16_mk4_8x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16MK4_8x8x4::usable(
        const KernSizeParam& kern_size_param) const {
    bool type_ok = can_be_treated_as_int8x8x16(kern_size_param);

    return type_ok && kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.M % 4 == 0 && kern_size_param.K % 4 == 0;
}

size_t MatrixMulImpl::AlgoInt8x8x16MK4_8x8x4::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x16K8x8x4::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_s8x8x16_mk4_8x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s8x8x16_mk4_8x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16MK4_8x8x4::get_kern(
        const KernSizeParam&) const {
    return kern_int8x8x16_mk4_k8x8x4;
}

bool MatrixMulImpl::AlgoInt8x8x16MK4_8x8x4::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K >= 4;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(AlgoInt8x8x16MK4_8x8x4,
                                            megdnn_armv7_matmul_kern,
                                            "AlgoInt8x8x16MK4_8x8x4"_hash,
                                            armv7::matmul::gemm_s8x8x16_mk4_8x8,
                                            int8_t, int16_t, int16_t,
                                            AlgoDataType::INT8X8X16, MK4);

/* ===================== Int16x16x32 Kernel 12x4x1 algo ===================== */

namespace {
void kern_int16x16x32K12x4x1(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int16x16x32K12x4x1"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int16>(), Bptr = kern_param.B<dt_int16>();
        auto Cptr = kern_param.C<dt_int32>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_s16x16x32_12x4 strategy(M, N, K, kern_param.A_type,
                                                    kern_param.B_type,
                                                    kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_s16x16x32_12x4>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace
bool MatrixMulImpl::AlgoInt16x16x32K12x4x1::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type == kern_size_param.B_type &&
           kern_size_param.A_type == dtype::Int16() &&
           kern_size_param.C_type == dtype::Int32() &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoInt16x16x32K12x4x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt16x16x32K12x4x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_s16x16x32_12x4 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_s16x16x32_12x4>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt16x16x32K12x4x1::get_kern(
        const KernSizeParam&) const {
    return kern_int16x16x32K12x4x1;
}

bool MatrixMulImpl::AlgoInt16x16x32K12x4x1::preferred(
        const KernSizeParam& /*kern_size_param*/) const {
    return true;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt16x16x32K12x4x1,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt16x16x32K12x4x1"_hash,
                                     armv7::matmul::gemm_s16x16x32_12x4,
                                     int16_t, int32_t,
                                     AlgoDataType::INT16X16X32, DEFAULT);
#if __ARM_FEATURE_DOTPROD
/* ===================== Int8 K6x8x4 algo ===================== */
namespace {
void int8_k6x8x4_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern, midout_iv("int8_k6x8x4_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        armv7::matmul::gemm_dots8_6x8 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_dots8_6x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // namespace

bool MatrixMulImpl::AlgoInt8x8x32K6x8x4::usable(
        const KernSizeParam& kern_size_param) const {
    return can_be_treated_as_int8x8x32(kern_size_param);
}

size_t MatrixMulImpl::AlgoInt8x8x32K6x8x4::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x32K6x8x4::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::gemm_dots8_6x8 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_dots8_6x8>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32K6x8x4::get_kern(
        const KernSizeParam&) const {
    return int8_k6x8x4_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32K6x8x4,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x32K6x8x4"_hash,
                                     armv7::matmul::gemm_dots8_6x8, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32,
                                     DEFAULT);
/* ===================== Quint8 K4x8x4 algo ===================== */
namespace {
void quint8_dot_k4x8x4_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("quint8_dot_k4x8x4_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_uint8>(),
                   Bptr = kern_param.B<dt_uint8>();
        auto Cptr = kern_param.C<dt_int32>();
        armv7::matmul::gemm_dot_quint8_4x8 strategy(M, N, K, A_type, B_type,
                                                    C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_dot_quint8_4x8>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // namespace

bool MatrixMulImpl::AlgoQuint8DotK4x8x4::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.B_type.enumv() == DTypeEnum::Quantized8Asymm &&
           kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32 &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT;
}

size_t MatrixMulImpl::AlgoQuint8DotK4x8x4::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoQuint8DotK4x8x4::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::gemm_dot_quint8_4x8 strategy(M, N, K, A_type, B_type,
                                                    C_type);
        return megdnn::matmul::GemmInterleaved<
                       armv7::matmul::gemm_dot_quint8_4x8>(M, N, K, trA, trB,
                                                           strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoQuint8DotK4x8x4::get_kern(
        const KernSizeParam&) const {
    return quint8_dot_k4x8x4_kern;
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoQuint8DotK4x8x4,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoQuint8DotK4x8x4"_hash,
                                     armv7::matmul::gemm_dot_quint8_4x8,
                                     uint8_t, int32_t,
                                     AlgoDataType::QUINT8X8X32, DEFAULT);

/* ======================== Int8 MK4 8x4x4 dot algo ======================== */
namespace {
void int8_mk4_8x4x4_dotprod_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("int8_mk4_8x4x4_dotprod_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        armv7::matmul::gemm_mk4_dots8_8x4 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_mk4_dots8_8x4>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // namespace

bool MatrixMulImpl::AlgoInt8x8x32MK4_8x4x4DotProd::usable(
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

size_t MatrixMulImpl::AlgoInt8x8x32MK4_8x4x4DotProd::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_armv7_matmul_kern,
            midout_iv("AlgoInt8x8x32MK4_8x4x4DotProd::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::gemm_mk4_dots8_8x4 strategy(M, N, K, A_type, B_type,
                                                   C_type);
        return megdnn::matmul::GemmInterleaved<
                       armv7::matmul::gemm_mk4_dots8_8x4>(M, N, K, trA, trB,
                                                          strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32MK4_8x4x4DotProd::get_kern(
        const KernSizeParam&) const {
    return int8_mk4_8x4x4_dotprod_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32MK4_8x4x4DotProd,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x32MK4_8x4x4DotProd"_hash,
                                     armv7::matmul::gemm_mk4_dots8_8x4, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32, MK4_DOT);
#endif

/* ===================== F32 algo K4x8 ===================== */

namespace {
void f32_mk4_4x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern, midout_iv("f32_mk4_4x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<float>(), Bptr = kern_param.B<float>();
        auto Cptr = kern_param.C<float>();

        armv7::matmul::sgemm_nopack_4x8 strategy(A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::sgemm_nopack_4x8, false>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace

bool MatrixMulImpl::AlgoF32MK4_4x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoF32MK4_4x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoF32MK4_4x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::sgemm_nopack_4x8 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<armv7::matmul::sgemm_nopack_4x8,
                                               false>(M, N, K, trA, trB,
                                                      strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32MK4_4x8::get_kern(
        const KernSizeParam&) const {
    return f32_mk4_4x8_kern;
}

/* ===================== Int16x16x32 MK8 4x8 algo ===================== */

bool MatrixMulImpl::AlgoInt16x16x32MK8_4x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK8 &&
           kern_size_param.A_type == dtype::Int16() &&
           kern_size_param.B_type == dtype::Int16() &&
           kern_size_param.C_type == dtype::Int32() &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoInt16x16x32MK8_4x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt16x16x32MK8_4x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::gemm_nopack_s16_4x8 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       armv7::matmul::gemm_nopack_s16_4x8, false>(M, N, K, trA,
                                                                  trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt16x16x32MK8_4x8::get_kern(
        const KernSizeParam&) const {
    auto kern_mk8_4x8 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                     midout_iv("AlgoInt16x16x32MK8_4x8::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<dt_int16>(),
                       Bptr = kern_param.B<dt_int16>();
            auto Cptr = kern_param.C<dt_int32>();

            armv7::matmul::gemm_nopack_s16_4x8 strategy(A_type, B_type, C_type);
            megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_nopack_s16_4x8,
                                            false>(M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return kern_mk8_4x8;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/* ===================== F16_MK8_4x8 algo ===================== */

bool MatrixMulImpl::AlgoF16MK8_4x8::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float16() &&
           kern_size_param.format == param::MatrixMul::Format::MK8 &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoF16MK8_4x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoF16MK8_4x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        armv7::matmul::gemm_nopack_f16_4x8 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       armv7::matmul::gemm_nopack_f16_4x8, false>(M, N, K, trA,
                                                                  trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF16MK8_4x8::get_kern(
        const KernSizeParam&) const {
    auto kern_mk8_4x8 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                     midout_iv("AlgoF16MK8_4x8::get_kern"_hash)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<dt_float16>(),
                       Bptr = kern_param.B<dt_float16>();
            auto Cptr = kern_param.C<dt_float16>();

            armv7::matmul::gemm_nopack_f16_4x8 strategy(A_type, B_type, C_type);
            megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_nopack_f16_4x8,
                                            false>(M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return kern_mk8_4x8;
}
#endif

/* ===================== Int8x8x16 Kernel 2x4x16 algo ===================== */

namespace {
void kern_int8x8x32_mk4_4x2x16(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("kern_int8x8x32_mk4_4x2x16"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto trA = kern_param.trA, trB = kern_param.trB;

        armv7::matmul::gemm_mk4_s8_4x2 strategy(M, N, K, kern_param.A_type,
                                                kern_param.B_type,
                                                kern_param.C_type);
        megdnn::matmul::GemmInterleaved<armv7::matmul::gemm_mk4_s8_4x2>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32MK4_4x2x16::usable(
        const KernSizeParam& param) const {
    return param.A_type.enumv() == param.B_type.enumv() &&
           (param.A_type.enumv() == DTypeEnum::Int8 ||
            param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
           (param.C_type.enumv() == DTypeEnum::Int32 ||
            param.C_type.enumv() == DTypeEnum::QuantizedS32) &&
           param.compute_mode == Param::ComputeMode::DEFAULT &&
           param.format == param::MatrixMul::Format::MK4 && param.M % 4 == 0 &&
           param.K % 4 == 0 && !param.trA && !param.trB;
}

size_t MatrixMulImpl::AlgoInt8x8x32MK4_4x2x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(megdnn_armv7_matmul_kern,
                 midout_iv("AlgoInt8x8x32MK4_4x2x16::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N,
             K = kern_size_param.K;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        matmul::gemm_mk4_s8_4x2 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::gemm_mk4_s8_4x2>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32MK4_4x2x16::get_kern(
        const KernSizeParam&) const {
    return kern_int8x8x32_mk4_4x2x16;
}

bool MatrixMulImpl::AlgoInt8x8x32MK4_4x2x16::preferred(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.K > 16;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32MK4_4x2x16,
                                     megdnn_armv7_matmul_kern,
                                     "AlgoInt8x8x32MK4_4x2x16"_hash,
                                     armv7::matmul::gemm_mk4_s8_4x2, int8_t,
                                     int32_t, AlgoDataType::QINT8X8X32, MK4);

// vim: syntax=cpp.doxygen
