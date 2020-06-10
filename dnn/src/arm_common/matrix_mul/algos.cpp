/**
 * \file dnn/src/arm_common/matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/matrix_mul/algos.h"
#include "src/arm_common/matrix_mul/exec_gemm_int8_int8_int16.h"
#include "src/arm_common/matrix_mul/fp16/hgemv.h"
#include "src/arm_common/matrix_mul/fp32/exec_sgemv.h"
#include "src/arm_common/matrix_mul/int8/gemv.h"
#include "midout.h"

MIDOUT_DECL(megdnn_arm_hgemv)
MIDOUT_DECL(megdnn_arm_exec_int8816)

using namespace megdnn;
using namespace arm_common;

/* ===================== Int8x8x16 algo ===================== */

namespace {
WorkspaceBundle get_workspace_bundle_int_8x8x16(
        const MatrixMulImpl::KernSizeParam& kern_size_param) {
    auto M = kern_size_param.M, K = kern_size_param.K, N = kern_size_param.N;
    // Use 8x8 tile
    return WorkspaceBundle(nullptr, {(M + 8) * K * sizeof(int8_t),
                                     K * (N + 8) * sizeof(int8_t)});
}

void exec_int_8x8x16(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_int8816, void) {
        auto bundle = get_workspace_bundle_int_8x8x16(kern_param);
        bundle.set(kern_param.workspace_ptr);
        auto w0 = static_cast<int8_t*>(bundle.get(0));
        auto w1 = static_cast<int8_t*>(bundle.get(1));
        size_t M = kern_param.M;
        size_t N = kern_param.N;
        size_t K = kern_param.K;
        size_t LDB = kern_param.LDB;
        exec_gemm_int8_int8_int16(kern_param.A<dt_int8>(),
                                  kern_param.B<dt_int8>(),
                                  kern_param.C<dt_int16>(), M, K, N, LDB, w0, w1);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x16::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type == dtype::Int8() &&
           kern_size_param.B_type == dtype::Int8() &&
           kern_size_param.C_type == dtype::Int16() &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoInt8x8x16::get_workspace(
        const KernSizeParam& kern_size_param) const {
    auto wbundle = get_workspace_bundle_int_8x8x16(kern_size_param);
    return wbundle.total_size_in_bytes();
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16::get_kern(
        const KernSizeParam&) const {
    return exec_int_8x8x16;
}

/* ===================== Int8x8x32 Gemv algo ===================== */
namespace {
void int8x8x32_gemv_kern(const MatrixMulImpl::KernParam& kern_param) {
    auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
    auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
    const auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
    auto Cptr = kern_param.C<dt_int32>();
    arm_common::matmul::gemv_like_int8(Aptr, Bptr, Cptr, M, N, K, LDA, LDB,
                                       LDC);
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32Gemv::usable(
        const KernSizeParam& kern_size_param) const {
    auto N = kern_size_param.N, LDB = kern_size_param.LDB;
    return can_be_treated_as_int8x8x32(kern_size_param) &&
           !kern_size_param.trA && !kern_size_param.trB && (N == 1 && LDB == 1);
}

bool MatrixMulImpl::AlgoInt8x8x32Gemv::preferred(
        const KernSizeParam& kern_size_param) const {
    auto N = kern_size_param.N, LDB = kern_size_param.LDB;
    return N == 1 && LDB == 1;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32Gemv::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_gemv_kern;
}

/* ===================== F32 Gemv algo ===================== */
namespace {
void f32_gemv_kern(const MatrixMulImpl::KernParam& kern_param) {
    auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
    auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
    const auto Aptr = kern_param.A<dt_float32>(),
               Bptr = kern_param.B<dt_float32>();
    auto Cptr = kern_param.C<dt_float32>();
    arm_common::sgemm_sgemv_like(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoF32Gemv::usable(
        const KernSizeParam& kern_size_param) const {
    // enumerate the M, N, K, only usable when preferred
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() && !kern_size_param.trA &&
           !kern_size_param.trB && preferred(kern_size_param);
}

bool MatrixMulImpl::AlgoF32Gemv::preferred(
        const KernSizeParam& kern_size_param) const {
    auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K,
         LDB = kern_size_param.LDB;

    return M < 8 || (M == 8 && K <= 2) || (N == 1 && LDB == 1);
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32Gemv::get_kern(
        const KernSizeParam&) const {
    return f32_gemv_kern;
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/* ===================== F16 Gemv algo ===================== */
namespace {
void f16_gemv_kern(const MatrixMulImpl::KernParam& kern_param) {
    auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
    auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
    const auto Aptr = kern_param.A<dt_float16>(),
               Bptr = kern_param.B<dt_float16>();
    auto Cptr = kern_param.C<dt_float16>();
    MIDOUT_BEGIN(megdnn_arm_hgemv, void) {
        arm_common::hgemv_exec(reinterpret_cast<const __fp16*>(Aptr),
                               reinterpret_cast<const __fp16*>(Bptr),
                               reinterpret_cast<__fp16*>(Cptr), M, N, K, LDA,
                               LDB, LDC);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoF16Gemv::usable(
        const KernSizeParam& kern_size_param) const {
    // enumerate the M, N, K, only usable when preferred
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float16() && !kern_size_param.trA &&
           !kern_size_param.trB && preferred(kern_size_param);
}

bool MatrixMulImpl::AlgoF16Gemv::preferred(
        const KernSizeParam& kern_size_param) const {
    auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K,
         LDB = kern_size_param.LDB;

    return M <= 4 || (M == 8 && K <= 2) || (N == 1 && LDB == 1);
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF16Gemv::get_kern(
        const KernSizeParam&) const {
    return f16_gemv_kern;
}
#endif

// vim: syntax=cpp.doxygen
