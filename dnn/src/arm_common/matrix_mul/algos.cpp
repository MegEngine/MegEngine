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
MIDOUT_DECL(megdnn_arm_exec_int8832)
MIDOUT_DECL(megdnn_arm_exec_fp32)

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
        exec_gemm_int8_int8_int16(
                kern_param.A<dt_int8>(), kern_param.B<dt_int8>(),
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
    MIDOUT_BEGIN(megdnn_arm_exec_int8816,
                 midout_iv("AlgoInt8x8x16::get_workspace"_hash)) {
        auto wbundle = get_workspace_bundle_int_8x8x16(kern_size_param);
        return wbundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16::get_kern(
        const KernSizeParam&) const {
    return exec_int_8x8x16;
}

/* ===================== Int8x8x32 Gemv algo ===================== */
namespace {
void int8x8x32_gemv_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_int8832,
                 midout_iv("int8x8x32_gemv_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        gemv_like(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
    }
    MIDOUT_END();
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

/* ===================== Int8x8x32 Gemv MK4 algo ===================== */
namespace {
void int8x8x32_gemv_mk4_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_int8832,
                 midout_iv("int8x8x32_gemv_mk4_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        gemv_like_mk4(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32GemvMK4::usable(
        const KernSizeParam& kern_size_param) const {
    auto M = kern_size_param.M;
    auto N = kern_size_param.N;
    auto K = kern_size_param.K;
    auto LDB = kern_size_param.LDB;

    bool is_dtype_ok =
            kern_size_param.A_type == kern_size_param.B_type &&
            (kern_size_param.A_type.enumv() == DTypeEnum::Int8 ||
             kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
            (kern_size_param.C_type.enumv() == DTypeEnum::Int32 ||
             kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32);

    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           is_dtype_ok && !kern_size_param.trA && !kern_size_param.trB &&
           M % 4 == 0 && K % 4 == 0 && N == 1 && LDB == 4;
}

bool MatrixMulImpl::AlgoInt8x8x32GemvMK4::preferred(
        const KernSizeParam& kern_size_param) const {
    MEGDNN_MARK_USED_VAR(kern_size_param);
    return true;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32GemvMK4::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_gemv_mk4_kern;
}

#if __ARM_FEATURE_DOTPROD
/* =================== Int8x8x32 Gemv MK4_DOT algo ==================== */
namespace {
void int8x8x32_gemv_mk4_dot_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_int8832,
                 midout_iv("int8x8x32_gemv_mk4_dot_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_int8>(), Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        gemv_like_mk4_dot(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoInt8x8x32GemvMK4Dot::usable(
        const KernSizeParam& kern_size_param) const {
    auto M = kern_size_param.M;
    auto N = kern_size_param.N;
    auto K = kern_size_param.K;
    auto LDB = kern_size_param.LDB;

    bool is_dtype_ok =
            kern_size_param.A_type == kern_size_param.B_type &&
            (kern_size_param.A_type.enumv() == DTypeEnum::Int8 ||
             kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
            (kern_size_param.C_type.enumv() == DTypeEnum::Int32 ||
             kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32);

    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4_DOT &&
           is_dtype_ok && !kern_size_param.trA && !kern_size_param.trB &&
           M % 4 == 0 && K % 4 == 0 && N == 1 && LDB == 4;
}

bool MatrixMulImpl::AlgoInt8x8x32GemvMK4Dot::preferred(
        const KernSizeParam& kern_size_param) const {
    return true;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32GemvMK4Dot::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_gemv_mk4_dot_kern;
}
#endif

/* ===================== F32 Gemv algo ===================== */
namespace {
void f32_gemv_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_fp32,
                 midout_iv("f32_gemv_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_float32>(),
                Bptr = kern_param.B<dt_float32>();
        auto Cptr = kern_param.C<dt_float32>();
        gemv_like(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
    }
    MIDOUT_END();
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

/* ================== F32 Gemv MK4 algo ================== */
namespace {
void f32_gemv_mk4_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_fp32,
                 midout_iv("f32_gemv_mk4_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_float32>(),
                Bptr = kern_param.B<dt_float32>();
        auto Cptr = kern_param.C<dt_float32>();
        gemv_like_mk4(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoF32GemvMK4::usable(
        const KernSizeParam& kern_size_param) const {
    // enumerate the M, N, K, only usable when preferred
    auto M = kern_size_param.M;
    auto N = kern_size_param.N;
    auto K = kern_size_param.K;
    auto LDB = kern_size_param.LDB;

    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() && !kern_size_param.trA &&
           !kern_size_param.trB && M % 4 == 0 && K % 4 == 0 && N == 1 &&
           LDB == 4;
}

bool MatrixMulImpl::AlgoF32GemvMK4::preferred(
        const KernSizeParam& kern_size_param) const {
    MEGDNN_MARK_USED_VAR(kern_size_param);
    return true;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32GemvMK4::get_kern(
        const KernSizeParam&) const {
    return f32_gemv_mk4_kern;
}

/* ===================== F32 Gevm algo ===================== */
namespace {
template <typename stype, typename dtype>
void gevm_like_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_arm_exec_fp32,
                 midout_iv("gevm_like_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDB = kern_param.LDB;
        const auto Aptr = kern_param.A<stype>(), Bptr = kern_param.B<stype>();
        auto Cptr = kern_param.C<dtype>();
        megdnn::arm_common::gemv_like(Bptr, Aptr, Cptr, N, M, K, LDB, 1, 1);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoGevm::usable(
        const KernSizeParam& kern_size_param) const {
    // enumerate the M, N, K, only usable when preferred
    bool fp32_ok =
            kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
            kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
            kern_size_param.B_type == kern_size_param.A_type &&
            kern_size_param.C_type == kern_size_param.A_type &&
            kern_size_param.A_type == dtype::Float32();
    bool fp16_ok = false;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    fp16_ok = kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
              kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
              kern_size_param.B_type == kern_size_param.A_type &&
              kern_size_param.C_type == kern_size_param.A_type &&
              kern_size_param.A_type == dtype::Float16();
#endif
    bool int8_ok = can_be_treated_as_int8x8x32(kern_size_param);
    return (fp32_ok || fp16_ok || int8_ok) && preferred(kern_size_param);
}

bool MatrixMulImpl::AlgoGevm::preferred(
        const KernSizeParam& kern_size_param) const {
    auto M = kern_size_param.M;
    return kern_size_param.trB && M == 1;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoGevm::get_kern(
        const KernSizeParam& kern_size_param) const {
    if (kern_size_param.A_type == dtype::Float32()) {
        return gevm_like_kern<dt_float32, dt_float32>;
    } else if (kern_size_param.A_type.enumv() == DTypeEnum::Int8 ||
               kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8) {
        return gevm_like_kern<dt_int8, dt_int32>;
    }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if (kern_size_param.A_type == dtype::Float16()) {
        return gevm_like_kern<__fp16, __fp16>;
    }
#endif
    else {
        megdnn_assert(
                false, "no avaliable kern got A_type: %s B_type: %s C_type: %s",
                kern_size_param.A_type.name(), kern_size_param.B_type.name(),
                kern_size_param.C_type.name());
    }
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
        arm_common::gemv_like(reinterpret_cast<const __fp16*>(Aptr),
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
