/**
 * \file dnn/src/x86/matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/utils.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/x86/matrix_mul/algos.h"
#include "src/x86/matrix_mul/f32/strategy.h"
#include "src/x86/matrix_mul/int8/strategy.h"

#include "midout.h"

MIDOUT_DECL(megdnn_x86_matmul_kern)
MIDOUT_DECL(megdnn_x86_matmul_kern_mk8_8x8)
using namespace megdnn;
using namespace x86;

/* ===================== F32 Blas algo ===================== */
namespace {

void f32_blas_kern(const MatrixMulImpl::KernParam& kern_param) {
#if MEGDNN_X86_WITH_MKL || MEGDNN_X86_WITH_OPENBLAS
    auto m = kern_param.M, n = kern_param.N, k = kern_param.K;
    bool trA = kern_param.trA, trB = kern_param.trB;
    const auto Aptr = kern_param.A<dt_float32>(),
               Bptr = kern_param.B<dt_float32>();
    auto Cptr = kern_param.C<dt_float32>();
    auto Atrd = kern_param.LDA, Btrd = kern_param.LDB, Ctrd = kern_param.LDC;
    disable_denorm();
    cblas_sgemm(CblasRowMajor, trA ? CblasTrans : CblasNoTrans,
                trB ? CblasTrans : CblasNoTrans, m, n, k, 1.0f, Aptr, Atrd,
                Bptr, Btrd, 0.0f, Cptr, Ctrd);
#else
    megdnn_throw("a blas library is required");
#endif
}

#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
void f32_blas_kern_only_packA(const MatrixMulImpl::KernParam& kern_param,
                              const void* a_panel, const void* b_panel) {
    MEGDNN_MARK_USED_VAR(b_panel);
    auto m = kern_param.M, n = kern_param.N, k = kern_param.K;
    const auto Bptr = kern_param.B<dt_float32>();
    auto Cptr = kern_param.C<dt_float32>();
    auto Atrd = kern_param.LDA, Btrd = kern_param.LDB, Ctrd = kern_param.LDC;
    disable_denorm();
    cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k,
                        static_cast<const float*>(a_panel), Atrd, Bptr, Btrd,
                        0.0f, Cptr, Ctrd);
}
#endif

}  // anonymous namespace

bool MatrixMulImpl::AlgoF32Blas::usable(
        const KernSizeParam& kern_size_param) const {
#if MEGDNN_X86_WITH_MKL || MEGDNN_X86_WITH_OPENBLAS
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() &&
           preferred(kern_size_param);
#else
    return false;
#endif
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32Blas::get_kern(
        const KernSizeParam&) const {
    return f32_blas_kern;
}

/* ===================== AlgoF32BlasPackA====================== */
#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
bool MatrixMulImpl::AlgoF32MKLPackA::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() &&
           preferred(kern_size_param);
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32MKLPackA::get_kern(
        const KernSizeParam&) const {
    return f32_blas_kern;
}

MatrixMulImpl::kern_naked_t MatrixMulImpl::AlgoF32MKLPackA::get_kern_naked(
        const KernSizeParam&) const {
    return f32_blas_kern_only_packA;
}

WorkspaceBundle MatrixMulImpl::AlgoF32MKLPackA::get_bundle(
        const KernSizeParam& param) const {
    auto M = param.M;
    auto N = param.N;
    auto K = param.K;
    size_t a_size = cblas_sgemm_pack_get_size(CblasAMatrix, M, N, K);
    return {nullptr, {a_size, 0, 0}};
}

void MatrixMulImpl::AlgoF32MKLPackA::pack_A(const KernParam& kern_param,
                                            void* out, size_t index,
                                            size_t stride) const {
    MEGDNN_MARK_USED_VAR(stride);
    MEGDNN_MARK_USED_VAR(index);
    auto m = kern_param.M, n = kern_param.N, k = kern_param.K;
    const auto Aptr = kern_param.A<dt_float32>();
    auto Atrd = kern_param.LDA;
    disable_denorm();
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, 1.0f,
                     Aptr, Atrd, static_cast<float*>(out));
}
#endif
/* ===================== Int8 Vnni algo ===================== */

#if MEGDNN_X86_WITH_VNNI
#define ALIGN_SIZE 64
namespace {
void int8x8x32_kern_vnni(const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_vnni, midout_iv(0)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_int8>(),
                   Bptr = kern_param.B<dt_int8>();
        auto Cptr = kern_param.C<dt_int32>();
        x86::matmul::gemm_int8_vnni_12x32x4 strategy(M, N, K, A_type, B_type,
                                                     C_type);
        megdnn::matmul::GemmInterleaved<x86::matmul::gemm_int8_vnni_12x32x4>(
                M, N, K, trA, trB, strategy, ALIGN_SIZE)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

size_t get_kern_workspace(MatrixMulImpl::KernSizeParam kern_size_param) {
    auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K;
    auto trA = kern_size_param.trA, trB = kern_size_param.trB;
    auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
         C_type = kern_size_param.C_type;
    x86::matmul::gemm_int8_vnni_12x32x4 strategy(M, N, K, A_type, B_type,
                                                 C_type);
    return megdnn::matmul::GemmInterleaved<x86::matmul::gemm_int8_vnni_12x32x4>(
                   M, N, K, trA, trB, strategy, ALIGN_SIZE)
            .get_workspace_size();
}
}  // namespace

bool MatrixMulImpl::AlgoInt8x8x32Vnni::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::Int32) ||
            (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32)) &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == Param::Format::DEFAULT &&
           preferred(kern_size_param) && is_supported(SIMDType::VNNI);
}

size_t MatrixMulImpl::AlgoInt8x8x32Vnni::get_workspace(
        const KernSizeParam& kern_size_param) const {
    return get_kern_workspace(kern_size_param);
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32Vnni::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_kern_vnni;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(
        AlgoInt8x8x32Vnni, megdnn_x86_matmul_kern, "AlgoInt8x8x32Vnni"_hash,
        x86::matmul::gemm_int8_vnni_12x32x4, dt_int8, dt_int32,
        dt_uint8AlgoDataType::QINT8X8X32, DEFAULT);
#endif

/* ===================== Int8 mkldnn algo ===================== */
#if MEGDNN_X86_WITH_MKL_DNN
namespace {
void int8x8x32_kern_mkldnn(const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_mkldnn, midout_iv(0)) {
        const char transA = kern_param.trA ? 'T' : 'N';
        const char transB = kern_param.trB ? 'T' : 'N';
        const char offsetC = 'F';
        const int64_t M = static_cast<int64_t>(kern_param.M);
        const int64_t N = static_cast<int64_t>(kern_param.N);
        const int64_t K = static_cast<int64_t>(kern_param.K);
        const int64_t LDA = static_cast<int64_t>(kern_param.LDA);
        const int64_t LDB = static_cast<int64_t>(kern_param.LDB);
        const int64_t LDC = static_cast<int64_t>(kern_param.LDC);

        const float alpha = 1.0f, beta = 0.0f;
        const int8_t ao = 0, bo = 0;
        const int32_t co = 0;
        const int8_t* A_ptr = static_cast<const int8_t*>(kern_param.A_ptr);
        const int8_t* B_ptr = static_cast<const int8_t*>(kern_param.B_ptr);
        int32_t* C_ptr = static_cast<int32_t*>(kern_param.C_ptr);
        auto status = mkldnn_gemm_s8s8s32(transA, transB, offsetC, M, N, K,
                                          alpha, A_ptr, LDA, ao, B_ptr, LDB, bo,
                                          beta, C_ptr, LDC, &co);
        megdnn_assert(status == mkldnn_success,
                      "mkldnn_gemm_s8s8s32 compute error!!!");
    }
    MIDOUT_END();
}
}  // namespace

bool MatrixMulImpl::AlgoInt8x8x32Mkldnn::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::Int32) ||
            (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32)) &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == Param::Format::DEFAULT &&
           is_supported(SIMDType::VNNI) && preferred(kern_size_param);
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32Mkldnn::get_kern(
        const KernSizeParam&) const {
    return int8x8x32_kern_mkldnn;
}
#endif

namespace {

void gemm_s8s8s32_avx2_2x4x16(const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_avx2_2x4x16, midout_iv(0)) {
        constexpr int cacheline = 64;
        const size_t m = kern_param.M;
        const size_t n = kern_param.N;
        const size_t k = kern_param.K;
        const bool trans_a = kern_param.trA;
        const bool trans_b = kern_param.trB;
        const size_t lda = kern_param.LDA;
        const size_t ldb = kern_param.LDB;
        const size_t ldc = kern_param.LDC;
        auto a_type = kern_param.A_type;
        auto b_type = kern_param.B_type;
        auto c_type = kern_param.C_type;
        const auto a_ptr = kern_param.A<dt_int8>();
        const auto b_ptr = kern_param.B<dt_int8>();
        auto c_ptr = kern_param.C<dt_int32>();
        x86::matmul::gemm_avx2_s8s8s32_2x4x16 strategy(m, n, k, a_type, b_type,
                                                       c_type);

        megdnn::matmul::GemmInterleaved<x86::matmul::gemm_avx2_s8s8s32_2x4x16>(
                m, n, k, trans_a, trans_b, strategy, cacheline)
                .execute(a_ptr, lda, b_ptr, ldb, c_ptr, ldc,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

void gemm_s8s8s32_avx2_4x16x2(const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_avx2_4x16x2, midout_iv(0)) {
        constexpr int cacheline = 64;
        const size_t m = kern_param.M;
        const size_t n = kern_param.N;
        const size_t k = kern_param.K;
        const bool trans_a = kern_param.trA;
        const bool trans_b = kern_param.trB;
        const size_t lda = kern_param.LDA;
        const size_t ldb = kern_param.LDB;
        const size_t ldc = kern_param.LDC;
        auto a_type = kern_param.A_type;
        auto b_type = kern_param.B_type;
        auto c_type = kern_param.C_type;
        const auto a_ptr = kern_param.A<dt_int8>();
        const auto b_ptr = kern_param.B<dt_int8>();
        auto c_ptr = kern_param.C<dt_int32>();
        x86::matmul::gemm_avx2_s8s8s32_4x16x2 strategy(m, n, k, a_type, b_type,
                                                       c_type);

        megdnn::matmul::GemmInterleaved<x86::matmul::gemm_avx2_s8s8s32_4x16x2>(
                m, n, k, trans_a, trans_b, strategy, cacheline)
                .execute(a_ptr, lda, b_ptr, ldb, c_ptr, ldc,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

void gemm_s8s8s32_sse_4x8x2(const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_sse_4x8x2, midout_iv(0)) {
        constexpr int cacheline = 64;
        x86::matmul::gemm_sse_s8s8s32_4x8x2 strategy(
                kern_param.M, kern_param.N, kern_param.K, kern_param.A_type,
                kern_param.B_type, kern_param.C_type);

        megdnn::matmul::GemmInterleaved<x86::matmul::gemm_sse_s8s8s32_4x8x2>(
                kern_param.M, kern_param.N, kern_param.K, kern_param.trA,
                kern_param.trB, strategy, cacheline)
                .execute(kern_param.A<dt_int8>(), kern_param.LDA,
                         kern_param.B<dt_int8>(), kern_param.LDB,
                         kern_param.C<dt_int32>(), kern_param.LDC,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // namespace

/*************************AlgoInt8x8x16AVX2********************/
void MatrixMulImpl::AlgoInt8x8x16AVX2::gemm_s8s8s16_avx2_4x16x2(
        const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_avx2_4x16x2, midout_iv(1)) {
        constexpr int cacheline = 64;
        const size_t m = kern_param.M;
        const size_t n = kern_param.N;
        const size_t k = kern_param.K;
        const bool trans_a = kern_param.trA;
        const bool trans_b = kern_param.trB;
        const size_t lda = kern_param.LDA;
        const size_t ldb = kern_param.LDB;
        const size_t ldc = kern_param.LDC;
        auto a_type = kern_param.A_type;
        auto b_type = kern_param.B_type;
        auto c_type = kern_param.C_type;
        const auto a_ptr = kern_param.A<dt_int8>();
        const auto b_ptr = kern_param.B<dt_int8>();
        auto c_ptr = kern_param.C<dt_int16>();
        x86::matmul::gemm_avx2_s8s8s16_4x16x2 strategy(m, n, k, a_type, b_type,
                                                       c_type);

        megdnn::matmul::GemmInterleaved<x86::matmul::gemm_avx2_s8s8s16_4x16x2>(
                m, n, k, trans_a, trans_b, strategy, cacheline)
                .execute(a_ptr, lda, b_ptr, ldb, c_ptr, ldc,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16AVX2::get_kern(
        const KernSizeParam&) const {
    return gemm_s8s8s16_avx2_4x16x2;
}
bool MatrixMulImpl::AlgoInt8x8x16AVX2::usable(
        const KernSizeParam& kern_size_param) const {
    bool is_ab_same =
            kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv();
    bool is_type_ok =
            ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
              kern_size_param.C_type.enumv() == DTypeEnum::Int16) ||
             (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
              kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS16));
    bool is_mode_ok =
            kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
            kern_size_param.format == Param::Format::DEFAULT &&
            is_supported(SIMDType::AVX2);
    bool is_param_ok = is_ab_same && is_type_ok && is_mode_ok;

    return is_param_ok;
}
bool MatrixMulImpl::AlgoInt8x8x16AVX2::preferred(const KernSizeParam&) const {
    return true;
}
size_t MatrixMulImpl::AlgoInt8x8x16AVX2::get_workspace(
        const KernSizeParam& kern_param) const {
    constexpr int cacheline = 64;
    const size_t m = kern_param.M;
    const size_t n = kern_param.N;
    const size_t k = kern_param.K;
    const bool trans_a = kern_param.trA;
    const bool trans_b = kern_param.trB;
    auto a_type = kern_param.A_type;
    auto b_type = kern_param.B_type;
    auto c_type = kern_param.C_type;
    x86::matmul::gemm_avx2_s8s8s16_4x16x2 strategy(m, n, k, a_type, b_type,
                                                   c_type);

    return megdnn::matmul::GemmInterleaved<
                   x86::matmul::gemm_avx2_s8s8s16_4x16x2>(
                   m, n, k, trans_a, trans_b, strategy, cacheline)
            .get_workspace_size();
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(
        AlgoInt8x8x16AVX2, megdnn_x86_matmul_kern, "AlgoInt8x8x16AVX2"_hash,
        x86::matmul::gemm_avx2_s8s8s16_4x16x2, dt_int8, dt_int16, dt_int16,
        AlgoDataType::INT8X8X16, DEFAULT);

/*************************AlgoInt8x8x16SSE********************/
void MatrixMulImpl::AlgoInt8x8x16SSE::gemm_s8s8s16_sse_4x8x2(
        const MatrixMulImpl::KernParam& kern_param) {
    MEGDNN_MARK_USED_VAR(kern_param);
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_sse_4x8x2, midout_iv(2)) {
        constexpr int cacheline = 64;
        const size_t m = kern_param.M;
        const size_t n = kern_param.N;
        const size_t k = kern_param.K;
        const bool trans_a = kern_param.trA;
        const bool trans_b = kern_param.trB;
        const size_t lda = kern_param.LDA;
        const size_t ldb = kern_param.LDB;
        const size_t ldc = kern_param.LDC;
        auto a_type = kern_param.A_type;
        auto b_type = kern_param.B_type;
        auto c_type = kern_param.C_type;
        const auto a_ptr = kern_param.A<dt_int8>();
        const auto b_ptr = kern_param.B<dt_int8>();
        auto c_ptr = kern_param.C<dt_int16>();
        x86::matmul::gemm_sse_s8s8s16_4x8x2 strategy(m, n, k, a_type, b_type,
                                                     c_type);

        megdnn::matmul::GemmInterleaved<x86::matmul::gemm_sse_s8s8s16_4x8x2>(
                m, n, k, trans_a, trans_b, strategy, cacheline)
                .execute(a_ptr, lda, b_ptr, ldb, c_ptr, ldc,
                         kern_param.workspace_ptr);
    }
    MIDOUT_END();
}
MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x16SSE::get_kern(
        const KernSizeParam&) const {
    return gemm_s8s8s16_sse_4x8x2;
}
bool MatrixMulImpl::AlgoInt8x8x16SSE::usable(
        const KernSizeParam& kern_size_param) const {
    bool is_ab_same =
            kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv();
    bool is_type_ok =
            ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
              kern_size_param.C_type.enumv() == DTypeEnum::Int16) ||
             (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
              kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS16));
    bool is_mode_ok =
            kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
            kern_size_param.format == Param::Format::DEFAULT &&
            is_supported(SIMDType::SSE4_1);
    bool is_param_ok = is_ab_same && is_type_ok && is_mode_ok;
    return is_param_ok;
}
bool MatrixMulImpl::AlgoInt8x8x16SSE::preferred(const KernSizeParam&) const {
    return true;
}
size_t MatrixMulImpl::AlgoInt8x8x16SSE::get_workspace(
        const KernSizeParam& kern_param) const {
    constexpr int cacheline = 64;
    const size_t m = kern_param.M;
    const size_t n = kern_param.N;
    const size_t k = kern_param.K;
    const bool trans_a = kern_param.trA;
    const bool trans_b = kern_param.trB;
    auto a_type = kern_param.A_type;
    auto b_type = kern_param.B_type;
    auto c_type = kern_param.C_type;
    x86::matmul::gemm_sse_s8s8s16_4x8x2 strategy(m, n, k, a_type, b_type,
                                                 c_type);

    return megdnn::matmul::GemmInterleaved<x86::matmul::gemm_sse_s8s8s16_4x8x2>(
                   m, n, k, trans_a, trans_b, strategy, cacheline)
            .get_workspace_size();
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(AlgoInt8x8x16SSE,
                                            megdnn_x86_matmul_kern,
                                            "AlgoInt8x8x16SSE"_hash,
                                            x86::matmul::gemm_sse_s8s8s16_4x8x2,
                                            dt_int8, dt_int16, dt_int16,
                                            AlgoDataType::INT8X8X16, DEFAULT);

/*************************AlgoInt8x8x32AVX2M4N16K2********************/
MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32AVX2M4N16K2::get_kern(
        const KernSizeParam&) const {
    return gemm_s8s8s32_avx2_4x16x2;
}
bool MatrixMulImpl::AlgoInt8x8x32AVX2M4N16K2::usable(
        const KernSizeParam& kern_size_param) const {
    bool is_param_ok =
            kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
            ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
              kern_size_param.C_type.enumv() == DTypeEnum::Int32) ||
             (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
              kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32)) &&
            kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
            kern_size_param.format == Param::Format::DEFAULT &&
            is_supported(SIMDType::AVX2);
    return is_param_ok;
}
size_t MatrixMulImpl::AlgoInt8x8x32AVX2M4N16K2::get_workspace(
        const KernSizeParam& kern_param) const {
    constexpr int cacheline = 64;
    const size_t m = kern_param.M;
    const size_t n = kern_param.N;
    const size_t k = kern_param.K;
    const bool trans_a = kern_param.trA;
    const bool trans_b = kern_param.trB;
    auto a_type = kern_param.A_type;
    auto b_type = kern_param.B_type;
    auto c_type = kern_param.C_type;
    x86::matmul::gemm_avx2_s8s8s32_4x16x2 strategy(m, n, k, a_type, b_type,
                                                   c_type);

    return megdnn::matmul::GemmInterleaved<
                   x86::matmul::gemm_avx2_s8s8s32_4x16x2>(
                   m, n, k, trans_a, trans_b, strategy, cacheline)
            .get_workspace_size();
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(
        AlgoInt8x8x32AVX2M4N16K2, megdnn_x86_matmul_kern,
        "AlgoInt8x8x32AVX2M4N16K2"_hash, x86::matmul::gemm_avx2_s8s8s32_4x16x2,
        dt_int8, dt_int32, dt_int16, AlgoDataType::QINT8X8X32, DEFAULT);

MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32AVX2M2N4K16::get_kern(
        const KernSizeParam&) const {
    return gemm_s8s8s32_avx2_2x4x16;
}
bool MatrixMulImpl::AlgoInt8x8x32AVX2M2N4K16::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::Int32) ||
            (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32)) &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == Param::Format::DEFAULT &&
           is_supported(SIMDType::AVX2);
}
size_t MatrixMulImpl::AlgoInt8x8x32AVX2M2N4K16::get_workspace(
        const KernSizeParam& kern_param) const {
    constexpr int cacheline = 64;
    const size_t m = kern_param.M;
    const size_t n = kern_param.N;
    const size_t k = kern_param.K;
    const bool trans_a = kern_param.trA;
    const bool trans_b = kern_param.trB;
    auto a_type = kern_param.A_type;
    auto b_type = kern_param.B_type;
    auto c_type = kern_param.C_type;
    x86::matmul::gemm_avx2_s8s8s32_2x4x16 strategy(m, n, k, a_type, b_type,
                                                   c_type);

    return megdnn::matmul::GemmInterleaved<
                   x86::matmul::gemm_avx2_s8s8s32_2x4x16>(
                   m, n, k, trans_a, trans_b, strategy, cacheline)
            .get_workspace_size();
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(AlgoInt8x8x32AVX2M2N4K16,
                                     megdnn_x86_matmul_kern,
                                     "AlgoInt8x8x32AVX2M2N4K16"_hash,
                                     x86::matmul::gemm_avx2_s8s8s32_2x4x16,
                                     dt_int8, dt_int32,
                                     AlgoDataType::QINT8X8X32, DEFAULT);

/*************************AlgoInt8x8x32SSEM4N8K2********************/
MatrixMulImpl::kern_t MatrixMulImpl::AlgoInt8x8x32SSEM4N8K2::get_kern(
        const KernSizeParam&) const {
    return gemm_s8s8s32_sse_4x8x2;
}
bool MatrixMulImpl::AlgoInt8x8x32SSEM4N8K2::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv() &&
           ((kern_size_param.A_type.enumv() == DTypeEnum::Int8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::Int32) ||
            (kern_size_param.A_type.enumv() == DTypeEnum::QuantizedS8 &&
             kern_size_param.C_type.enumv() == DTypeEnum::QuantizedS32)) &&
           kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == Param::Format::DEFAULT &&
           is_supported(SIMDType::SSE4_1);
}
size_t MatrixMulImpl::AlgoInt8x8x32SSEM4N8K2::get_workspace(
        const KernSizeParam& kern_param) const {
    constexpr int cacheline = 64;
    const size_t m = kern_param.M;
    const size_t n = kern_param.N;
    const size_t k = kern_param.K;
    const bool trans_a = kern_param.trA;
    const bool trans_b = kern_param.trB;
    auto a_type = kern_param.A_type;
    auto b_type = kern_param.B_type;
    auto c_type = kern_param.C_type;
    x86::matmul::gemm_sse_s8s8s32_4x8x2 strategy(m, n, k, a_type, b_type,
                                                 c_type);

    return megdnn::matmul::GemmInterleaved<x86::matmul::gemm_sse_s8s8s32_4x8x2>(
                   m, n, k, trans_a, trans_b, strategy, cacheline)
            .get_workspace_size();
}
MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL_DETAIL(AlgoInt8x8x32SSEM4N8K2,
                                            megdnn_x86_matmul_kern,
                                            "AlgoInt8x8x32SSEM4N8K2"_hash,
                                            x86::matmul::gemm_sse_s8s8s32_4x8x2,
                                            dt_int8, dt_int32, dt_int16,
                                            AlgoDataType::QINT8X8X32, DEFAULT);

/*************************AlgoF32MK8_8x8********************/
MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32MK8_8x8::get_kern(
        const KernSizeParam&) const {
    auto f32_kern_mk8_8x8 = [](const MatrixMulImpl::KernParam& kern_param) {
        MIDOUT_BEGIN(megdnn_x86_matmul_kern_mk8_8x8, midout_iv(0)) {
            auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
            auto trA = kern_param.trA, trB = kern_param.trB;
            auto LDA = kern_param.LDA, LDB = kern_param.LDB,
                 LDC = kern_param.LDC;
            auto A_type = kern_param.A_type, B_type = kern_param.B_type,
                 C_type = kern_param.C_type;
            const auto Aptr = kern_param.A<float>(),
                       Bptr = kern_param.B<float>();
            auto Cptr = kern_param.C<float>();

            x86::matmul::sgemm_nopack_8x8_avx2 strategy(A_type, B_type, C_type);
            megdnn::matmul::GemmInterleaved<x86::matmul::sgemm_nopack_8x8_avx2,
                                            false>(M, N, K, trA, trB, strategy)
                    .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC,
                             kern_param.workspace_ptr);
        }
        MIDOUT_END();
    };
    return f32_kern_mk8_8x8;
}

bool MatrixMulImpl::AlgoF32MK8_8x8::usable(
        const KernSizeParam& kern_size_param) const {
    constexpr static size_t MB = 8;
    constexpr static size_t KB = 8;
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.B_type.enumv() == kern_size_param.A_type.enumv() &&
           kern_size_param.C_type.enumv() == kern_size_param.A_type.enumv() &&
           kern_size_param.A_type.enumv() == DTypeEnum::Float32 &&
           kern_size_param.format == param::MatrixMul::Format::MK8 &&
           !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.M % MB == 0 && kern_size_param.K % KB == 0 &&
           is_supported(SIMDType::FMA);
}

size_t MatrixMulImpl::AlgoF32MK8_8x8::get_workspace(
        const KernSizeParam& kern_param) const {
    MIDOUT_BEGIN(megdnn_x86_matmul_kern_mk8_8x8, midout_iv(0)) {
        const size_t m = kern_param.M;
        const size_t n = kern_param.N;
        const size_t k = kern_param.K;
        const bool trans_a = kern_param.trA;
        const bool trans_b = kern_param.trB;
        auto a_type = kern_param.A_type;
        auto b_type = kern_param.B_type;
        auto c_type = kern_param.C_type;
        x86::matmul::sgemm_nopack_8x8_avx2 strategy(a_type, b_type, c_type);
        return megdnn::matmul::GemmInterleaved<
                       x86::matmul::sgemm_nopack_8x8_avx2, false>(
                       m, n, k, trans_a, trans_b, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
}

// vim: syntax=cpp.doxygen
