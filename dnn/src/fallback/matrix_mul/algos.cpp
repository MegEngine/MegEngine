/**
 * \file dnn/src/fallback/matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/matrix_mul/algos.h"
#include "megdnn/opr_param_defs.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/fallback/matrix_mul/gemv.h"
#include "src/fallback/matrix_mul/generic_strategy.h"

#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fb_matmul_f32_kern)
MIDOUT_DECL(megdnn_fb_matmul_f32_gemm_gemv_like)
MIDOUT_DECL(megdnn_fb_matmul_naive)

using namespace megdnn;
using namespace fallback;

/* ===================== F32 8x12x1 algo ===================== */

namespace {
void f32_8x12x1_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_fb_matmul_f32_kern, void) {
        size_t M = kern_param.M, N = kern_param.N, K = kern_param.K;
        matmul::fallback::sgemm_8x12 strategy(
                M, N, K, kern_param.A_type, kern_param.B_type, kern_param.C_type);
        matmul::GemmInterleaved<matmul::fallback::sgemm_8x12>(
                M, N, K, kern_param.trA, kern_param.trB, strategy)
                .execute(
                        kern_param.A<float>(), kern_param.LDA, kern_param.B<float>(),
                        kern_param.LDB, kern_param.C<float>(), kern_param.LDC,
                        kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

void kern_naive(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_fb_matmul_naive, void) {
        size_t M = kern_param.M, N = kern_param.N, K = kern_param.K;
        size_t LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto get_pack_size = [kern_param]() -> size_t {
            switch (kern_param.format) {
                case param::MatrixMul::Format::MK4:
                case param::MatrixMul::Format::MK4_DOT:
                    return 4_z;
                case param::MatrixMul::Format::MK8:
                    return 8_z;
                default:
                    return 1_z;
            }
        };

        size_t pack_size = get_pack_size();
        megdnn_assert(
                (M % pack_size == 0 && K % pack_size == 0),
                "M and N must time of pack_size  M: %zu N: %zu pack_size: %zu", M, N,
                pack_size);

#define DISPATCH(TA, TB)                                                             \
    if (kern_param.trA == TA && kern_param.trB == TB) {                              \
        naive::dispatch_ta_tb<TA, TB>(                                               \
                kern_param.A_ptr, kern_param.B_ptr, kern_param.C_ptr,                \
                kern_param.workspace_ptr, M / pack_size, N, K / pack_size, LDA, LDB, \
                LDC, kern_param.A_type, kern_param.B_type, kern_param.C_type,        \
                kern_param.format, kern_param.compute_mode);                         \
        return;                                                                      \
    }
        DISPATCH(true, true);
        DISPATCH(true, false);
        DISPATCH(false, true);
        DISPATCH(false, false);
#undef DISPATCH
        megdnn_assert_internal(0);
    }
    MIDOUT_END();
}
}  // anonymous namespace

////////////////////// AlgoF32K8x12x1 ///////////////////////////

bool MatrixMulImpl::AlgoF32K8x12x1::usable(const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == param::MatrixMul::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32{};
}

size_t MatrixMulImpl::AlgoF32K8x12x1::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_fb_matmul_f32_kern,
            midout_iv("AlgoF32K8x12x1::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K;
        matmul::fallback::sgemm_8x12 strategy(
                M, N, K, kern_size_param.A_type, kern_size_param.B_type,
                kern_size_param.C_type);
        return matmul::GemmInterleaved<matmul::fallback::sgemm_8x12>(
                       M, N, K, kern_size_param.trA, kern_size_param.trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32K8x12x1::get_kern(
        const KernSizeParam&) const {
    return f32_8x12x1_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(
        AlgoF32K8x12x1, megdnn_fb_matmul_f32_kern, 5, matmul::fallback::sgemm_8x12,
        float, float, AlgoDataType::FLOAT32, DEFAULT);

/* ===================== gemv algo ===================== */
bool MatrixMulImpl::AlgoGemv::usable(const KernSizeParam& kern_size_param) const {
    return !kern_size_param.trA && !kern_size_param.trB &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.compute_mode == param::MatrixMul::ComputeMode::DEFAULT &&
           !((kern_size_param.A_type.enumv() == kern_size_param.B_type.enumv()) &&
             (kern_size_param.A_type.enumv() == DTypeEnum::Int16) &&
             (kern_size_param.C_type.enumv() == DTypeEnum::Int32));
}

bool MatrixMulImpl::AlgoGemv::preferred(const KernSizeParam& kern_size_param) const {
    return kern_size_param.M <= 2 &&
           kern_size_param.A_type.category() != DTypeCategory::FLOAT;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoGemv::get_kern(
        const KernSizeParam& kern_size_param) const {
#define DISPATCH(A, C, func, _midout_iv)                                           \
    if (kern_size_param.A_type.enumv() == DTypeEnum::A &&                          \
        kern_size_param.B_type.enumv() == DTypeEnum::A &&                          \
        kern_size_param.C_type.enumv() == DTypeEnum::C &&                          \
        kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&             \
        kern_size_param.format == param::MatrixMul::Format::DEFAULT) {             \
        MIDOUT_BEGIN(megdnn_fb_matmul_f32_gemm_gemv_like, midout_iv(_midout_iv)) { \
            return func;                                                           \
        }                                                                          \
        MIDOUT_END();                                                              \
    }

    DISPATCH(Float32, Float32, (gemm_gemv_like<dt_float32, dt_float32>), 0);
    DNN_INC_FLOAT16(
            DISPATCH(Float16, Float16, (gemm_gemv_like<dt_float16, dt_float16>), 1));
    DISPATCH(Int8, Int16, (gemm_gemv_like<dt_int8, dt_int16>), 2);
    DISPATCH(
            Quantized8Asymm, QuantizedS32, (gemm_gemv_like<dt_uint8, dt_int32, true>),
            3);
    if (can_be_treated_as_int8x8x32(kern_size_param)) {
        MIDOUT_BEGIN(megdnn_fb_matmul_f32_gemm_gemv_like, midout_iv(4)) {
            return gemm_gemv_like<dt_int8, dt_int32>;
        }
        MIDOUT_END();
    }
#undef DISPATCH
    megdnn_assert(0);
}

/* ===================== naive algo ===================== */
bool MatrixMulImpl::AlgoNaive::usable(const KernSizeParam&) const {
    return true;
}

bool MatrixMulImpl::AlgoNaive::preferred(const KernSizeParam&) const {
    return false;
}

size_t MatrixMulImpl::AlgoNaive::get_workspace(const KernSizeParam& kern_param) const {
    MIDOUT_BEGIN(
            megdnn_fb_matmul_naive,
            midout_iv("MatrixMulForwardImpl::get_workspace_in_bytes"_hash)) {
        if (kern_param.A_type.enumv() == DTypeEnum::Quantized4Asymm ||
            kern_param.A_type.enumv() == DTypeEnum::QuantizedS4) {
            size_t ret = 0;
            if (kern_param.trA) {
                ret += kern_param.LDA * kern_param.K;
            } else {
                ret += kern_param.LDA * kern_param.M;
            }
            if (kern_param.trB) {
                ret += kern_param.LDB * kern_param.N;
            } else {
                ret += kern_param.LDB * kern_param.K;
            }
            return ret;
        }
        return 0;
    }
    MIDOUT_END();
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoNaive::get_kern(const KernSizeParam&) const {
    return kern_naive;
}

// vim: syntax=cpp.doxygen
