#include "src/fallback/matrix_mul/algos.h"
#include "megdnn/opr_param_defs.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/fallback/matrix_mul/gemv.h"
#include "src/fallback/matrix_mul/generic_strategy.h"

#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "src/fallback/matrix_mul/gi/fp32/exec_sgemv.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fb_matmul_f32_kern)
MIDOUT_DECL(megdnn_fb_matmul_f32_gemm_gemv_like)
MIDOUT_DECL(megdnn_fb_matmul_naive)
MIDOUT_DECL(megdnn_fb_gi_exec_fp32)
MIDOUT_DECL(megdnn_fb_gi_matmul_kern)
MIDOUT_DECL(megdnn_fb_gi_f32_4x12)

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
                "M and K must time of pack_size  M: %zu K: %zu pack_size: %zu", M, N,
                pack_size);

#define DISPATCH(TA, TB)                                                               \
    if (kern_param.trA == TA && kern_param.trB == TB) {                                \
        naive::dispatch_ta_tb<TA, TB>(                                                 \
                kern_param.A_ptr.get_ptr(), kern_param.B_ptr.get_ptr(),                \
                kern_param.C_ptr.get_ptr(), kern_param.workspace_ptr, M / pack_size,   \
                N, K / pack_size, LDA, LDB, LDC, kern_param.A_type, kern_param.B_type, \
                kern_param.C_type, kern_param.format, kern_param.compute_mode);        \
        return;                                                                        \
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

/* ================== F32 Gemv MK4 gi algo ================== */
namespace {
void gi_f32_gemv_mk4_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_fb_gi_exec_fp32, midout_iv("f32_gemv_mk4_gi_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        const auto Aptr = kern_param.A<dt_float32>(), Bptr = kern_param.B<dt_float32>();
        auto Cptr = kern_param.C<dt_float32>();
        gi_gemv_like_mk4(Aptr, Bptr, Cptr, M, N, K, LDA, LDB, LDC);
    }
    MIDOUT_END();
}
}  // anonymous namespace

bool MatrixMulImpl::AlgoF32GiGemvMK4::usable(
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
           !kern_size_param.trB && M % 4 == 0 && K % 4 == 0 && N == 1 && LDB == 4;
}

bool MatrixMulImpl::AlgoF32GiGemvMK4::preferred(
        const KernSizeParam& kern_size_param) const {
    MEGDNN_MARK_USED_VAR(kern_size_param);
    return true;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32GiGemvMK4::get_kern(
        const KernSizeParam&) const {
    return gi_f32_gemv_mk4_kern;
}

/* ================== F32 Gemm MK4 gi algo ================== */
namespace {
void gi_f32_mk4_4x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_fb_gi_matmul_kern, midout_iv("gi_f32_mk4_4x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<float>(), Bptr = kern_param.B<float>();
        auto Cptr = kern_param.C<float>();

        matmul::fallback::gi_sgemm_nopack_4x8 strategy(A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<matmul::fallback::gi_sgemm_nopack_4x8, false>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC, kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace
bool MatrixMulImpl::AlgoF32GiMK4_4x8::usable(
        const KernSizeParam& kern_size_param) const {
    constexpr size_t MB = 4;
    constexpr size_t KB = 4;
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() && !kern_size_param.trA &&
           !kern_size_param.trB && kern_size_param.M % MB == 0 &&
           kern_size_param.K % KB == 0;
}

size_t MatrixMulImpl::AlgoF32GiMK4_4x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_fb_gi_matmul_kern,
            midout_iv("AlgoF32GiMK4_4x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        matmul::fallback::gi_sgemm_nopack_4x8 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       matmul::fallback::gi_sgemm_nopack_4x8, false>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32GiMK4_4x8::get_kern(
        const KernSizeParam&) const {
    return gi_f32_mk4_4x8_kern;
}
#if defined(GI_SUPPORT_F16)
/* ================== F16 Gemm MK8 gi algo ================== */
namespace {
void gi_f16_mk8_8x8_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_fb_gi_matmul_kern, midout_iv("gi_f16_mk8_8x8_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<dt_float16>(), Bptr = kern_param.B<dt_float16>();
        auto Cptr = kern_param.C<dt_float16>();

        matmul::fallback::gi_sgemm_nopack_mk8_8x8_fp16 strategy(A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<
                matmul::fallback::gi_sgemm_nopack_mk8_8x8_fp16, false>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC, kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace
bool MatrixMulImpl::AlgoF16GiMK8_8x8::usable(
        const KernSizeParam& kern_size_param) const {
    constexpr size_t MB = 8;
    constexpr size_t KB = 8;
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK8 &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float16() && !kern_size_param.trA &&
           !kern_size_param.trB && kern_size_param.M % MB == 0 &&
           kern_size_param.K % KB == 0;
}

size_t MatrixMulImpl::AlgoF16GiMK8_8x8::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_fb_gi_matmul_kern,
            midout_iv("AlgoF16GiMK8_8x8::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        matmul::fallback::gi_sgemm_nopack_mk8_8x8_fp16 strategy(A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       matmul::fallback::gi_sgemm_nopack_mk8_8x8_fp16, false>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF16GiMK8_8x8::get_kern(
        const KernSizeParam&) const {
    return gi_f16_mk8_8x8_kern;
}
#endif
/* ===================== F32 algo gi mk4 pack K4x12 ===================== */
namespace {
void f32_gi_mk4_pack_4x12_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(
            megdnn_fb_gi_matmul_kern, midout_iv("f32_gi_mk4_pack_4x12_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<float>(), Bptr = kern_param.B<float>();
        auto Cptr = kern_param.C<float>();

        matmul::fallback::gi_sgemm_mk4_pack_4x12 strategy(
                M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<matmul::fallback::gi_sgemm_mk4_pack_4x12>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC, kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace

bool MatrixMulImpl::AlgoF32GiMK4Pack4x12::usable(
        const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::MK4 &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32() && !kern_size_param.trA &&
           !kern_size_param.trB && kern_size_param.M % 4 == 0 &&
           kern_size_param.K % 4 == 0 && !kern_size_param.trA && !kern_size_param.trB;
}

size_t MatrixMulImpl::AlgoF32GiMK4Pack4x12::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_fb_gi_matmul_kern,
            midout_iv("AlgoF32GiMK4Pack4x12::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        matmul::fallback::gi_sgemm_mk4_pack_4x12 strategy(
                M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<
                       matmul::fallback::gi_sgemm_mk4_pack_4x12>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32GiMK4Pack4x12::get_kern(
        const KernSizeParam&) const {
    return f32_gi_mk4_pack_4x12_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(
        AlgoF32GiMK4Pack4x12, megdnn_fb_gi_matmul_kern, "AlgoF32GiMK4Pack4x12"_hash,
        matmul::fallback::gi_sgemm_mk4_pack_4x12, float, float, AlgoDataType::FLOAT32,
        MK4);

/* ===================== F32 algo ===================== */
namespace {
void f32_kern(const MatrixMulImpl::KernParam& kern_param) {
    MIDOUT_BEGIN(megdnn_fb_gi_f32_4x12, midout_iv("f32_kern"_hash)) {
        auto M = kern_param.M, N = kern_param.N, K = kern_param.K;
        auto trA = kern_param.trA, trB = kern_param.trB;
        auto LDA = kern_param.LDA, LDB = kern_param.LDB, LDC = kern_param.LDC;
        auto A_type = kern_param.A_type, B_type = kern_param.B_type,
             C_type = kern_param.C_type;
        const auto Aptr = kern_param.A<float>(), Bptr = kern_param.B<float>();
        auto Cptr = kern_param.C<float>();

        matmul::fallback::gi_sgemm_4x12 strategy(M, N, K, A_type, B_type, C_type);
        megdnn::matmul::GemmInterleaved<matmul::fallback::gi_sgemm_4x12>(
                M, N, K, trA, trB, strategy)
                .execute(Aptr, LDA, Bptr, LDB, Cptr, LDC, kern_param.workspace_ptr);
    }
    MIDOUT_END();
}

}  // anonymous namespace

bool MatrixMulImpl::AlgoF32Gi4x12::usable(const KernSizeParam& kern_size_param) const {
    return kern_size_param.compute_mode == Param::ComputeMode::DEFAULT &&
           kern_size_param.format == param::MatrixMul::Format::DEFAULT &&
           kern_size_param.B_type == kern_size_param.A_type &&
           kern_size_param.C_type == kern_size_param.A_type &&
           kern_size_param.A_type == dtype::Float32();
}

size_t MatrixMulImpl::AlgoF32Gi4x12::get_workspace(
        const KernSizeParam& kern_size_param) const {
    MIDOUT_BEGIN(
            megdnn_fb_gi_f32_4x12, midout_iv("AlgoF32Gi4x12::get_workspace"_hash)) {
        auto M = kern_size_param.M, N = kern_size_param.N, K = kern_size_param.K;
        auto trA = kern_size_param.trA, trB = kern_size_param.trB;
        auto A_type = kern_size_param.A_type, B_type = kern_size_param.B_type,
             C_type = kern_size_param.C_type;
        matmul::fallback::gi_sgemm_4x12 strategy(M, N, K, A_type, B_type, C_type);
        return megdnn::matmul::GemmInterleaved<matmul::fallback::gi_sgemm_4x12>(
                       M, N, K, trA, trB, strategy)
                .get_workspace_size();
    }
    MIDOUT_END();
    return 0;
}

MatrixMulImpl::kern_t MatrixMulImpl::AlgoF32Gi4x12::get_kern(
        const KernSizeParam&) const {
    return f32_kern;
}

MEGDNN_REG_GEMM_FUNC_FOR_IM2COL_IMPL(
        AlgoF32Gi4x12, megdnn_fb_gi_f32_4x12, "AlgoF32Gi4x12Impl"_hash,
        matmul::fallback::gi_sgemm_4x12, float, float, AlgoDataType::FLOAT32, DEFAULT);

// vim: syntax=cpp.doxygen
