#include "src/arm_common/simd_macro/marm_neon.h"

#include "megdnn/oprs.h"
#include "src/arm_common/matrix_mul/int8/gemv.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_int8_gemv)

using namespace megdnn;
using namespace arm_common;

namespace {

void gemv_naive_n(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 1);
    size_t m = 0;
    for (; m + 2 <= M; m += 2) {
        int32_t acc0 = 0, acc1 = 0;
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(A + m * Astride + k);
            int8x16_t a1 = vld1q_s8(A + (m + 1) * Astride + k);

            int8x16_t b0 = vld1q_s8(B + k);

            int16x8_t c0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
            c0 = vmlal_high_s8(c0, a0, b0);

            int16x8_t c1 = vmull_s8(vget_low_s8(a1), vget_low_s8(b0));
            c1 = vmlal_high_s8(c1, a1, b0);
            acc0 += vaddlvq_s16(c0);
            acc1 += vaddlvq_s16(c1);
        }

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t a1 = vld1_s8(A + (m + 1) * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);

            int16x8_t c0 = vmull_s8(a0, b0);

            int16x8_t c1 = vmull_s8(a1, b0);
            acc0 += vaddlvq_s16(c0);
            acc1 += vaddlvq_s16(c1);
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(A[m * Astride + k]) * B[k];
            acc1 += static_cast<int32_t>(A[(m + 1) * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc0;
        C[(m + 1) * Cstride] = acc1;
    }

    for (; m < M; ++m) {
        int32_t acc0 = 0;
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(A + m * Astride + k);
            int8x16_t b0 = vld1q_s8(B + k);

            int16x8_t c0 = vmull_s8(vget_low_s8(a0), vget_low_s8(b0));
            c0 = vmlal_high_s8(c0, a0, b0);

            acc0 += vaddlvq_s16(c0);
        }

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);

            int16x8_t c0 = vmull_s8(a0, b0);
            acc0 += vaddlvq_s16(c0);
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(A[m * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc0;
    }
}

void gemv_naive_n_mk4(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == 4);
    auto Aptr = A;
    size_t m = 0;
    for (; m < M; m += PACK_SIZE) {
        auto Bptr = B;
        auto Aptr0 = Aptr;
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16x4_t a = vld4q_s8(Aptr0);
            int8x16_t b = vld1q_s8(Bptr);
            int16x8_t c[4];

            c[0] = vmull_s8(vget_low_s8(a.val[0]), vget_low_s8(b));
            c[1] = vmull_s8(vget_low_s8(a.val[1]), vget_low_s8(b));
            c[2] = vmull_s8(vget_low_s8(a.val[2]), vget_low_s8(b));
            c[3] = vmull_s8(vget_low_s8(a.val[3]), vget_low_s8(b));

            c[0] = vmlal_high_s8(c[0], a.val[0], b);
            c[1] = vmlal_high_s8(c[1], a.val[1], b);
            c[2] = vmlal_high_s8(c[2], a.val[2], b);
            c[3] = vmlal_high_s8(c[3], a.val[3], b);

            acc0 += vaddlvq_s16(c[0]);
            acc1 += vaddlvq_s16(c[1]);
            acc2 += vaddlvq_s16(c[2]);
            acc3 += vaddlvq_s16(c[3]);

            Bptr += 16;
            Aptr0 += PACK_SIZE * 16;
        }

        for (; k + 8 <= K; k += 8) {
            int8x8x4_t a = vld4_s8(Aptr0);
            int8x8_t b = vld1_s8(Bptr);
            int16x8_t c[4];

            c[0] = vmull_s8(a.val[0], b);
            c[1] = vmull_s8(a.val[1], b);
            c[2] = vmull_s8(a.val[2], b);
            c[3] = vmull_s8(a.val[3], b);

            acc0 += vaddlvq_s16(c[0]);
            acc1 += vaddlvq_s16(c[1]);
            acc2 += vaddlvq_s16(c[2]);
            acc3 += vaddlvq_s16(c[3]);

            Bptr += 8;
            Aptr0 += PACK_SIZE * 8;
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(*(Aptr0 + 0)) * B[k];
            acc1 += static_cast<int32_t>(*(Aptr0 + 1)) * B[k];
            acc2 += static_cast<int32_t>(*(Aptr0 + 2)) * B[k];
            acc3 += static_cast<int32_t>(*(Aptr0 + 3)) * B[k];
            Aptr0 += 4;
        }

        C[0] = acc0;
        C[1] = acc1;
        C[2] = acc2;
        C[3] = acc3;

        Aptr += Astride;
        C += Cstride;
    }
}

}  // namespace

#if MGB_ENABLE_DOT
namespace {
MEGDNN_ATTRIBUTE_TARGET("dotprod")
void gemv_naive_n_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1 && Bstride == 1);
    size_t m = 0;
    for (; m + 2 <= M; m += 2) {
        int32_t acc[4];
        int32x4_t acc_neon = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int64x2_t a0 = vreinterpretq_s64_s8(vld1q_s8(A + m * Astride + k));
            int64x2_t a1 = vreinterpretq_s64_s8(vld1q_s8(A + (m + 1) * Astride + k));
            //! the first 8 elements is m, the last 8 elements is m + 1
            int8x16_t a2 = vreinterpretq_s8_s64(vzip1q_s64(a0, a1));
            int8x16_t a3 = vreinterpretq_s8_s64(vzip2q_s64(a0, a1));

            int64x2_t b0 = vreinterpretq_s64_s8(vld1q_s8(B + k));
            int8x16_t b2 = vreinterpretq_s8_s64(vzip1q_s64(b0, b0));
            int8x16_t b3 = vreinterpretq_s8_s64(vzip2q_s64(b0, b0));

            acc_neon = vdotq_s32(acc_neon, a2, b2);
            acc_neon = vdotq_s32(acc_neon, a3, b3);
        }
        vst1q_s32(acc, acc_neon);

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t a1 = vld1_s8(A + (m + 1) * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);
            uint32x2_t zero = vdup_n_s32(0);
            acc[0] += vaddv_s32(vdot_s32(zero, a0, b0));
            zero = vdup_n_s32(0);
            acc[3] += vaddv_s32(vdot_s32(zero, a1, b0));
        }

        for (; k < K; ++k) {
            acc[0] += static_cast<int32_t>(A[m * Astride + k]) * B[k];
            acc[3] += static_cast<int32_t>(A[(m + 1) * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc[0] + acc[1];
        C[(m + 1) * Cstride] = acc[2] + acc[3];
    }

    for (; m < M; ++m) {
        int32_t acc[4];
        int32x4_t acc_neon = vdupq_n_s32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(A + m * Astride + k);
            int8x16_t b0 = vld1q_s8(B + k);
            acc_neon = vdotq_s32(acc_neon, a0, b0);
        }
        vst1q_s32(acc, acc_neon);

        for (; k + 8 <= K; k += 8) {
            int8x8_t a0 = vld1_s8(A + m * Astride + k);
            int8x8_t b0 = vld1_s8(B + k);
            uint32x2_t zero = vdup_n_s32(0);
            acc[0] += vaddv_s32(vdot_s32(zero, a0, b0));
        }

        for (; k < K; ++k) {
            acc[0] += static_cast<int32_t>(A[m * Astride + k]) * B[k];
        }
        C[m * Cstride] = acc[0] + acc[1] + acc[2] + acc[3];
    }
}

MEGDNN_ATTRIBUTE_TARGET("dotprod")
void gemv_naive_n_mk4_dotprod(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == 4);

    auto Aptr = A;
    size_t m = 0;
    for (; m < M; m += PACK_SIZE) {
        auto Bptr = B;
        auto Aptr0 = Aptr;
        int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        size_t k = 0;
        if (k + 16 <= K) {
            int32x4_t acc_neon[4];
            acc_neon[0] = vdupq_n_s32(0);
            acc_neon[1] = vdupq_n_s32(0);
            acc_neon[2] = vdupq_n_s32(0);
            acc_neon[3] = vdupq_n_s32(0);
            for (; k + 16 <= K; k += 16) {
                int8x16x4_t a = vld4q_s8(Aptr0);
                int8x16_t b = vld1q_s8(Bptr);

                acc_neon[0] = vdotq_s32(acc_neon[0], a.val[0], b);
                acc_neon[1] = vdotq_s32(acc_neon[1], a.val[1], b);
                acc_neon[2] = vdotq_s32(acc_neon[2], a.val[2], b);
                acc_neon[3] = vdotq_s32(acc_neon[3], a.val[3], b);

                Bptr += 16;
                Aptr0 += PACK_SIZE * 16;
            }
            acc0 = vaddvq_s32(acc_neon[0]);
            acc1 = vaddvq_s32(acc_neon[1]);
            acc2 = vaddvq_s32(acc_neon[2]);
            acc3 = vaddvq_s32(acc_neon[3]);
        }

        if (k + 8 <= K) {
            int32x2_t acc_neon[4];
            acc_neon[0] = vdup_n_s32(0);
            acc_neon[1] = vdup_n_s32(0);
            acc_neon[2] = vdup_n_s32(0);
            acc_neon[3] = vdup_n_s32(0);

            int8x8x4_t a = vld4_s8(Aptr0);
            int8x8_t b = vld1_s8(Bptr);
            acc_neon[0] = vdot_s32(acc_neon[0], a.val[0], b);
            acc_neon[1] = vdot_s32(acc_neon[1], a.val[1], b);
            acc_neon[2] = vdot_s32(acc_neon[2], a.val[2], b);
            acc_neon[3] = vdot_s32(acc_neon[3], a.val[3], b);

            Bptr += 8;
            Aptr0 += PACK_SIZE * 8;
            k += 8;

            acc0 += vaddv_s32(acc_neon[0]);
            acc1 += vaddv_s32(acc_neon[1]);
            acc2 += vaddv_s32(acc_neon[2]);
            acc3 += vaddv_s32(acc_neon[3]);
        }

        for (; k < K; ++k) {
            acc0 += static_cast<int32_t>(*(Aptr0 + 0)) * B[k];
            acc1 += static_cast<int32_t>(*(Aptr0 + 1)) * B[k];
            acc2 += static_cast<int32_t>(*(Aptr0 + 2)) * B[k];
            acc3 += static_cast<int32_t>(*(Aptr0 + 3)) * B[k];
            Aptr0 += 4;
        }

        C[0] = acc0;
        C[1] = acc1;
        C[2] = acc2;
        C[3] = acc3;

        Aptr += Astride;
        C += Cstride;
    }
}

MEGDNN_ATTRIBUTE_TARGET("dotprod")
void gemv_naive_n_mk4_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    constexpr size_t PACK_SIZE = 4;
    megdnn_assert(N == 1 && Bstride == 4);

    auto Aptr = A;
    size_t m = 0;
    for (; m < M; m += PACK_SIZE) {
        auto Bptr = B;
        auto Aptr0 = Aptr;
        size_t k = 0;
        int32x4_t acc_neon;
        acc_neon = vdupq_n_s32(0);
        for (; k + 16 <= K; k += 16) {
            int8x16_t a0 = vld1q_s8(Aptr0);
            int8x16_t a1 = vld1q_s8(Aptr0 + 16);
            int8x16_t a2 = vld1q_s8(Aptr0 + 32);
            int8x16_t a3 = vld1q_s8(Aptr0 + 48);
            int8x16_t b = vld1q_s8(Bptr);
            acc_neon = vdotq_laneq_s32(acc_neon, a0, b, 0);
            acc_neon = vdotq_laneq_s32(acc_neon, a1, b, 1);
            acc_neon = vdotq_laneq_s32(acc_neon, a2, b, 2);
            acc_neon = vdotq_laneq_s32(acc_neon, a3, b, 3);
            Bptr += 16;
            Aptr0 += PACK_SIZE * 16;
        }

        if (k + 8 <= K) {
            int8x16_t a0 = vld1q_s8(Aptr0);
            int8x16_t a1 = vld1q_s8(Aptr0 + 16);
            int8x8_t b = vld1_s8(Bptr);
            acc_neon = vdotq_lane_s32(acc_neon, a0, b, 0);
            acc_neon = vdotq_lane_s32(acc_neon, a1, b, 1);
            Bptr += 8;
            Aptr0 += PACK_SIZE * 8;
            k += 8;
        }

        if (k + 4 <= K) {
            int8x16_t a = vld1q_s8(Aptr0);
            int32_t tmp = *(reinterpret_cast<const int32_t*>(Bptr));
            int8x8_t b = vdup_n_s32(tmp);
            acc_neon = vdotq_lane_s32(acc_neon, a, b, 0);
        }

        vst1q_s32(C, acc_neon);
        Aptr += Astride;
        C += Cstride;
    }
}

}  // namespace
#endif

bool arm_common::is_gemv_like_preferred_int8(
        bool transposeA, bool transposeB, size_t M, size_t N, size_t K, size_t LDA,
        size_t LDB, size_t LDC) {
    MEGDNN_MARK_USED_VAR(LDA);
    MEGDNN_MARK_USED_VAR(LDB);
    MEGDNN_MARK_USED_VAR(LDC);
    MEGDNN_MARK_USED_VAR(M);
    MEGDNN_MARK_USED_VAR(K);
    if (transposeA)
        return false;
    if (transposeB)
        return false;

    return N == 1 && LDB == 1;
}

void arm_common::gemv_like(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv, midout_iv("INT8_gemv_like"_hash)) {
#if MGB_ENABLE_DOT
        if (cpuinfo_has_arm_neon_dot()) {
            return gemv_naive_n_dot(A, B, C, M, N, K, Astride, Bstride, Cstride);
        } else {
            return gemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride);
        }
#else
        return gemv_naive_n(A, B, C, M, N, K, Astride, Bstride, Cstride);
#endif
    }
    MIDOUT_END();
}

void arm_common::gemv_like_mk4(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv, midout_iv("INT8_gemv_like_mk4"_hash)) {
#if MGB_ENABLE_DOT
        if (cpuinfo_has_arm_neon_dot()) {
            return gemv_naive_n_mk4_dotprod(
                    A, B, C, M, N, K, Astride, Bstride, Cstride);
        } else {
            return gemv_naive_n_mk4(A, B, C, M, N, K, Astride, Bstride, Cstride);
        }
#else
        return gemv_naive_n_mk4(A, B, C, M, N, K, Astride, Bstride, Cstride);
#endif
    }
    MIDOUT_END();
}

#if MGB_ENABLE_DOT
void arm_common::gemv_like_mk4_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(N == 1);
    MIDOUT_BEGIN(
            megdnn_arm_common_int8_gemv, midout_iv("INT8_gemv_like_mk4_dot"_hash)) {
        return gemv_naive_n_mk4_dot(A, B, C, M, N, K, Astride, Bstride, Cstride);
    }
    MIDOUT_END();
}
#endif
#if MGB_ENABLE_DOT
namespace {
MEGDNN_ATTRIBUTE_TARGET("dotprod")
void gevm_naive_dot_impl(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride,
        bool load_c) {
    constexpr size_t n_block = 32;
    const size_t n_end = N / n_block * n_block;
    const size_t n_remain = N - n_end;

    constexpr size_t k_block = 4;
    constexpr size_t k_block_x2 = k_block * 2;
    const size_t k_end = (K / k_block_x2) * k_block_x2;
    const size_t k_remain = K - k_end;
    for (size_t n = 0; n < n_end; n += n_block) {
        if (K < k_block_x2) {
            if (!load_c) {
                for (size_t i = 0; i < n_block; ++i) {
                    C[n + i] = 0;
                }
            }
            for (size_t k = 0; k < K; ++k) {
                for (size_t i = 0; i < n_block; ++i) {
                    C[n + i] += A[k] * B[k * Bstride + n + i];
                }
            }
            continue;
        }
        int32x4_t c[8];
        if (load_c) {
#define cb(step) c[step] = vld1q_s32(C + n + step * 4);
            UNROLL_CALL_RAW(8, cb);
#undef cb
        } else {
#define cb(step) c[step] = vdupq_n_s32(0);
            UNROLL_CALL_RAW(8, cb);
#undef cb
        }
        int8x16_t a[2];
        a[0] = vld1q_dup_s32(A);
        int8x16_t b[2][8];
#define cb(step)                                                 \
    b[0][step * 2 + 0] = vld1q_s8(B + (0 + step) * Bstride + n); \
    b[0][step * 2 + 1] = vld1q_s8(B + (0 + step) * Bstride + n + 16);
        UNROLL_CALL_RAW(4, cb);
#undef cb
        size_t k_buffer_end = k_end - k_block_x2;
        for (size_t k = 0; k < k_buffer_end; k += k_block_x2) {
            //! double buffer main
#define cb(step)                                                           \
    b[1][step * 2 + 0] = vld1q_s8(B + (k + step + k_block) * Bstride + n); \
    b[1][step * 2 + 1] = vld1q_s8(B + (k + step + k_block) * Bstride + n + 16);
            UNROLL_CALL_RAW(4, cb);
#undef cb
            a[1] = vld1q_dup_s32(A + k + k_block);

            int8x16x2_t ab0 = vzipq_s8(b[0][0], b[0][2]);
            int8x16x2_t cd0 = vzipq_s8(b[0][4], b[0][6]);
            int8x16x2_t ab1 = vzipq_s8(b[0][1], b[0][3]);
            int8x16x2_t cd1 = vzipq_s8(b[0][5], b[0][7]);
            int16x8x2_t abcd0 = vzipq_s16(ab0.val[0], cd0.val[0]);
            int16x8x2_t abcd1 = vzipq_s16(ab0.val[1], cd0.val[1]);
            int16x8x2_t abcd2 = vzipq_s16(ab1.val[0], cd1.val[0]);
            int16x8x2_t abcd3 = vzipq_s16(ab1.val[1], cd1.val[1]);
            c[0] = vdotq_s32(c[0], abcd0.val[0], a[0]);
            c[1] = vdotq_s32(c[1], abcd0.val[1], a[0]);
            c[2] = vdotq_s32(c[2], abcd1.val[0], a[0]);
            c[3] = vdotq_s32(c[3], abcd1.val[1], a[0]);
            c[4] = vdotq_s32(c[4], abcd2.val[0], a[0]);
            c[5] = vdotq_s32(c[5], abcd2.val[1], a[0]);
            c[6] = vdotq_s32(c[6], abcd3.val[0], a[0]);
            c[7] = vdotq_s32(c[7], abcd3.val[1], a[0]);
#define cb(step)                                                              \
    b[0][step * 2 + 0] = vld1q_s8(B + (k + step + k_block_x2) * Bstride + n); \
    b[0][step * 2 + 1] = vld1q_s8(B + (k + step + k_block_x2) * Bstride + n + 16);
            UNROLL_CALL_RAW(4, cb);
#undef cb
            a[0] = vld1q_dup_s32(A + k + k_block_x2);

            ab0 = vzipq_s8(b[1][0], b[1][2]);
            cd0 = vzipq_s8(b[1][4], b[1][6]);
            ab1 = vzipq_s8(b[1][1], b[1][3]);
            cd1 = vzipq_s8(b[1][5], b[1][7]);

            abcd0 = vzipq_s16(ab0.val[0], cd0.val[0]);
            abcd1 = vzipq_s16(ab0.val[1], cd0.val[1]);
            abcd2 = vzipq_s16(ab1.val[0], cd1.val[0]);
            abcd3 = vzipq_s16(ab1.val[1], cd1.val[1]);
            c[0] = vdotq_s32(c[0], abcd0.val[0], a[1]);
            c[1] = vdotq_s32(c[1], abcd0.val[1], a[1]);
            c[2] = vdotq_s32(c[2], abcd1.val[0], a[1]);
            c[3] = vdotq_s32(c[3], abcd1.val[1], a[1]);
            c[4] = vdotq_s32(c[4], abcd2.val[0], a[1]);
            c[5] = vdotq_s32(c[5], abcd2.val[1], a[1]);
            c[6] = vdotq_s32(c[6], abcd3.val[0], a[1]);
            c[7] = vdotq_s32(c[7], abcd3.val[1], a[1]);
        }
        //! double buffer remain
#define cb(step)                                                                      \
    b[1][step * 2 + 0] = vld1q_s8(B + (k_buffer_end + step + k_block) * Bstride + n); \
    b[1][step * 2 + 1] =                                                              \
            vld1q_s8(B + (k_buffer_end + step + k_block) * Bstride + n + 16);
        UNROLL_CALL_RAW(4, cb);
#undef cb
        a[1] = vld1q_dup_s32(A + k_buffer_end + k_block);

        int8x16x2_t ab0 = vzipq_s8(b[0][0], b[0][2]);
        int8x16x2_t cd0 = vzipq_s8(b[0][4], b[0][6]);
        int8x16x2_t ab1 = vzipq_s8(b[0][1], b[0][3]);
        int8x16x2_t cd1 = vzipq_s8(b[0][5], b[0][7]);
        int16x8x2_t abcd0 = vzipq_s16(ab0.val[0], cd0.val[0]);
        int16x8x2_t abcd1 = vzipq_s16(ab0.val[1], cd0.val[1]);
        int16x8x2_t abcd2 = vzipq_s16(ab1.val[0], cd1.val[0]);
        int16x8x2_t abcd3 = vzipq_s16(ab1.val[1], cd1.val[1]);
        c[0] = vdotq_s32(c[0], abcd0.val[0], a[0]);
        c[1] = vdotq_s32(c[1], abcd0.val[1], a[0]);
        c[2] = vdotq_s32(c[2], abcd1.val[0], a[0]);
        c[3] = vdotq_s32(c[3], abcd1.val[1], a[0]);
        c[4] = vdotq_s32(c[4], abcd2.val[0], a[0]);
        c[5] = vdotq_s32(c[5], abcd2.val[1], a[0]);
        c[6] = vdotq_s32(c[6], abcd3.val[0], a[0]);
        c[7] = vdotq_s32(c[7], abcd3.val[1], a[0]);

        ab0 = vzipq_s8(b[1][0], b[1][2]);
        cd0 = vzipq_s8(b[1][4], b[1][6]);
        ab1 = vzipq_s8(b[1][1], b[1][3]);
        cd1 = vzipq_s8(b[1][5], b[1][7]);
        abcd0 = vzipq_s16(ab0.val[0], cd0.val[0]);
        abcd1 = vzipq_s16(ab0.val[1], cd0.val[1]);
        abcd2 = vzipq_s16(ab1.val[0], cd1.val[0]);
        abcd3 = vzipq_s16(ab1.val[1], cd1.val[1]);
        c[0] = vdotq_s32(c[0], abcd0.val[0], a[1]);
        c[1] = vdotq_s32(c[1], abcd0.val[1], a[1]);
        c[2] = vdotq_s32(c[2], abcd1.val[0], a[1]);
        c[3] = vdotq_s32(c[3], abcd1.val[1], a[1]);
        c[4] = vdotq_s32(c[4], abcd2.val[0], a[1]);
        c[5] = vdotq_s32(c[5], abcd2.val[1], a[1]);
        c[6] = vdotq_s32(c[6], abcd3.val[0], a[1]);
        c[7] = vdotq_s32(c[7], abcd3.val[1], a[1]);

        vst1q_s32(C + n + 0 * 4, c[0]);
        vst1q_s32(C + n + 1 * 4, c[1]);
        vst1q_s32(C + n + 2 * 4, c[2]);
        vst1q_s32(C + n + 3 * 4, c[3]);
        vst1q_s32(C + n + 4 * 4, c[4]);
        vst1q_s32(C + n + 5 * 4, c[5]);
        vst1q_s32(C + n + 6 * 4, c[6]);
        vst1q_s32(C + n + 7 * 4, c[7]);
        if (k_remain > 0) {
            for (size_t k = k_end; k < K; ++k) {
                for (size_t i = 0; i < n_block; ++i) {
                    C[n + i] += A[k] * B[k * Bstride + n + i];
                }
            }
        }
    }

    if (n_remain > 0) {
        for (size_t n = n_end; n < N; ++n) {
            if (!load_c) {
                C[n] = 0;
            }
            for (size_t k = 0; k < K; ++k) {
                C[n] += A[k] * B[k * Bstride + n];
            }
        }
    }
}
#if MEGDNN_ARMV7
MEGDNN_ATTRIBUTE_TARGET("dotprod")
void gevm_naive_dot_n32k4_impl(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride,
        bool load_c) {
    //! input must be N/32, k/4, 32, 4
    //! TODO: add prefetch
    //! TODO: add double buffer
    constexpr size_t n_block = 32;
    constexpr size_t k_block = 4;
    for (size_t n = 0; n < N; n += n_block) {
        int32x4_t c[n_block / 4];
#define cb(step) c[step] = vdupq_n_s32(0);
        UNROLL_CALL_RAW(8, cb);
#undef cb
        const int8_t* b_base = B + n * K;
        for (size_t k = 0; k < K; k += k_block) {
            int8x16_t a[1];
            int8x16_t b[1][8];
#define cb(step) b[0][step] = vld1q_s8(b_base + k * 32 + 16 * step);
            UNROLL_CALL_RAW(8, cb);
#undef cb
            a[0] = vld1q_dup_s32(A + k);

            c[0] = vdotq_s32(c[0], b[0][0], a[0]);
            c[1] = vdotq_s32(c[1], b[0][1], a[0]);
            c[2] = vdotq_s32(c[2], b[0][2], a[0]);
            c[3] = vdotq_s32(c[3], b[0][3], a[0]);
            c[4] = vdotq_s32(c[4], b[0][4], a[0]);
            c[5] = vdotq_s32(c[5], b[0][5], a[0]);
            c[6] = vdotq_s32(c[6], b[0][6], a[0]);
            c[7] = vdotq_s32(c[7], b[0][7], a[0]);
        }
        vst1q_s32(C + n + 0 * 4, c[0]);
        vst1q_s32(C + n + 1 * 4, c[1]);
        vst1q_s32(C + n + 2 * 4, c[2]);
        vst1q_s32(C + n + 3 * 4, c[3]);
        vst1q_s32(C + n + 4 * 4, c[4]);
        vst1q_s32(C + n + 5 * 4, c[5]);
        vst1q_s32(C + n + 6 * 4, c[6]);
        vst1q_s32(C + n + 7 * 4, c[7]);
    }
}
#else
MEGDNN_ATTRIBUTE_TARGET("dotprod")
inline void n32k4_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t K) {
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    //! C q0-q7
    //! A q8-q9
    //! B q10-q25
    asm volatile(
            // load accumulator C
            "1:\n"
            "eor v0.16b,  v0.16b,  v0.16b\n"
            "eor v1.16b,  v1.16b,  v1.16b\n"
            "eor v2.16b,  v2.16b,  v2.16b\n"
            "eor v3.16b,  v3.16b,  v3.16b\n"
            "eor v4.16b,  v4.16b,  v4.16b\n"
            "eor v5.16b,  v5.16b,  v5.16b\n"
            "eor v6.16b,  v6.16b,  v6.16b\n"
            "eor v7.16b,  v7.16b,  v7.16b\n"

            "ld1r  {v8.4s}, [%[a_ptr]]\n"
            "ld1 {v10.4s, v11.4s, v12.4s, v13.4s}, [%[b_ptr]], 64\n"
            "ld1 {v14.4s, v15.4s, v16.4s, v17.4s}, [%[b_ptr]], 64\n"
            "add  %[a_ptr], %[a_ptr], #4\n"

            "cmp %w[k], #0\n"
            "beq 4f\n"

            "2: \n"
            // Loop proper
            "3:\n"
            "ld1r  {v9.4s}, [%[a_ptr]]\n"
            "sdot  v0.4s, v10.16b, v8.16b\n"
            "ldr   q18, [%[b_ptr], #0]\n"
            "sdot  v1.4s, v11.16b, v8.16b\n"
            "ldr   q19, [%[b_ptr], #16]\n"
            "sdot  v2.4s, v12.16b, v8.16b\n"
            "ldr   q20, [%[b_ptr], #32]\n"
            "add  %[a_ptr], %[a_ptr], #4\n"
            "sdot  v3.4s, v13.16b, v8.16b\n"
            "ldr   q21, [%[b_ptr], #48]\n"
            "sdot  v4.4s, v14.16b, v8.16b\n"
            "ldr   q22, [%[b_ptr], #64]\n"
            "sdot  v5.4s, v15.16b, v8.16b\n"
            "ldr   q23, [%[b_ptr], #80]\n"
            "sdot  v6.4s, v16.16b, v8.16b\n"
            "ldr   q24, [%[b_ptr], #96]\n"
            "sdot  v7.4s, v17.16b, v8.16b\n"
            "ldr   q25, [%[b_ptr], #112]\n"

            "ld1r  {v8.4s}, [%[a_ptr]]\n"
            "sdot  v0.4s, v18.16b, v9.16b\n"
            "ldr   q10, [%[b_ptr], #128]\n"
            "sdot  v1.4s, v19.16b, v9.16b\n"
            "ldr   q11, [%[b_ptr], #144]\n"
            "sdot  v2.4s, v20.16b, v9.16b\n"
            "ldr   q12, [%[b_ptr], #160]\n"
            "sdot  v3.4s, v21.16b, v9.16b\n"
            "ldr   q13, [%[b_ptr], #176]\n"
            "sdot  v4.4s, v22.16b, v9.16b\n"
            "ldr   q14, [%[b_ptr], #192]\n"
            "sdot  v5.4s, v23.16b, v9.16b\n"
            "ldr   q15, [%[b_ptr], #208]\n"
            "sdot  v6.4s, v24.16b, v9.16b\n"
            "ldr   q16, [%[b_ptr], #224]\n"
            "sdot  v7.4s, v25.16b, v9.16b\n"
            "ldr   q17, [%[b_ptr], #240]\n"

            "add  %[a_ptr], %[a_ptr], #4\n"
            "add  %[b_ptr], %[b_ptr], #256\n"

            "subs %w[k], %w[k], #1\n"
            "bne 3b\n"

            "4:\n"
            "cmp %w[oddk], #1\n"
            "beq 5f\n"
            // Even tail

            "ld1r  {v9.4s}, [%[a_ptr]]\n"
            "sdot  v0.4s, v10.16b, v8.16b\n"
            "ldr   q18, [%[b_ptr], #0]\n"
            "sdot  v1.4s, v11.16b, v8.16b\n"
            "ldr   q19, [%[b_ptr], #16]\n"
            "sdot  v2.4s, v12.16b, v8.16b\n"
            "ldr   q20, [%[b_ptr], #32]\n"
            "sdot  v3.4s, v13.16b, v8.16b\n"
            "ldr   q21, [%[b_ptr], #48]\n"
            "sdot  v4.4s, v14.16b, v8.16b\n"
            "ldr   q22, [%[b_ptr], #64]\n"
            "sdot  v5.4s, v15.16b, v8.16b\n"
            "ldr   q23, [%[b_ptr], #80]\n"
            "sdot  v6.4s, v16.16b, v8.16b\n"
            "ldr   q24, [%[b_ptr], #96]\n"
            "sdot  v7.4s, v17.16b, v8.16b\n"
            "ldr   q25, [%[b_ptr], #112]\n"

            "sdot  v0.4s, v18.16b, v9.16b\n"
            "sdot  v1.4s, v19.16b, v9.16b\n"
            "sdot  v2.4s, v20.16b, v9.16b\n"
            "sdot  v3.4s, v21.16b, v9.16b\n"
            "sdot  v4.4s, v22.16b, v9.16b\n"
            "sdot  v5.4s, v23.16b, v9.16b\n"
            "sdot  v6.4s, v24.16b, v9.16b\n"
            "sdot  v7.4s, v25.16b, v9.16b\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[c_ptr]], 64\n"
            "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[c_ptr]], 64\n"
            "b 6f\n"

            "5:\n"
            // Odd tail
            "sdot  v0.4s, v10.16b, v8.16b\n"
            "sdot  v1.4s, v11.16b, v8.16b\n"
            "sdot  v2.4s, v12.16b, v8.16b\n"
            "sdot  v3.4s, v13.16b, v8.16b\n"
            "sdot  v4.4s, v14.16b, v8.16b\n"
            "sdot  v5.4s, v15.16b, v8.16b\n"
            "sdot  v6.4s, v16.16b, v8.16b\n"
            "sdot  v7.4s, v17.16b, v8.16b\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[c_ptr]], 64\n"
            "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[c_ptr]], 64\n"
            "6:\n"

            : [a_ptr] "+r"(A), [b_ptr] "+r"(B), [k] "+r"(K), [c_ptr] "+r"(C),
              [oddk] "+r"(oddk)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
              "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "v24", "v25", "cc", "memory");
}
MEGDNN_ATTRIBUTE_TARGET("dotprod")
void gevm_naive_dot_n32k4_impl(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride,
        bool load_c) {
    //! input must be N/32, k/4, 32, 4
    //! TODO: add prefetch
    //! TODO: add double buffer
    constexpr size_t n_block = 32;
    for (size_t n = 0; n < N; n += n_block) {
        n32k4_dot(A, B + n * K, C + n, K / 4);
    }
}
#endif
}  // namespace

void arm_common::gevm_naive_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(M == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv, midout_iv("INT8_gevm_dot"_hash)) {
        size_t cache_size = 256 * 1024;
        size_t k_group = N * K / cache_size;
        constexpr size_t k_align = 8;
        if (k_group >= 2) {
            size_t k_per_group = ((K / k_group) + k_align - 1) / k_align * k_align;
            for (size_t k = 0; k < K; k += k_per_group) {
                size_t real_k = std::min(K - k, k_per_group);
                gevm_naive_dot_impl(
                        A + k, B + k * Bstride, C, M, N, real_k, Astride, Bstride,
                        Cstride, k != 0);
            }
        } else {
            gevm_naive_dot_impl(A, B, C, M, N, K, Astride, Bstride, Cstride, false);
        }
    }
    MIDOUT_END();
}

void arm_common::gevm_naive_n32k4_dot(
        const int8_t* __restrict A, const int8_t* __restrict B, int32_t* __restrict C,
        size_t M, size_t N, size_t K, size_t Astride, size_t Bstride, size_t Cstride) {
    megdnn_assert(M == 1);
    MIDOUT_BEGIN(megdnn_arm_common_int8_gemv, midout_iv("INT8_gevm_dot_nk4"_hash)) {
        gevm_naive_dot_n32k4_impl(A, B, C, M, N, K, Astride, Bstride, Cstride, false);
    }
    MIDOUT_END();
}
#endif
// vim: syntax=cpp.doxygen
