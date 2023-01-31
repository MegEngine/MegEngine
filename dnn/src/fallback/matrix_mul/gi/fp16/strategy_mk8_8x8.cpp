#include "src/fallback/general_intrinsic/gi_float16.h"

#if defined(GI_SUPPORT_F16)
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/generic_strategy.h"

using namespace megdnn;
using namespace matmul::fallback;

namespace {

#define MLA GiMultiplyAddScalarFloat16
void kern_8x1(
        const gi_float16_t* A, const gi_float16_t* B, size_t LDB, size_t K,
        gi_float16_t* C) {
    LDB = LDB - 8;
    K = K - 8;

    GI_FLOAT16_t d0 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d1 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d2 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d3 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d4 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d5 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d6 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d7 = GiLoadFloat16(A);
    A = A + 8;

    GI_FLOAT16_t vfzero = GiBroadcastFloat16(0.0);

    GI_FLOAT16_t d8 = MLA(vfzero, d0, *(B));
    d8 = MLA(d8, d1, *(B + 1));
    d8 = MLA(d8, d2, *(B + 2));
    d8 = MLA(d8, d3, *(B + 3));
    d8 = MLA(d8, d4, *(B + 4));
    d8 = MLA(d8, d5, *(B + 5));
    d8 = MLA(d8, d6, *(B + 6));
    d8 = MLA(d8, d7, *(B + 7));
    B += 8;

    B += LDB;

    for (; K > 0; K -= 8) {
        d0 = GiLoadFloat16(A);
        A = A + 8;
        d1 = GiLoadFloat16(A);
        A = A + 8;
        d2 = GiLoadFloat16(A);
        A = A + 8;
        d3 = GiLoadFloat16(A);
        A = A + 8;
        d4 = GiLoadFloat16(A);
        A = A + 8;
        d5 = GiLoadFloat16(A);
        A = A + 8;
        d6 = GiLoadFloat16(A);
        A = A + 8;
        d7 = GiLoadFloat16(A);
        A = A + 8;

        d8 = MLA(d8, d0, *(B));
        d8 = MLA(d8, d1, *(B + 1));
        d8 = MLA(d8, d2, *(B + 2));
        d8 = MLA(d8, d3, *(B + 3));
        d8 = MLA(d8, d4, *(B + 4));
        d8 = MLA(d8, d5, *(B + 5));
        d8 = MLA(d8, d6, *(B + 6));
        d8 = MLA(d8, d7, *(B + 7));
        B += 8;

        B += LDB;
    }

    GiStoreFloat16(C, d8);
}

void kern_8x4(
        const gi_float16_t* A, const gi_float16_t* B, size_t LDB, size_t K,
        gi_float16_t* C) {
    LDB = LDB - 32;
    K = K - 8;

    GI_FLOAT16_t d0 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d1 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d2 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d3 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d4 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d5 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d6 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d7 = GiLoadFloat16(A);
    A = A + 8;

    GI_FLOAT16_t vfzero = GiBroadcastFloat16(0.0);

    GI_FLOAT16_t d8 = MLA(vfzero, d0, *(B));
    d8 = MLA(d8, d1, *(B + 1));
    d8 = MLA(d8, d2, *(B + 2));
    d8 = MLA(d8, d3, *(B + 3));
    d8 = MLA(d8, d4, *(B + 4));
    d8 = MLA(d8, d5, *(B + 5));
    d8 = MLA(d8, d6, *(B + 6));
    d8 = MLA(d8, d7, *(B + 7));
    B += 8;

    GI_FLOAT16_t d9 = MLA(vfzero, d0, *(B));
    d9 = MLA(d9, d1, *(B + 1));
    d9 = MLA(d9, d2, *(B + 2));
    d9 = MLA(d9, d3, *(B + 3));
    d9 = MLA(d9, d4, *(B + 4));
    d9 = MLA(d9, d5, *(B + 5));
    d9 = MLA(d9, d6, *(B + 6));
    d9 = MLA(d9, d7, *(B + 7));
    B += 8;

    GI_FLOAT16_t d10 = MLA(vfzero, d0, *(B));
    d10 = MLA(d10, d1, *(B + 1));
    d10 = MLA(d10, d2, *(B + 2));
    d10 = MLA(d10, d3, *(B + 3));
    d10 = MLA(d10, d4, *(B + 4));
    d10 = MLA(d10, d5, *(B + 5));
    d10 = MLA(d10, d6, *(B + 6));
    d10 = MLA(d10, d7, *(B + 7));
    B += 8;

    GI_FLOAT16_t d11 = MLA(vfzero, d0, *(B));
    d11 = MLA(d11, d1, *(B + 1));
    d11 = MLA(d11, d2, *(B + 2));
    d11 = MLA(d11, d3, *(B + 3));
    d11 = MLA(d11, d4, *(B + 4));
    d11 = MLA(d11, d5, *(B + 5));
    d11 = MLA(d11, d6, *(B + 6));
    d11 = MLA(d11, d7, *(B + 7));
    B += 8;

    B += LDB;

    for (; K > 0; K -= 8) {
        d0 = GiLoadFloat16(A);
        A = A + 8;
        d1 = GiLoadFloat16(A);
        A = A + 8;
        d2 = GiLoadFloat16(A);
        A = A + 8;
        d3 = GiLoadFloat16(A);
        A = A + 8;
        d4 = GiLoadFloat16(A);
        A = A + 8;
        d5 = GiLoadFloat16(A);
        A = A + 8;
        d6 = GiLoadFloat16(A);
        A = A + 8;
        d7 = GiLoadFloat16(A);
        A = A + 8;

        d8 = MLA(d8, d0, *(B));
        d8 = MLA(d8, d1, *(B + 1));
        d8 = MLA(d8, d2, *(B + 2));
        d8 = MLA(d8, d3, *(B + 3));
        d8 = MLA(d8, d4, *(B + 4));
        d8 = MLA(d8, d5, *(B + 5));
        d8 = MLA(d8, d6, *(B + 6));
        d8 = MLA(d8, d7, *(B + 7));
        B += 8;

        d9 = MLA(d9, d0, *(B));
        d9 = MLA(d9, d1, *(B + 1));
        d9 = MLA(d9, d2, *(B + 2));
        d9 = MLA(d9, d3, *(B + 3));
        d9 = MLA(d9, d4, *(B + 4));
        d9 = MLA(d9, d5, *(B + 5));
        d9 = MLA(d9, d6, *(B + 6));
        d9 = MLA(d9, d7, *(B + 7));
        B += 8;

        d10 = MLA(d10, d0, *(B));
        d10 = MLA(d10, d1, *(B + 1));
        d10 = MLA(d10, d2, *(B + 2));
        d10 = MLA(d10, d3, *(B + 3));
        d10 = MLA(d10, d4, *(B + 4));
        d10 = MLA(d10, d5, *(B + 5));
        d10 = MLA(d10, d6, *(B + 6));
        d10 = MLA(d10, d7, *(B + 7));
        B += 8;

        d11 = MLA(d11, d0, *(B));
        d11 = MLA(d11, d1, *(B + 1));
        d11 = MLA(d11, d2, *(B + 2));
        d11 = MLA(d11, d3, *(B + 3));
        d11 = MLA(d11, d4, *(B + 4));
        d11 = MLA(d11, d5, *(B + 5));
        d11 = MLA(d11, d6, *(B + 6));
        d11 = MLA(d11, d7, *(B + 7));
        B += 8;

        B += LDB;
    }

    GiStoreFloat16(C, d8);
    C = C + 8;
    GiStoreFloat16(C, d9);
    C = C + 8;
    GiStoreFloat16(C, d10);
    C = C + 8;
    GiStoreFloat16(C, d11);
    C = C + 8;
}

void kern_8x8(
        const gi_float16_t* A, const gi_float16_t* B, size_t LDB, size_t K,
        gi_float16_t* C) {
    LDB -= 64;
    K = K - 8;

    GI_FLOAT16_t d0 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d1 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d2 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d3 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d4 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d5 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d6 = GiLoadFloat16(A);
    A = A + 8;
    GI_FLOAT16_t d7 = GiLoadFloat16(A);
    A = A + 8;

    GI_FLOAT16_t vfzero = GiZeroFloat16();

    GI_FLOAT16_t d8 = MLA(vfzero, d0, *(B));
    d8 = MLA(d8, d1, *(B + 1));
    d8 = MLA(d8, d2, *(B + 2));
    d8 = MLA(d8, d3, *(B + 3));
    d8 = MLA(d8, d4, *(B + 4));
    d8 = MLA(d8, d5, *(B + 5));
    d8 = MLA(d8, d6, *(B + 6));
    d8 = MLA(d8, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d9 = MLA(vfzero, d0, *(B));
    d9 = MLA(d9, d1, *(B + 1));
    d9 = MLA(d9, d2, *(B + 2));
    d9 = MLA(d9, d3, *(B + 3));
    d9 = MLA(d9, d4, *(B + 4));
    d9 = MLA(d9, d5, *(B + 5));
    d9 = MLA(d9, d6, *(B + 6));
    d9 = MLA(d9, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d10 = MLA(vfzero, d0, *(B));
    d10 = MLA(d10, d1, *(B + 1));
    d10 = MLA(d10, d2, *(B + 2));
    d10 = MLA(d10, d3, *(B + 3));
    d10 = MLA(d10, d4, *(B + 4));
    d10 = MLA(d10, d5, *(B + 5));
    d10 = MLA(d10, d6, *(B + 6));
    d10 = MLA(d10, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d11 = MLA(vfzero, d0, *(B));
    d11 = MLA(d11, d1, *(B + 1));
    d11 = MLA(d11, d2, *(B + 2));
    d11 = MLA(d11, d3, *(B + 3));
    d11 = MLA(d11, d4, *(B + 4));
    d11 = MLA(d11, d5, *(B + 5));
    d11 = MLA(d11, d6, *(B + 6));
    d11 = MLA(d11, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d12 = MLA(vfzero, d0, *(B));
    d12 = MLA(d12, d1, *(B + 1));
    d12 = MLA(d12, d2, *(B + 2));
    d12 = MLA(d12, d3, *(B + 3));
    d12 = MLA(d12, d4, *(B + 4));
    d12 = MLA(d12, d5, *(B + 5));
    d12 = MLA(d12, d6, *(B + 6));
    d12 = MLA(d12, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d13 = MLA(vfzero, d0, *(B));
    d13 = MLA(d13, d1, *(B + 1));
    d13 = MLA(d13, d2, *(B + 2));
    d13 = MLA(d13, d3, *(B + 3));
    d13 = MLA(d13, d4, *(B + 4));
    d13 = MLA(d13, d5, *(B + 5));
    d13 = MLA(d13, d6, *(B + 6));
    d13 = MLA(d13, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d14 = MLA(vfzero, d0, *(B));
    d14 = MLA(d14, d1, *(B + 1));
    d14 = MLA(d14, d2, *(B + 2));
    d14 = MLA(d14, d3, *(B + 3));
    d14 = MLA(d14, d4, *(B + 4));
    d14 = MLA(d14, d5, *(B + 5));
    d14 = MLA(d14, d6, *(B + 6));
    d14 = MLA(d14, d7, *(B + 7));
    B = B + 8;

    GI_FLOAT16_t d15 = MLA(vfzero, d0, *(B));
    d15 = MLA(d15, d1, *(B + 1));
    d15 = MLA(d15, d2, *(B + 2));
    d15 = MLA(d15, d3, *(B + 3));
    d15 = MLA(d15, d4, *(B + 4));
    d15 = MLA(d15, d5, *(B + 5));
    d15 = MLA(d15, d6, *(B + 6));
    d15 = MLA(d15, d7, *(B + 7));
    B = B + 8;

    B = B + LDB;
    for (; K > 0; K -= 8) {
        d0 = GiLoadFloat16(A);
        A = A + 8;
        d1 = GiLoadFloat16(A);
        A = A + 8;
        d2 = GiLoadFloat16(A);
        A = A + 8;
        d3 = GiLoadFloat16(A);
        A = A + 8;
        d4 = GiLoadFloat16(A);
        A = A + 8;
        d5 = GiLoadFloat16(A);
        A = A + 8;
        d6 = GiLoadFloat16(A);
        A = A + 8;
        d7 = GiLoadFloat16(A);
        A = A + 8;

        d8 = MLA(d8, d0, *(B));
        d8 = MLA(d8, d1, *(B + 1));
        d8 = MLA(d8, d2, *(B + 2));
        d8 = MLA(d8, d3, *(B + 3));
        d8 = MLA(d8, d4, *(B + 4));
        d8 = MLA(d8, d5, *(B + 5));
        d8 = MLA(d8, d6, *(B + 6));
        d8 = MLA(d8, d7, *(B + 7));
        B = B + 8;

        d9 = MLA(d9, d0, *(B));
        d9 = MLA(d9, d1, *(B + 1));
        d9 = MLA(d9, d2, *(B + 2));
        d9 = MLA(d9, d3, *(B + 3));
        d9 = MLA(d9, d4, *(B + 4));
        d9 = MLA(d9, d5, *(B + 5));
        d9 = MLA(d9, d6, *(B + 6));
        d9 = MLA(d9, d7, *(B + 7));
        B = B + 8;

        d10 = MLA(d10, d0, *(B));
        d10 = MLA(d10, d1, *(B + 1));
        d10 = MLA(d10, d2, *(B + 2));
        d10 = MLA(d10, d3, *(B + 3));
        d10 = MLA(d10, d4, *(B + 4));
        d10 = MLA(d10, d5, *(B + 5));
        d10 = MLA(d10, d6, *(B + 6));
        d10 = MLA(d10, d7, *(B + 7));
        B = B + 8;

        d11 = MLA(d11, d0, *(B));
        d11 = MLA(d11, d1, *(B + 1));
        d11 = MLA(d11, d2, *(B + 2));
        d11 = MLA(d11, d3, *(B + 3));
        d11 = MLA(d11, d4, *(B + 4));
        d11 = MLA(d11, d5, *(B + 5));
        d11 = MLA(d11, d6, *(B + 6));
        d11 = MLA(d11, d7, *(B + 7));
        B = B + 8;

        d12 = MLA(d12, d0, *(B));
        d12 = MLA(d12, d1, *(B + 1));
        d12 = MLA(d12, d2, *(B + 2));
        d12 = MLA(d12, d3, *(B + 3));
        d12 = MLA(d12, d4, *(B + 4));
        d12 = MLA(d12, d5, *(B + 5));
        d12 = MLA(d12, d6, *(B + 6));
        d12 = MLA(d12, d7, *(B + 7));
        B = B + 8;

        d13 = MLA(d13, d0, *(B));
        d13 = MLA(d13, d1, *(B + 1));
        d13 = MLA(d13, d2, *(B + 2));
        d13 = MLA(d13, d3, *(B + 3));
        d13 = MLA(d13, d4, *(B + 4));
        d13 = MLA(d13, d5, *(B + 5));
        d13 = MLA(d13, d6, *(B + 6));
        d13 = MLA(d13, d7, *(B + 7));
        B = B + 8;

        d14 = MLA(d14, d0, *(B));
        d14 = MLA(d14, d1, *(B + 1));
        d14 = MLA(d14, d2, *(B + 2));
        d14 = MLA(d14, d3, *(B + 3));
        d14 = MLA(d14, d4, *(B + 4));
        d14 = MLA(d14, d5, *(B + 5));
        d14 = MLA(d14, d6, *(B + 6));
        d14 = MLA(d14, d7, *(B + 7));
        B = B + 8;

        d15 = MLA(d15, d0, *(B));
        d15 = MLA(d15, d1, *(B + 1));
        d15 = MLA(d15, d2, *(B + 2));
        d15 = MLA(d15, d3, *(B + 3));
        d15 = MLA(d15, d4, *(B + 4));
        d15 = MLA(d15, d5, *(B + 5));
        d15 = MLA(d15, d6, *(B + 6));
        d15 = MLA(d15, d7, *(B + 7));
        B = B + 8 + LDB;
    }
    GiStoreFloat16(C, d8);
    C = C + 8;
    GiStoreFloat16(C, d9);
    C = C + 8;
    GiStoreFloat16(C, d10);
    C = C + 8;
    GiStoreFloat16(C, d11);
    C = C + 8;
    GiStoreFloat16(C, d12);
    C = C + 8;
    GiStoreFloat16(C, d13);
    C = C + 8;
    GiStoreFloat16(C, d14);
    C = C + 8;
    GiStoreFloat16(C, d15);
    C = C + 8;
}

#undef MLA
}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(gi_sgemm_nopack_mk8_8x8_fp16);

void gi_sgemm_nopack_mk8_8x8_fp16::kern(
        const dt_float16* A, size_t LDA, const dt_float16* B, size_t LDB, dt_float16* C,
        size_t LDC, size_t M, size_t K, size_t N, const dt_float16*, void*, bool trA,
        bool trB) const {
    constexpr size_t MB = 8;
    constexpr size_t KB = 8;
    constexpr size_t NB = 8;
    constexpr size_t NB_HALF = 4;
    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    for (size_t m = 0; m < M; m += MB) {
        gi_float16_t* output = reinterpret_cast<gi_float16_t*>(C) + (m / MB) * LDC;
        const gi_float16_t* cur_B = reinterpret_cast<const gi_float16_t*>(B);
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_8x8(reinterpret_cast<const gi_float16_t*>(A), cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_8x4(reinterpret_cast<const gi_float16_t*>(A), cur_B, LDB, K, output);
            cur_B += KB * NB_HALF;
            output += MB * NB_HALF;
            n += 4;
        }
        while (n < N) {
            kern_8x1(reinterpret_cast<const gi_float16_t*>(A), cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
}

#endif
// vim: syntax=cpp.doxygen
