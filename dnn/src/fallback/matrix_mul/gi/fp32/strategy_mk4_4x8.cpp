#include "src/common/utils.h"
#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/matrix_mul/generic_strategy.h"

using namespace megdnn;
using namespace matmul::fallback;

namespace {

//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MLA GiMultiplyAddScalarFloat32
void kern_4x1(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB = LDB - 4;
    K = K - 4;

    GI_FLOAT32_t d8d9 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d12d13 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d16d17 = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d18d19 = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d20d21 = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d22d23 = GiBroadcastFloat32(0.0f);

    d16d17 = MLA(d16d17, d8d9, *(B));
    d18d19 = MLA(d18d19, d10d11, *(B + 1));
    B = B + 4;

    for (; K > 0; K -= 4) {
        d8d9 = GiLoadFloat32(A);
        A = A + 4;
        d10d11 = GiLoadFloat32(A);
        A = A + 4;

        d20d21 = MLA(d20d21, d12d13, *(B + 2 - 4));
        d22d23 = MLA(d22d23, d14d15, *(B + 3 - 4));
        B = B + LDB;

        d12d13 = GiLoadFloat32(A);
        A = A + 4;
        d14d15 = GiLoadFloat32(A);
        A = A + 4;

        d16d17 = MLA(d16d17, d8d9, *(B));
        d18d19 = MLA(d18d19, d10d11, *(B + 1));
        B = B + 4;
    }

    d20d21 = MLA(d20d21, d12d13, *(B + 2 - 4));
    d22d23 = MLA(d22d23, d14d15, *(B + 3 - 4));
    d16d17 = GiAddFloat32(d16d17, d20d21);
    d18d19 = GiAddFloat32(d18d19, d22d23);
    d16d17 = GiAddFloat32(d16d17, d18d19);

    GiStoreFloat32(C, d16d17);
    C = C + 4;
}

void kern_4x4(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB = (LDB - 16);
    K = K - 4;

    GI_FLOAT32_t d8d9 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d12d13 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(A);
    A = A + 4;

    GI_FLOAT32_t vfzero = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t d16d17 = MLA(vfzero, d8d9, *(B));
    d16d17 = MLA(d16d17, d10d11, *(B + 1));

    GI_FLOAT32_t d18d19 = MLA(vfzero, d8d9, *(B + 4));
    d18d19 = MLA(d18d19, d10d11, *(B + 5));

    GI_FLOAT32_t d20d21 = MLA(vfzero, d8d9, *(B + 8));
    d20d21 = MLA(d20d21, d10d11, *(B + 9));

    GI_FLOAT32_t d22d23 = MLA(vfzero, d8d9, *(B + 12));
    d22d23 = MLA(d22d23, d10d11, *(B + 13));
    B = B + 16;

    for (; K > 0; K -= 4) {
        d8d9 = GiLoadFloat32(A);
        A = A + 4;
        d10d11 = GiLoadFloat32(A);
        A = A + 4;

        d16d17 = MLA(d16d17, d12d13, *(B + 2 - 16));
        d16d17 = MLA(d16d17, d14d15, *(B + 3 - 16));

        d18d19 = MLA(d18d19, d12d13, *(B + 6 - 16));
        d18d19 = MLA(d18d19, d14d15, *(B + 7 - 16));

        d20d21 = MLA(d20d21, d12d13, *(B + 10 - 16));
        d20d21 = MLA(d20d21, d14d15, *(B + 11 - 16));

        d22d23 = MLA(d22d23, d12d13, *(B + 14 - 16));
        d22d23 = MLA(d22d23, d14d15, *(B + 15 - 16));

        B = B + LDB;

        d16d17 = MLA(d16d17, d8d9, *(B));
        d16d17 = MLA(d16d17, d10d11, *(B + 1));

        d18d19 = MLA(d18d19, d8d9, *(B + 4));
        d18d19 = MLA(d18d19, d10d11, *(B + 5));

        d20d21 = MLA(d20d21, d8d9, *(B + 8));
        d20d21 = MLA(d20d21, d10d11, *(B + 9));

        d22d23 = MLA(d22d23, d8d9, *(B + 12));
        d22d23 = MLA(d22d23, d10d11, *(B + 13));

        d12d13 = GiLoadFloat32(A);
        A = A + 4;
        d14d15 = GiLoadFloat32(A);
        A = A + 4;
        B = B + 16;
    }

    d16d17 = MLA(d16d17, d12d13, *(B + 2 - 16));
    d16d17 = MLA(d16d17, d14d15, *(B + 3 - 16));

    d18d19 = MLA(d18d19, d12d13, *(B + 6 - 16));
    d18d19 = MLA(d18d19, d14d15, *(B + 7 - 16));

    d20d21 = MLA(d20d21, d12d13, *(B + 10 - 16));
    d20d21 = MLA(d20d21, d14d15, *(B + 11 - 16));

    d22d23 = MLA(d22d23, d12d13, *(B + 14 - 16));
    d22d23 = MLA(d22d23, d14d15, *(B + 15 - 16));

    GiStoreFloat32(C, d16d17);
    C = C + 4;
    GiStoreFloat32(C, d18d19);
    C = C + 4;
    GiStoreFloat32(C, d20d21);
    C = C + 4;
    GiStoreFloat32(C, d22d23);
    C = C + 4;
}

void kern_4x8(const float* A, const float* B, size_t LDB, size_t K, float* C) {
    LDB -= 32;
    GI_FLOAT32_t d8d9 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d10d11 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d12d13 = GiLoadFloat32(A);
    A = A + 4;
    GI_FLOAT32_t d14d15 = GiLoadFloat32(A);
    A = A + 4;

    GI_FLOAT32_t vfzero = GiBroadcastFloat32(0.0f);

    GI_FLOAT32_t d16d17 = MLA(vfzero, d8d9, *(B));
    d16d17 = MLA(d16d17, d10d11, *(B + 1));
    d16d17 = MLA(d16d17, d12d13, *(B + 2));
    d16d17 = MLA(d16d17, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d18d19 = MLA(vfzero, d8d9, *(B));
    d18d19 = MLA(d18d19, d10d11, *(B + 1));
    d18d19 = MLA(d18d19, d12d13, *(B + 2));
    d18d19 = MLA(d18d19, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d20d21 = MLA(vfzero, d8d9, *(B));
    d20d21 = MLA(d20d21, d10d11, *(B + 1));
    d20d21 = MLA(d20d21, d12d13, *(B + 2));
    d20d21 = MLA(d20d21, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d22d23 = MLA(vfzero, d8d9, *(B));
    d22d23 = MLA(d22d23, d10d11, *(B + 1));
    d22d23 = MLA(d22d23, d12d13, *(B + 2));
    d22d23 = MLA(d22d23, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d24d25 = MLA(vfzero, d8d9, *(B));
    d24d25 = MLA(d24d25, d10d11, *(B + 1));
    d24d25 = MLA(d24d25, d12d13, *(B + 2));
    d24d25 = MLA(d24d25, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d26d27 = MLA(vfzero, d8d9, *(B));
    d26d27 = MLA(d26d27, d10d11, *(B + 1));
    d26d27 = MLA(d26d27, d12d13, *(B + 2));
    d26d27 = MLA(d26d27, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d28d29 = MLA(vfzero, d8d9, *(B));
    d28d29 = MLA(d28d29, d10d11, *(B + 1));
    d28d29 = MLA(d28d29, d12d13, *(B + 2));
    d28d29 = MLA(d28d29, d14d15, *(B + 3));
    B = B + 4;

    GI_FLOAT32_t d30d31 = MLA(vfzero, d8d9, *(B));
    d30d31 = MLA(d30d31, d10d11, *(B + 1));
    d30d31 = MLA(d30d31, d12d13, *(B + 2));
    d30d31 = MLA(d30d31, d14d15, *(B + 3));
    B = B + 4;

    B = B + LDB;
    K = K - 4;
    for (; K > 0; K -= 4) {
        d8d9 = GiLoadFloat32(A);
        A = A + 4;
        d10d11 = GiLoadFloat32(A);
        A = A + 4;
        d12d13 = GiLoadFloat32(A);
        A = A + 4;
        d14d15 = GiLoadFloat32(A);
        A = A + 4;

        d16d17 = MLA(d16d17, d8d9, *(B));
        d16d17 = MLA(d16d17, d10d11, *(B + 1));
        d16d17 = MLA(d16d17, d12d13, *(B + 2));
        d16d17 = MLA(d16d17, d14d15, *(B + 3));
        B = B + 4;

        d18d19 = MLA(d18d19, d8d9, *(B));
        d18d19 = MLA(d18d19, d10d11, *(B + 1));
        d18d19 = MLA(d18d19, d12d13, *(B + 2));
        d18d19 = MLA(d18d19, d14d15, *(B + 3));
        B = B + 4;

        d20d21 = MLA(d20d21, d8d9, *(B));
        d20d21 = MLA(d20d21, d10d11, *(B + 1));
        d20d21 = MLA(d20d21, d12d13, *(B + 2));
        d20d21 = MLA(d20d21, d14d15, *(B + 3));
        B = B + 4;

        d22d23 = MLA(d22d23, d8d9, *(B));
        d22d23 = MLA(d22d23, d10d11, *(B + 1));
        d22d23 = MLA(d22d23, d12d13, *(B + 2));
        d22d23 = MLA(d22d23, d14d15, *(B + 3));
        B = B + 4;

        d24d25 = MLA(d24d25, d8d9, *(B));
        d24d25 = MLA(d24d25, d10d11, *(B + 1));
        d24d25 = MLA(d24d25, d12d13, *(B + 2));
        d24d25 = MLA(d24d25, d14d15, *(B + 3));
        B = B + 4;

        d26d27 = MLA(d26d27, d8d9, *(B));
        d26d27 = MLA(d26d27, d10d11, *(B + 1));
        d26d27 = MLA(d26d27, d12d13, *(B + 2));
        d26d27 = MLA(d26d27, d14d15, *(B + 3));
        B = B + 4;

        d28d29 = MLA(d28d29, d8d9, *(B));
        d28d29 = MLA(d28d29, d10d11, *(B + 1));
        d28d29 = MLA(d28d29, d12d13, *(B + 2));
        d28d29 = MLA(d28d29, d14d15, *(B + 3));
        B = B + 4;

        d30d31 = MLA(d30d31, d8d9, *(B));
        d30d31 = MLA(d30d31, d10d11, *(B + 1));
        d30d31 = MLA(d30d31, d12d13, *(B + 2));
        d30d31 = MLA(d30d31, d14d15, *(B + 3));
        B = B + 4 + LDB;
    }
    GiStoreFloat32(C, d16d17);
    C = C + 4;
    GiStoreFloat32(C, d18d19);
    C = C + 4;
    GiStoreFloat32(C, d20d21);
    C = C + 4;
    GiStoreFloat32(C, d22d23);
    C = C + 4;
    GiStoreFloat32(C, d24d25);
    C = C + 4;
    GiStoreFloat32(C, d26d27);
    C = C + 4;
    GiStoreFloat32(C, d28d29);
    C = C + 4;
    GiStoreFloat32(C, d30d31);
    C = C + 4;
}

#undef MLA
}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(gi_sgemm_nopack_4x8);

void gi_sgemm_nopack_4x8::kern(
        const float* A, size_t LDA, const float* B, size_t LDB, float* C, size_t LDC,
        size_t M, size_t K, size_t N, const float*, void*, bool trA, bool trB) const {
    constexpr size_t MB = 4;
    constexpr size_t KB = 4;
    constexpr size_t NB = 8;
    constexpr size_t NB_HALF = 4;

    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    for (size_t m = 0; m < M; m += MB) {
        float* output = C + (m / MB) * LDC;
        const float* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_4x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_4x4(A, cur_B, LDB, K, output);
            cur_B += KB * NB_HALF;
            output += MB * NB_HALF;
            n += 4;
        }
        while (n < N) {
            kern_4x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
}

// vim: syntax=cpp.doxygen
