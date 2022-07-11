//! risc-v gcc will error report uninitialized var at if/else case when use RVV type
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"

#ifdef __GNUC__
#ifndef __has_warning
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#else
#if __has_warning("-Wmaybe-uninitialized")
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
#endif

#include "src/fallback/matrix_mul/generic_strategy.h"
#include "src/fallback/matrix_mul/gi/fp32/common.h"

using namespace megdnn;
using namespace matmul::fallback;

namespace {

//! x86 and rvv GiSimdFmaLane API is slowly, as an alternate, use
//! GiMultiplyAddScalarFloat32
#define MLA GiMultiplyAddScalarFloat32
void kern_4x12(
        const float* packA, const float* packB, int K, float* output, int LDC,
        bool is_first_k) {
    MEGDNN_MARK_USED_VAR(LDC);
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    float* r1 = output;

    GI_FLOAT32_t d0d1, d2d3, d8d9, d10d11, d12d13, d14d15, d16d17, d18d19, d20d21,
            d22d23, d24d25, d26d27, d28d29, d30d31;

    if (is_first_k) {
        d8d9 = GiBroadcastFloat32(0.0f);
        d10d11 = GiBroadcastFloat32(0.0f);
        d12d13 = GiBroadcastFloat32(0.0f);
        d14d15 = GiBroadcastFloat32(0.0f);
        d16d17 = GiBroadcastFloat32(0.0f);
        d18d19 = GiBroadcastFloat32(0.0f);
        d20d21 = GiBroadcastFloat32(0.0f);
        d22d23 = GiBroadcastFloat32(0.0f);
        d24d25 = GiBroadcastFloat32(0.0f);
        d26d27 = GiBroadcastFloat32(0.0f);
        d28d29 = GiBroadcastFloat32(0.0f);
        d30d31 = GiBroadcastFloat32(0.0f);
    } else {
        d8d9 = GiLoadFloat32(r1);
        r1 = r1 + 4;
        d10d11 = GiLoadFloat32(r1);
        r1 = r1 + 4;

        d12d13 = GiLoadFloat32(r1);
        r1 = r1 + 4;
        d14d15 = GiLoadFloat32(r1);
        r1 = r1 + 4;

        d16d17 = GiLoadFloat32(r1);
        r1 = r1 + 4;
        d18d19 = GiLoadFloat32(r1);
        r1 = r1 + 4;

        d20d21 = GiLoadFloat32(r1);
        r1 = r1 + 4;
        d22d23 = GiLoadFloat32(r1);
        r1 = r1 + 4;

        d24d25 = GiLoadFloat32(r1);
        r1 = r1 + 4;
        d26d27 = GiLoadFloat32(r1);
        r1 = r1 + 4;

        d28d29 = GiLoadFloat32(r1);
        r1 = r1 + 4;
        d30d31 = GiLoadFloat32(r1);
        r1 = r1 + 4;
    }
    for (; K > 0; K--) {
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;

        d8d9 = MLA(d8d9, d0d1, *(b_ptr));
        d10d11 = MLA(d10d11, d0d1, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d0d1, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d16d17 = MLA(d16d17, d0d1, *(b_ptr));
        d18d19 = MLA(d18d19, d0d1, *(b_ptr + 1));
        d20d21 = MLA(d20d21, d0d1, *(b_ptr + 2));
        d22d23 = MLA(d22d23, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d24d25 = MLA(d24d25, d0d1, *(b_ptr));
        d26d27 = MLA(d26d27, d0d1, *(b_ptr + 1));
        d28d29 = MLA(d28d29, d0d1, *(b_ptr + 2));
        d30d31 = MLA(d30d31, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, *(b_ptr));
        d10d11 = MLA(d10d11, d2d3, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d2d3, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d2d3, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d16d17 = MLA(d16d17, d2d3, *(b_ptr));
        d18d19 = MLA(d18d19, d2d3, *(b_ptr + 1));
        d20d21 = MLA(d20d21, d2d3, *(b_ptr + 2));
        d22d23 = MLA(d22d23, d2d3, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d24d25 = MLA(d24d25, d2d3, *(b_ptr));
        d26d27 = MLA(d26d27, d2d3, *(b_ptr + 1));
        d28d29 = MLA(d28d29, d2d3, *(b_ptr + 2));
        d30d31 = MLA(d30d31, d2d3, *(b_ptr + 3));
        b_ptr = b_ptr + 4;
    }

    d0d1 = GiLoadFloat32(a_ptr);
    a_ptr = a_ptr + 4;
    if (1 == oddk) {
        d8d9 = MLA(d8d9, d0d1, *(b_ptr));
        d10d11 = MLA(d10d11, d0d1, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d0d1, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d16d17 = MLA(d16d17, d0d1, *(b_ptr));
        GiStoreFloat32(output0, d8d9);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d10d11);
        output0 = output0 + 4;
        d18d19 = MLA(d18d19, d0d1, *(b_ptr + 1));
        d20d21 = MLA(d20d21, d0d1, *(b_ptr + 2));
        GiStoreFloat32(output0, d12d13);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d14d15);
        output0 = output0 + 4;
        d22d23 = MLA(d22d23, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d24d25 = MLA(d24d25, d0d1, *(b_ptr));
        GiStoreFloat32(output0, d16d17);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d18d19);
        output0 = output0 + 4;
        d26d27 = MLA(d26d27, d0d1, *(b_ptr + 1));
        GiStoreFloat32(output0, d20d21);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d22d23);
        output0 = output0 + 4;
        d28d29 = MLA(d28d29, d0d1, *(b_ptr + 2));
        GiStoreFloat32(output0, d24d25);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d26d27);
        output0 = output0 + 4;
        d30d31 = MLA(d30d31, d0d1, *(b_ptr + 3));
        GiStoreFloat32(output0, d28d29);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d30d31);
        output0 = output0 + 4;
        b_ptr = b_ptr + 4;
    } else {
        d8d9 = MLA(d8d9, d0d1, *(b_ptr));
        d10d11 = MLA(d10d11, d0d1, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d0d1, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d16d17 = MLA(d16d17, d0d1, *(b_ptr));
        d18d19 = MLA(d18d19, d0d1, *(b_ptr + 1));
        d20d21 = MLA(d20d21, d0d1, *(b_ptr + 2));
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d22d23 = MLA(d22d23, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d24d25 = MLA(d24d25, d0d1, *(b_ptr));
        d26d27 = MLA(d26d27, d0d1, *(b_ptr + 1));
        d28d29 = MLA(d28d29, d0d1, *(b_ptr + 2));
        d30d31 = MLA(d30d31, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, *(b_ptr));
        d10d11 = MLA(d10d11, d2d3, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d2d3, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d2d3, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d16d17 = MLA(d16d17, d2d3, *(b_ptr));
        d18d19 = MLA(d18d19, d2d3, *(b_ptr + 1));
        GiStoreFloat32(output0, d8d9);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d10d11);
        output0 = output0 + 4;
        d20d21 = MLA(d20d21, d2d3, *(b_ptr + 2));
        d22d23 = MLA(d22d23, d2d3, *(b_ptr + 3));
        GiStoreFloat32(output0, d12d13);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d14d15);
        output0 = output0 + 4;
        b_ptr = b_ptr + 4;

        d24d25 = MLA(d24d25, d2d3, *(b_ptr));
        d26d27 = MLA(d26d27, d2d3, *(b_ptr + 1));
        GiStoreFloat32(output0, d16d17);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d18d19);
        output0 = output0 + 4;
        d28d29 = MLA(d28d29, d2d3, *(b_ptr + 2));
        d30d31 = MLA(d30d31, d2d3, *(b_ptr + 3));
        GiStoreFloat32(output0, d20d21);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d22d23);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d24d25);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d26d27);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d28d29);
        output0 = output0 + 4;
        GiStoreFloat32(output0, d30d31);
        output0 = output0 + 4;
        b_ptr = b_ptr + 4;
    }
}

void kern_4x4(
        const float* packA, const float* packB, int K, float* output, int LDC,
        bool is_first_k, int n_remain) {
    MEGDNN_MARK_USED_VAR(LDC);
    const float* a_ptr = packA;
    const float* b_ptr = packB;

    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    float* r1 = output;

    GI_FLOAT32_t d0d1, d2d3, d8d9, d10d11, d12d13, d14d15;

    if (is_first_k) {
        d8d9 = GiBroadcastFloat32(0.0f);
        d10d11 = GiBroadcastFloat32(0.0f);

        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;

        d12d13 = GiBroadcastFloat32(0.0f);

        d14d15 = GiBroadcastFloat32(0.0f);
    } else {
        if (n_remain == 4) {
            d8d9 = GiLoadFloat32(r1);
            r1 = r1 + 4;
            d10d11 = GiLoadFloat32(r1);
            r1 = r1 + 4;
            d12d13 = GiLoadFloat32(r1);
            r1 = r1 + 4;
            d14d15 = GiLoadFloat32(r1);
            r1 = r1 + 4;
        } else if (n_remain == 3) {
            d8d9 = GiLoadFloat32(r1);
            r1 = r1 + 4;
            d10d11 = GiLoadFloat32(r1);
            r1 = r1 + 4;
            d12d13 = GiLoadFloat32(r1);
            r1 = r1 + 4;
        } else if (n_remain == 2) {
            d8d9 = GiLoadFloat32(r1);
            r1 = r1 + 4;
            d10d11 = GiLoadFloat32(r1);
            r1 = r1 + 4;
        } else if (n_remain == 1) {
            d8d9 = GiLoadFloat32(r1);
            r1 = r1 + 4;
        }
    }

    for (; K > 0; K--) {
        d8d9 = MLA(d8d9, d0d1, *(b_ptr));
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d10d11 = MLA(d10d11, d0d1, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d0d1, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, *(b_ptr));
        d10d11 = MLA(d10d11, d2d3, *(b_ptr + 1));
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d12d13 = MLA(d12d13, d2d3, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d2d3, *(b_ptr + 3));
        b_ptr = b_ptr + 4;
    }

    if (1 == oddk) {
        d8d9 = MLA(d8d9, d0d1, *(b_ptr));
        d10d11 = MLA(d10d11, d0d1, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d0d1, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;
    } else {
        d8d9 = MLA(d8d9, d0d1, *(b_ptr));
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d10d11 = MLA(d10d11, d0d1, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d0d1, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d0d1, *(b_ptr + 3));
        b_ptr = b_ptr + 4;

        d8d9 = MLA(d8d9, d2d3, *(b_ptr));
        d10d11 = MLA(d10d11, d2d3, *(b_ptr + 1));
        d12d13 = MLA(d12d13, d2d3, *(b_ptr + 2));
        d14d15 = MLA(d14d15, d2d3, *(b_ptr + 3));
        b_ptr = b_ptr + 4;
    }

    if (n_remain == 4) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
        GiStoreFloat32(output, d10d11);
        output = output + 4;
        GiStoreFloat32(output, d12d13);
        output = output + 4;
        GiStoreFloat32(output, d14d15);
        output = output + 4;
    } else if (n_remain == 3) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
        GiStoreFloat32(output, d10d11);
        output = output + 4;
        GiStoreFloat32(output, d12d13);
        output = output + 4;
    } else if (n_remain == 2) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
        GiStoreFloat32(output, d10d11);
        output = output + 4;
    } else if (n_remain == 1) {
        GiStoreFloat32(output, d8d9);
        output = output + 4;
    }
}
#undef MLA
}  // namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL(gi_sgemm_mk4_pack_4x12);
//! Now no matmul mode of only packB support in conv1x1 and im2col, so just copy
//! the weight
void gi_sgemm_mk4_pack_4x12::pack_A(
        float* out, const float* in, int ldin, int y0, int ymax, int k0, int kmax,
        bool) const {
    megdnn_assert(y0 % 4 == 0 && ymax % 4 == 0, "M must be time of 4");
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    constexpr int PACK_C_SIZE = 4;
    size_t cp_length = (kmax - k0) * PACK_C_SIZE;
    for (int m = y0; m < ymax; m += 4) {
        const float* src = in + (m / PACK_C_SIZE) * ldin + k0 * PACK_C_SIZE;
        memcpy(out, src, cp_length * sizeof(float));
        out += cp_length;
    }
}

void gi_sgemm_mk4_pack_4x12::pack_B(
        float* out, const float* in, int ldin, int x0, int xmax, int k0, int kmax,
        bool transpose_B) const {
    megdnn_assert(!transpose_B);
    megdnn_assert(k0 % 4 == 0 && kmax % 4 == 0, "K must be time of 4");
    float tmpbuff[16] = {0.0f};

    constexpr int PACK_C_SIZE = 4;
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = out;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;

    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr = in + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;

        int x = x0;
        auto outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            auto outptr_interleave = outptr;
            transpose_1x12_4_s(inptr, outptr_interleave);
            outptr += ksize12;
        }
        outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            auto outptr_interleave = outptr;
            transpose_1x4_4_s(inptr, outptr_interleave);
            outptr += ksize4;
        }
        if (x < xmax) {
            memcpy(tmpbuff, inptr, sizeof(float) * (xmax - x) * PACK_C_SIZE);
            auto outptr_interleave = outptr;
            const float* tmp_ptr = &tmpbuff[0];
            transpose_1x4_4_s<float>(tmp_ptr, outptr_interleave);
            outptr += ksize4;
        }
        outptr_base += 12 * PACK_C_SIZE;
        outptr_base4 += 4 * PACK_C_SIZE;
    }
}

void gi_sgemm_mk4_pack_4x12::kern(
        const float* packA, const float* packB, size_t M, size_t N, size_t K, float* C,
        size_t LDC, bool is_first_k, const float*, float*) const {
    megdnn_assert(
            A_dtype.enumv() == B_dtype.enumv() && A_dtype.enumv() == C_dtype.enumv() &&
            A_dtype.enumv() == DTypeEnum::Float32);
    constexpr int PACK_C_SIZE = 4;
    constexpr size_t A_INTERLEAVE = 4;
    constexpr size_t B_INTERLEAVE = 12;
    const int K12 = K * 12;
    const int K4 = K * 4;
    size_t m = 0;
    for (; m < M; m += A_INTERLEAVE) {
        float* output = C + (m / 4 * LDC);

        size_t n = 0;
        const float* cur_packB = packB;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(packA, cur_packB, K, output, LDC, is_first_k);
            output += PACK_C_SIZE * B_INTERLEAVE;
            cur_packB += K12;
        }
        for (; n < N; n += 4) {
            kern_4x4(
                    packA, cur_packB, K, output, LDC, is_first_k,
                    std::min<size_t>(N - n, 4));
            output += PACK_C_SIZE * 4;
            cur_packB += K4;
        }
        packA += K4;
    }
}

// vim: syntax=cpp.doxygen
