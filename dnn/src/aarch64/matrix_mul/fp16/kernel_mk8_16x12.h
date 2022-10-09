#pragma once
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/aarch64/matrix_mul/fp16/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace aarch64 {
struct matmul_mk8_16x12 {
    template <size_t M_BLOCK, size_t N_BLOCK>
    static void kern(
            const dt_float16* packedA, const dt_float16* packedB, int K,
            dt_float16* out, int LDC, bool is_first_k);

    static void hgemm_16x12_pack_A(
            dt_float16* outptr, const dt_float16* inptr, int ldin, int y0, int ymax,
            int k0, int kmax) {
        megdnn_assert(y0 % 8 == 0 && ymax % 8 == 0, "M must be time of 8");
        megdnn_assert(k0 % 8 == 0 && kmax % 8 == 0, "K must be time of 8");
        constexpr int PACK_SIZE_128 = 16 * 8;
        constexpr int PACK_SIZE_64 = 8 * 8;
        constexpr int PACK_C_SIZE = 8;
        int y = y0;
        for (; y + 15 < ymax; y += 16) {
            const dt_float16* inptr0 = inptr + y / PACK_C_SIZE * ldin + k0;
            const dt_float16* inptr1 = inptr0 + ldin;
            prefetch_4x(inptr0);
            prefetch_4x(inptr1);
            for (int k = k0; k < kmax; k += 8) {
                interleave_2x8_2_h(inptr0, inptr1, outptr);
                outptr += PACK_SIZE_128;
            }
        }

        for (; y < ymax; y += 8) {
            const dt_float16* inptr0 = inptr + y / PACK_C_SIZE * ldin + k0;
            prefetch_4x(inptr0);
            for (int k = k0; k < kmax; k += 8) {
                interleave_1x8_2_h(inptr0, outptr);
                outptr += PACK_SIZE_64;
            }
        }
    }

    static void hgemm_16x12_pack_B(
            dt_float16* out, const dt_float16* in, int ldin, int x0, int xmax, int k0,
            int kmax) {
        megdnn_assert(k0 % 8 == 0 && kmax % 8 == 0, "K must be time of 8");
        dt_float16 tmpbuff[96] = {static_cast<dt_float16>(0.0)};

        constexpr int PACK_C_SIZE = 8;
        int ksize = kmax - k0;
        int ksize12 = ksize * 12;
        dt_float16* outptr_base = out;

        for (int k = k0; k < kmax; k += 8) {
            const dt_float16* inptr = in + k / PACK_C_SIZE * ldin + x0 * PACK_C_SIZE;
            prefetch_3x(inptr);

            int x = x0;
            auto outptr = outptr_base;
            for (; x + 12 <= xmax; x += 12) {
                auto outptr_interleave = outptr;
                transpose_1x12_2_h(inptr, outptr_interleave);
                outptr += ksize12;
            }

            if (x < xmax) {
                std::memcpy(
                        tmpbuff, inptr, sizeof(dt_float16) * (xmax - x) * PACK_C_SIZE);
                auto outptr_interleave = outptr;
                inptr = tmpbuff;
                transpose_1x12_2_h(inptr, outptr_interleave);
            }
            outptr_base += 12 * 8;
        }
    }
};

#define M_BLOCK 1
#define N_BLOCK 1
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 2
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 3
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 4
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 5
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 6
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 7
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 8
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 9
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 10
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 11
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 12
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#undef M_BLOCK
#define M_BLOCK 2
#define N_BLOCK 1
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 2
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 3
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 4
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 5
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 6
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 7
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 8
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 9
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 10
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 11
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#define N_BLOCK 12
#include "mk8_16x12_kern.inc"
#undef N_BLOCK
#undef M_BLOCK

}  // namespace aarch64
}  // namespace megdnn
#endif