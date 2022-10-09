#include "src/aarch64/matrix_mul/fp16/kernel_mk8_16x12.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

typedef void (*kern_func)(
        const dt_float16*, const dt_float16*, int, dt_float16*, int, bool);
static kern_func kern_func_table[2][12] = {
        {matmul_mk8_16x12::kern<1, 1>, matmul_mk8_16x12::kern<1, 2>,
         matmul_mk8_16x12::kern<1, 3>, matmul_mk8_16x12::kern<1, 4>,
         matmul_mk8_16x12::kern<1, 5>, matmul_mk8_16x12::kern<1, 6>,
         matmul_mk8_16x12::kern<1, 7>, matmul_mk8_16x12::kern<1, 8>,
         matmul_mk8_16x12::kern<1, 9>, matmul_mk8_16x12::kern<1, 10>,
         matmul_mk8_16x12::kern<1, 11>, matmul_mk8_16x12::kern<1, 12>},
        {matmul_mk8_16x12::kern<2, 1>, matmul_mk8_16x12::kern<2, 2>,
         matmul_mk8_16x12::kern<2, 3>, matmul_mk8_16x12::kern<2, 4>,
         matmul_mk8_16x12::kern<2, 5>, matmul_mk8_16x12::kern<2, 6>,
         matmul_mk8_16x12::kern<2, 7>, matmul_mk8_16x12::kern<2, 8>,
         matmul_mk8_16x12::kern<2, 9>, matmul_mk8_16x12::kern<2, 10>,
         matmul_mk8_16x12::kern<2, 11>, matmul_mk8_16x12::kern<2, 12>}};

MEGDNN_REG_GEMM_STRATEGY_IMPL(hgemm_mk8_16x12);

void hgemm_mk8_16x12::pack_A(
        dt_float16* out, const dt_float16* in, int ldin, int y0, int ymax, int k0,
        int kmax, bool transpose_A) const {
    megdnn_assert(!transpose_A, "mk8 float16 matmul not support transpose A");
    matmul_mk8_16x12::hgemm_16x12_pack_A(out, in, ldin, y0, ymax, k0, kmax);
}

void hgemm_mk8_16x12::pack_B(
        dt_float16* out, const dt_float16* in, int ldin, int x0, int xmax, int k0,
        int kmax, bool transpose_B) const {
    megdnn_assert(!transpose_B, "mk8 float16 matmul not support transpose B");
    matmul_mk8_16x12::hgemm_16x12_pack_B(out, in, ldin, x0, xmax, k0, kmax);
}

// Overview of register layout:
//
// A 12x2 cell of Rhs is stored in 16bit in q2-q4.
// A 2x16 cell of Lhs is stored in 16bit in q0-q1 and q5-q6
// A 12x16 block of accumulators is stored in 16bit in q8--q31.
//
//                                             +----+----+
//                                             | v0 | v1 |
//                                        Rhs  +----+----+
//                                             | v5 | v6 |
//                                             +----+----+
//
//                                             |    |    |
//
//                 Lhs                         |    |    |
//
//  +---------------+---------------+ - - - -  +----+----+
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v8 | v20|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v9 | v21|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v10| v22|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v11| v23|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v12| v24|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v13| v25|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v14| v26|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v15| v27|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v16| v28|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v17| v29|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v18| v30|
//  |v2[0-7] v3[0-3]|v3[4-7] v4[0-7]|          | v19| v31|
//  +---------------+---------------+ - - - -  +----+----+
//
//                            Accumulator

void hgemm_mk8_16x12::kern(
        const dt_float16* packedA, const dt_float16* packedB, size_t M, size_t N,
        size_t K, dt_float16* C, size_t LDC, bool is_first_k, const dt_float16*,
        dt_float16*) const {
    megdnn_assert(
            A_dtype.enumv() == B_dtype.enumv() && A_dtype.enumv() == C_dtype.enumv() &&
            A_dtype.enumv() == DTypeEnum::Float16);

    const size_t K16 = K * 16;
    const size_t K8 = K * 8;
    const size_t K12 = K * 12;

    constexpr size_t PACK_C_SIZE = 8;
    constexpr size_t A_BLOCK = 16;
    constexpr size_t B_BLOCK = 12;

    size_t m = 0;

    for (; m < M; m += A_BLOCK) {
        dt_float16* outptr = C + (m / PACK_C_SIZE * LDC);
        const size_t m_func_idx = std::min<size_t>(M - m, A_BLOCK) / 8 - 1;
        size_t n = 0;
        const dt_float16* cur_packedB = packedB;
        for (; n < N; n += B_BLOCK) {
            const size_t n_func_idx = std::min<size_t>(N - n, B_BLOCK) - 1;
            kern_func_table[m_func_idx][n_func_idx](
                    packedA, cur_packedB, K, outptr, LDC, is_first_k);
            cur_packedB += K12;
            outptr += B_BLOCK * PACK_C_SIZE;
        }
        packedA += (m_func_idx ? K16 : K8);
    }
}

#endif