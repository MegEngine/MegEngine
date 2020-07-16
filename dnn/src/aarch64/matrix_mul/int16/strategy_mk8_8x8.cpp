/**
 * \file dnn/src/aarch64/matrix_mul/int16/strategy_mk8_8x8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/aarch64/matrix_mul/int16/strategy.h"
#include "src/aarch64/matrix_mul/asm/common.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace aarch64;
using namespace aarch64::matmul;

namespace {

void kern_8x1(const dt_int16* a_ptr, const dt_int16* b_ptr, int LDB, int K,
              dt_int32* output) {
    //! As each load 32 number from B, but the pos add 24 * 2, so we minus 24
    //! here.
    LDB *= sizeof(dt_int16);

    asm volatile(
            "subs %w[K], %w[K], #8\n"
            "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[a_ptr]], 64\n"
            "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[a_ptr]], 64\n"
            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"

            "smull v16.4s, v24.4h, v0.h[0]\n"
            "smull2 v17.4s, v24.8h, v0.h[0]\n"
            "smull v18.4s, v25.4h, v0.h[1]\n"
            "smull2 v19.4s, v25.8h, v0.h[1]\n"
            "smull v20.4s, v26.4h, v0.h[2]\n"
            "smull2 v21.4s, v26.8h, v0.h[2]\n"
            "smull v22.4s, v27.4h, v0.h[3]\n"
            "smull2 v23.4s, v27.8h, v0.h[3]\n"

            "beq 2f\n"

            "1:\n"
            "ld1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[a_ptr]], 64\n"
            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v29.4h, v0.h[5]\n"
            "smlal2 v19.4s, v29.8h, v0.h[5]\n"
            "smlal v20.4s, v30.4h, v0.h[6]\n"
            "smlal2 v21.4s, v30.8h, v0.h[6]\n"
            "smlal v22.4s, v31.4h, v0.h[7]\n"
            "smlal2 v23.4s, v31.8h, v0.h[7]\n"

            "ld1 {v0.4s}, [%[b_ptr]], %x[LDB]\n"
            "ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[a_ptr]], 64\n"

            "smlal v16.4s, v24.4h, v0.h[0]\n"
            "smlal2 v17.4s, v24.8h, v0.h[0]\n"
            "smlal v18.4s, v25.4h, v0.h[1]\n"
            "smlal2 v19.4s, v25.8h, v0.h[1]\n"
            "smlal v20.4s, v26.4h, v0.h[2]\n"
            "smlal2 v21.4s, v26.8h, v0.h[2]\n"
            "smlal v22.4s, v27.4h, v0.h[3]\n"
            "smlal2 v23.4s, v27.8h, v0.h[3]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v29.4h, v0.h[5]\n"
            "smlal2 v19.4s, v29.8h, v0.h[5]\n"
            "smlal v20.4s, v30.4h, v0.h[6]\n"
            "smlal2 v21.4s, v30.8h, v0.h[6]\n"
            "smlal v22.4s, v31.4h, v0.h[7]\n"
            "smlal2 v23.4s, v31.8h, v0.h[7]\n"

            "add v16.4s, v16.4s, v18.4s\n"
            "add v20.4s, v20.4s, v22.4s\n"
            "add v17.4s, v17.4s, v19.4s\n"
            "add v21.4s, v21.4s, v23.4s\n"
            "add v16.4s, v16.4s, v20.4s\n"
            "add v17.4s, v17.4s, v21.4s\n"

            "st1 {v16.4s, v17.4s}, [%[output]], 32\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v16", "v17", "v18", "v19", "v20", "v21",
              "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
}

// Overview of register layout:
//
// A 8x1 cell of Lhs is stored in 16bit in v24-v27
// A 8x1 cell of Rhs is stored in 16bit in v0-v7
// A 8x2 block of accumulators is stored in 32bit in v16-v23.
//
//                  Lhs +--------+
//                      |v0[0-7]|
//                      |v1[0-7]|
//                      |v2[0-7]|
//                      |v3[0-7]|
//                      +--------+
//      Rhs
//  +---------+ - - - - -+--------+
//  | v24[0-7]|          |v16[0-3]|
//  | v25[0-7]|          |v17[0-3]|
//  | v26[0-7]|          |v18[0-3]|
//  | v27[0-7]|          |v19[0-3]|
//  | v28[0-7]|          |v20[0-3]|
//  | v29[0-7]|          |v21[0-3]|
//  | v30[0-7]|          |v22[0-3]|
//  | v31[0-7]|          |v23[0-3]|
//  +---------+          +--------+
//                      Accumulator
void kern_8x4(const dt_int16* a_ptr, const dt_int16* b_ptr, int LDB, int K,
              dt_int32* output) {
    //! As each load 32 number from B, but the pos add 24 * 2, so we minus 24
    //! here.
    LDB = (LDB - 24) * sizeof(dt_int16);

    asm volatile(
            "subs %w[K], %w[K], #8\n"

            "ld1 {v24.4s}, [%[a_ptr]], 16\n"
            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "ld1 {v2.4s}, [%[b_ptr]], 16\n"
            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"

            "smull v16.4s, v24.4h, v0.h[0]\n"
            "smull2 v17.4s, v24.8h, v0.h[0]\n"
            "smull v18.4s, v24.4h, v1.h[0]\n"
            "smull2 v19.4s, v24.8h, v1.h[0]\n"

            "ld1 {v25.4s}, [%[a_ptr]], 16\n"

            "smull v20.4s, v24.4h, v2.h[0]\n"
            "smull2 v21.4s, v24.8h, v2.h[0]\n"
            "smull v22.4s, v24.4h, v3.h[0]\n"
            "smull2 v23.4s, v24.8h, v3.h[0]\n"

            "smlal v16.4s, v25.4h, v0.h[1]\n"
            "smlal2 v17.4s, v25.8h, v0.h[1]\n"
            "smlal v18.4s, v25.4h, v1.h[1]\n"
            "smlal2 v19.4s, v25.8h, v1.h[1]\n"

            "ld1 {v26.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v25.4h, v2.h[1]\n"
            "smlal2 v21.4s, v25.8h, v2.h[1]\n"
            "smlal v22.4s, v25.4h, v3.h[1]\n"
            "smlal2 v23.4s, v25.8h, v3.h[1]\n"

            "smlal v16.4s, v26.4h, v0.h[2]\n"
            "smlal2 v17.4s, v26.8h, v0.h[2]\n"
            "smlal v18.4s, v26.4h, v1.h[2]\n"
            "smlal2 v19.4s, v26.8h, v1.h[2]\n"

            "ld1 {v27.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v26.4h, v2.h[2]\n"
            "smlal2 v21.4s, v26.8h, v2.h[2]\n"
            "smlal v22.4s, v26.4h, v3.h[2]\n"
            "smlal2 v23.4s, v26.8h, v3.h[2]\n"

            "smlal v16.4s, v27.4h, v0.h[3]\n"
            "smlal2 v17.4s, v27.8h, v0.h[3]\n"
            "smlal v18.4s, v27.4h, v1.h[3]\n"
            "smlal2 v19.4s, v27.8h, v1.h[3]\n"

            "ld1 {v28.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v27.4h, v2.h[3]\n"
            "smlal2 v21.4s, v27.8h, v2.h[3]\n"
            "smlal v22.4s, v27.4h, v3.h[3]\n"
            "smlal2 v23.4s, v27.8h, v3.h[3]\n"

            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v28.4h, v1.h[4]\n"
            "smlal2 v19.4s, v28.8h, v1.h[4]\n"

            "ld1 {v29.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v28.4h, v2.h[4]\n"
            "smlal2 v21.4s, v28.8h, v2.h[4]\n"
            "smlal v22.4s, v28.4h, v3.h[4]\n"
            "smlal2 v23.4s, v28.8h, v3.h[4]\n"

            "smlal v16.4s, v29.4h, v0.h[5]\n"
            "smlal2 v17.4s, v29.8h, v0.h[5]\n"
            "smlal v18.4s, v29.4h, v1.h[5]\n"
            "smlal2 v19.4s, v29.8h, v1.h[5]\n"

            "ld1 {v30.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v29.4h, v2.h[5]\n"
            "smlal2 v21.4s, v29.8h, v2.h[5]\n"
            "smlal v22.4s, v29.4h, v3.h[5]\n"
            "smlal2 v23.4s, v29.8h, v3.h[5]\n"

            "smlal v16.4s, v30.4h, v0.h[6]\n"
            "smlal2 v17.4s, v30.8h, v0.h[6]\n"
            "smlal v18.4s, v30.4h, v1.h[6]\n"
            "smlal2 v19.4s, v30.8h, v1.h[6]\n"

            "ld1 {v31.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v30.4h, v2.h[6]\n"
            "smlal2 v21.4s, v30.8h, v2.h[6]\n"
            "smlal v22.4s, v30.4h, v3.h[6]\n"
            "smlal2 v23.4s, v30.8h, v3.h[6]\n"

            "beq 2f\n"

            "1:\n"

            "ld1 {v24.4s}, [%[a_ptr]], 16\n"

            "smlal v16.4s, v31.4h, v0.h[7]\n"
            "smlal2 v17.4s, v31.8h, v0.h[7]\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"

            "smlal v18.4s, v31.4h, v1.h[7]\n"
            "smlal2 v19.4s, v31.8h, v1.h[7]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"

            "smlal v20.4s, v31.4h, v2.h[7]\n"
            "smlal2 v21.4s, v31.8h, v2.h[7]\n"

            "ld1 {v2.4s}, [%[b_ptr]], 16\n"

            "smlal v22.4s, v31.4h, v3.h[7]\n"
            "smlal2 v23.4s, v31.8h, v3.h[7]\n"

            "ld1 {v3.4s}, [%[b_ptr]], %x[LDB]\n"

            "smlal v16.4s, v24.4h, v0.h[0]\n"
            "smlal2 v17.4s, v24.8h, v0.h[0]\n"
            "smlal v18.4s, v24.4h, v1.h[0]\n"
            "smlal2 v19.4s, v24.8h, v1.h[0]\n"

            "ld1 {v25.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v24.4h, v2.h[0]\n"
            "smlal2 v21.4s, v24.8h, v2.h[0]\n"
            "smlal v22.4s, v24.4h, v3.h[0]\n"
            "smlal2 v23.4s, v24.8h, v3.h[0]\n"

            "smlal v16.4s, v25.4h, v0.h[1]\n"
            "smlal2 v17.4s, v25.8h, v0.h[1]\n"
            "smlal v18.4s, v25.4h, v1.h[1]\n"
            "smlal2 v19.4s, v25.8h, v1.h[1]\n"

            "ld1 {v26.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v25.4h, v2.h[1]\n"
            "smlal2 v21.4s, v25.8h, v2.h[1]\n"
            "smlal v22.4s, v25.4h, v3.h[1]\n"
            "smlal2 v23.4s, v25.8h, v3.h[1]\n"

            "smlal v16.4s, v26.4h, v0.h[2]\n"
            "smlal2 v17.4s, v26.8h, v0.h[2]\n"
            "smlal v18.4s, v26.4h, v1.h[2]\n"
            "smlal2 v19.4s, v26.8h, v1.h[2]\n"

            "ld1 {v27.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v26.4h, v2.h[2]\n"
            "smlal2 v21.4s, v26.8h, v2.h[2]\n"
            "smlal v22.4s, v26.4h, v3.h[2]\n"
            "smlal2 v23.4s, v26.8h, v3.h[2]\n"

            "smlal v16.4s, v27.4h, v0.h[3]\n"
            "smlal2 v17.4s, v27.8h, v0.h[3]\n"
            "smlal v18.4s, v27.4h, v1.h[3]\n"
            "smlal2 v19.4s, v27.8h, v1.h[3]\n"

            "ld1 {v28.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v27.4h, v2.h[3]\n"
            "smlal2 v21.4s, v27.8h, v2.h[3]\n"
            "smlal v22.4s, v27.4h, v3.h[3]\n"
            "smlal2 v23.4s, v27.8h, v3.h[3]\n"

            "smlal v16.4s, v28.4h, v0.h[4]\n"
            "smlal2 v17.4s, v28.8h, v0.h[4]\n"
            "smlal v18.4s, v28.4h, v1.h[4]\n"
            "smlal2 v19.4s, v28.8h, v1.h[4]\n"

            "ld1 {v29.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v28.4h, v2.h[4]\n"
            "smlal2 v21.4s, v28.8h, v2.h[4]\n"
            "smlal v22.4s, v28.4h, v3.h[4]\n"
            "smlal2 v23.4s, v28.8h, v3.h[4]\n"

            "smlal v16.4s, v29.4h, v0.h[5]\n"
            "smlal2 v17.4s, v29.8h, v0.h[5]\n"
            "smlal v18.4s, v29.4h, v1.h[5]\n"
            "smlal2 v19.4s, v29.8h, v1.h[5]\n"

            "ld1 {v30.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v29.4h, v2.h[5]\n"
            "smlal2 v21.4s, v29.8h, v2.h[5]\n"
            "smlal v22.4s, v29.4h, v3.h[5]\n"
            "smlal2 v23.4s, v29.8h, v3.h[5]\n"

            "smlal v16.4s, v30.4h, v0.h[6]\n"
            "smlal2 v17.4s, v30.8h, v0.h[6]\n"
            "smlal v18.4s, v30.4h, v1.h[6]\n"
            "smlal2 v19.4s, v30.8h, v1.h[6]\n"

            "ld1 {v31.4s}, [%[a_ptr]], 16\n"

            "smlal v20.4s, v30.4h, v2.h[6]\n"
            "smlal2 v21.4s, v30.8h, v2.h[6]\n"
            "smlal v22.4s, v30.4h, v3.h[6]\n"
            "smlal2 v23.4s, v30.8h, v3.h[6]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"

            "smlal v16.4s, v31.4h, v0.h[7]\n"
            "smlal2 v17.4s, v31.8h, v0.h[7]\n"
            "smlal v18.4s, v31.4h, v1.h[7]\n"
            "smlal2 v19.4s, v31.8h, v1.h[7]\n"
            "smlal v20.4s, v31.4h, v2.h[7]\n"
            "smlal2 v21.4s, v31.8h, v2.h[7]\n"
            "smlal v22.4s, v31.4h, v3.h[7]\n"
            "smlal2 v23.4s, v31.8h, v3.h[7]\n"

            "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output]], 64\n"
            "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
}

// Overview of register layout:
//
// A 8x1 cell of Rhs is stored in 16bit in v8-v15
// A 8x1 cell of Lhs is stored in 16bit in v0-v7
// A 8x2 block of accumulators is stored in 32bit in v16-v31.
//
//                  Rhs +--------+
//                      | v8[0-7]|
//                      | v9[0-7]|
//                      |v10[0-7]|
//                      |v11[0-7]|
//                      |v12[0-7]|
//                      |v13[0-7]|
//                      |v14[0-7]|
//                      |v15[0-7]|
//                      +--------+
//      Lhs
//  +--------+ - - - - -+--------+--------+
//  | v0[0-7]|          |v16[0-3]|v17[0-3]|
//  | v1[0-7]|          |v18[0-3]|v19[0-3]|
//  | v2[0-7]|          |v20[0-3]|v21[0-3]|
//  | v3[0-7]|          |v22[0-3]|v23[0-3]|
//  | v4[0-7]|          |v24[0-3]|v25[0-3]|
//  | v5[0-7]|          |v26[0-3]|v27[0-3]|
//  | v6[0-7]|          |v28[0-3]|v29[0-3]|
//  | v7[0-7]|          |v30[0-3]|v31[0-3]|
//  +--------+          +--------+--------+
//                      Accumulator
void kern_8x8(const dt_int16* a_ptr, const dt_int16* b_ptr, int LDB, int K,
              dt_int32* output) {
    //! As each load 64 number from B, but the pos add 48 * 2, so we minus 48
    //! here.
    LDB = (LDB - 48) * sizeof(dt_int16);

    asm volatile(
            "subs %w[K], %w[K], #8\n"
            "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[a_ptr]], 64\n"
            "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[a_ptr]], 64\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "smull v16.4s, v8.4h, v0.h[0]\n"
            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "smlal v16.4s, v9.4h, v0.h[1]\n"
            "smull v18.4s, v8.4h, v1.h[0]\n"
            "smull2 v17.4s, v8.8h, v0.h[0]\n"
            "smull2 v19.4s, v8.8h, v1.h[0]\n"
            "smlal v16.4s, v10.4h, v0.h[2]\n"
            "smlal v18.4s, v9.4h, v1.h[1]\n"
            "smlal2 v17.4s, v9.8h, v0.h[1]\n"
            "smlal2 v19.4s, v9.8h, v1.h[1]\n"
            "smlal v16.4s, v11.4h, v0.h[3]\n"
            "smlal v18.4s, v10.4h, v1.h[2]\n"
            "smlal2 v17.4s, v10.8h, v0.h[2]\n"
            "smlal2 v19.4s, v10.8h, v1.h[2]\n"
            "smlal v16.4s, v12.4h, v0.h[4]\n"
            "smlal v18.4s, v11.4h, v1.h[3]\n"
            "smlal2 v17.4s, v11.8h, v0.h[3]\n"
            "smlal2 v19.4s, v11.8h, v1.h[3]\n"
            "smlal v16.4s, v13.4h, v0.h[5]\n"
            "smlal v18.4s, v12.4h, v1.h[4]\n"
            "smlal2 v17.4s, v12.8h, v0.h[4]\n"
            "smlal2 v19.4s, v12.8h, v1.h[4]\n"
            "smlal2 v17.4s, v13.8h, v0.h[5]\n"

            "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
            "smlal v16.4s, v14.4h, v0.h[6]\n"
            "smlal v18.4s, v13.4h, v1.h[5]\n"
            "smlal2 v17.4s, v14.8h, v0.h[6]\n"
            "smlal2 v19.4s, v13.8h, v1.h[5]\n"
            "smull v20.4s, v8.4h, v2.h[0]\n"
            "smull v22.4s, v8.4h, v3.h[0]\n"
            "smull2 v21.4s, v8.8h, v2.h[0]\n"
            "smull2 v23.4s, v8.8h, v3.h[0]\n"
            "smlal v16.4s, v15.4h, v0.h[7]\n"
            "smlal v18.4s, v14.4h, v1.h[6]\n"
            "smlal2 v17.4s, v15.8h, v0.h[7]\n"
            "smlal2 v19.4s, v14.8h, v1.h[6]\n"
            "smlal v20.4s, v9.4h, v2.h[1]\n"
            "smlal v22.4s, v9.4h, v3.h[1]\n"
            "smlal2 v21.4s, v9.8h, v2.h[1]\n"
            "smlal2 v23.4s, v9.8h, v3.h[1]\n"
            "smlal v18.4s, v15.4h, v1.h[7]\n"
            "smlal v20.4s, v10.4h, v2.h[2]\n"
            "smlal v22.4s, v10.4h, v3.h[2]\n"
            "smlal2 v21.4s, v10.8h, v2.h[2]\n"
            "smlal2 v23.4s, v10.8h, v3.h[2]\n"
            "smlal2 v19.4s, v15.8h, v1.h[7]\n"
            "smlal v20.4s, v11.4h, v2.h[3]\n"
            "smlal v22.4s, v11.4h, v3.h[3]\n"
            "smlal2 v21.4s, v11.8h, v2.h[3]\n"
            "smlal2 v23.4s, v11.8h, v3.h[3]\n"
            "smlal v20.4s, v12.4h, v2.h[4]\n"
            "smlal v22.4s, v12.4h, v3.h[4]\n"
            "smlal2 v21.4s, v12.8h, v2.h[4]\n"
            "smlal2 v23.4s, v12.8h, v3.h[4]\n"
            "smlal v20.4s, v13.4h, v2.h[5]\n"
            "smlal v22.4s, v13.4h, v3.h[5]\n"
            "smlal2 v21.4s, v13.8h, v2.h[5]\n"
            "smlal2 v23.4s, v13.8h, v3.h[5]\n"

            "ld1 {v4.4s, v5.4s}, [%[b_ptr]], 32\n"
            "smlal v20.4s, v14.4h, v2.h[6]\n"
            "smlal v22.4s, v14.4h, v3.h[6]\n"
            "smlal2 v21.4s, v14.8h, v2.h[6]\n"
            "smlal2 v23.4s, v14.8h, v3.h[6]\n"
            "smull v24.4s, v8.4h, v4.h[0]\n"
            "smull v26.4s, v8.4h, v5.h[0]\n"
            "smull2 v25.4s, v8.8h, v4.h[0]\n"
            "smull2 v27.4s, v8.8h, v5.h[0]\n"
            "smlal v20.4s, v15.4h, v2.h[7]\n"
            "smlal v22.4s, v15.4h, v3.h[7]\n"
            "smlal2 v21.4s, v15.8h, v2.h[7]\n"
            "smlal2 v23.4s, v15.8h, v3.h[7]\n"
            "smlal v24.4s, v9.4h, v4.h[1]\n"
            "smlal v26.4s, v9.4h, v5.h[1]\n"
            "smlal2 v25.4s, v9.8h, v4.h[1]\n"
            "smlal2 v27.4s, v9.8h, v5.h[1]\n"
            "smlal v24.4s, v10.4h, v4.h[2]\n"
            "smlal v26.4s, v10.4h, v5.h[2]\n"
            "smlal2 v25.4s, v10.8h, v4.h[2]\n"
            "smlal2 v27.4s, v10.8h, v5.h[2]\n"
            "smlal v24.4s, v11.4h, v4.h[3]\n"
            "smlal v26.4s, v11.4h, v5.h[3]\n"
            "smlal2 v25.4s, v11.8h, v4.h[3]\n"
            "smlal2 v27.4s, v11.8h, v5.h[3]\n"
            "smlal v24.4s, v12.4h, v4.h[4]\n"
            "smlal v26.4s, v12.4h, v5.h[4]\n"
            "smlal2 v25.4s, v12.8h, v4.h[4]\n"
            "smlal2 v27.4s, v12.8h, v5.h[4]\n"
            "smlal v24.4s, v13.4h, v4.h[5]\n"
            "smlal v26.4s, v13.4h, v5.h[5]\n"
            "smlal2 v25.4s, v13.8h, v4.h[5]\n"
            "smlal2 v27.4s, v13.8h, v5.h[5]\n"

            "ld1 {v6.4s, v7.4s}, [%[b_ptr]], %x[LDB]\n"
            "smlal v24.4s, v14.4h, v4.h[6]\n"
            "smlal v26.4s, v14.4h, v5.h[6]\n"
            "smlal2 v25.4s, v14.8h, v4.h[6]\n"
            "smlal2 v27.4s, v14.8h, v5.h[6]\n"
            "smull v28.4s, v8.4h, v6.h[0]\n"
            "smull v30.4s, v8.4h, v7.h[0]\n"
            "smull2 v29.4s, v8.8h, v6.h[0]\n"
            "smull2 v31.4s, v8.8h, v7.h[0]\n"
            "smlal v28.4s, v9.4h, v6.h[1]\n"
            "smlal v30.4s, v9.4h, v7.h[1]\n"
            "smlal2 v29.4s, v9.8h, v6.h[1]\n"
            "smlal2 v31.4s, v9.8h, v7.h[1]\n"
            "smlal v28.4s, v10.4h, v6.h[2]\n"
            "smlal v30.4s, v10.4h, v7.h[2]\n"
            "smlal2 v29.4s, v10.8h, v6.h[2]\n"
            "smlal2 v31.4s, v10.8h, v7.h[2]\n"
            "smlal v28.4s, v11.4h, v6.h[3]\n"
            "smlal v30.4s, v11.4h, v7.h[3]\n"
            "smlal2 v29.4s, v11.8h, v6.h[3]\n"
            "smlal2 v31.4s, v11.8h, v7.h[3]\n"
            "smlal v28.4s, v12.4h, v6.h[4]\n"
            "smlal v30.4s, v12.4h, v7.h[4]\n"
            "smlal2 v29.4s, v12.8h, v6.h[4]\n"
            "smlal2 v31.4s, v12.8h, v7.h[4]\n"
            "smlal v28.4s, v13.4h, v6.h[5]\n"
            "smlal v30.4s, v13.4h, v7.h[5]\n"
            "smlal2 v29.4s, v13.8h, v6.h[5]\n"
            "smlal2 v31.4s, v13.8h, v7.h[5]\n"

            "beq 2f\n"

            "1:\n"

            "smlal v24.4s, v15.4h, v4.h[7]\n"
            "smlal v26.4s, v15.4h, v5.h[7]\n"
            "smlal2 v25.4s, v15.8h, v4.h[7]\n"

            "ld1 {v8.4s, v9.4s}, [%[a_ptr]], 32\n"
            "smlal2 v27.4s, v15.8h, v5.h[7]\n"
            "smlal v28.4s, v14.4h, v6.h[6]\n"
            "smlal v30.4s, v14.4h, v7.h[6]\n"

            "ld1 {v10.4s, v11.4s}, [%[a_ptr]], 32\n"
            "smlal2 v29.4s, v15.8h, v6.h[7]\n"
            "smlal2 v31.4s, v14.8h, v7.h[6]\n"
            "smlal v28.4s, v15.4h, v6.h[7]\n"

            "ld1 {v12.4s, v13.4s}, [%[a_ptr]], 32\n"
            "smlal v30.4s, v15.4h, v7.h[7]\n"
            "smlal2 v29.4s, v14.8h, v6.h[6]\n"

            "ld1 {v0.4s}, [%[b_ptr]], 16\n"
            "smlal2 v31.4s, v15.8h, v7.h[7]\n"
            "smlal v16.4s, v8.4h, v0.h[0]\n"

            "ld1 {v1.4s}, [%[b_ptr]], 16\n"
            "smlal v16.4s, v9.4h, v0.h[1]\n"
            "smlal2 v17.4s, v8.8h, v0.h[0]\n"
            "smlal v16.4s, v10.4h, v0.h[2]\n"
            "smlal v18.4s, v8.4h, v1.h[0]\n"
            "smlal2 v17.4s, v9.8h, v0.h[1]\n"
            "smlal2 v19.4s, v8.8h, v1.h[0]\n"

            "ld1 {v14.4s, v15.4s}, [%[a_ptr]], 32\n"
            "smlal v16.4s, v11.4h, v0.h[3]\n"
            "smlal v18.4s, v9.4h, v1.h[1]\n"
            "smlal2 v17.4s, v10.8h, v0.h[2]\n"
            "smlal2 v19.4s, v9.8h, v1.h[1]\n"
            "smlal v16.4s, v12.4h, v0.h[4]\n"
            "smlal v18.4s, v10.4h, v1.h[2]\n"
            "smlal2 v17.4s, v11.8h, v0.h[3]\n"
            "smlal2 v19.4s, v10.8h, v1.h[2]\n"
            "smlal v16.4s, v13.4h, v0.h[5]\n"
            "smlal v18.4s, v11.4h, v1.h[3]\n"
            "smlal2 v17.4s, v12.8h, v0.h[4]\n"
            "smlal2 v19.4s, v11.8h, v1.h[3]\n"
            "smlal v16.4s, v14.4h, v0.h[6]\n"
            "smlal v18.4s, v12.4h, v1.h[4]\n"
            "smlal2 v17.4s, v13.8h, v0.h[5]\n"
            "smlal2 v19.4s, v12.8h, v1.h[4]\n"
            "smlal v16.4s, v15.4h, v0.h[7]\n"
            "smlal v18.4s, v13.4h, v1.h[5]\n"
            "smlal2 v17.4s, v14.8h, v0.h[6]\n"
            "smlal2 v19.4s, v13.8h, v1.h[5]\n"

            "ld1 {v2.4s, v3.4s}, [%[b_ptr]], 32\n"
            "smlal v18.4s, v14.4h, v1.h[6]\n"
            "smlal2 v17.4s, v15.8h, v0.h[7]\n"
            "smlal2 v19.4s, v14.8h, v1.h[6]\n"
            "smlal v20.4s, v8.4h, v2.h[0]\n"
            "smlal v22.4s, v8.4h, v3.h[0]\n"
            "smlal2 v21.4s, v8.8h, v2.h[0]\n"
            "smlal2 v23.4s, v8.8h, v3.h[0]\n"
            "smlal v18.4s, v15.4h, v1.h[7]\n"
            "smlal v20.4s, v9.4h, v2.h[1]\n"
            "smlal v22.4s, v9.4h, v3.h[1]\n"
            "smlal2 v21.4s, v9.8h, v2.h[1]\n"
            "smlal2 v23.4s, v9.8h, v3.h[1]\n"
            "smlal2 v19.4s, v15.8h, v1.h[7]\n"
            "smlal v20.4s, v10.4h, v2.h[2]\n"
            "smlal v22.4s, v10.4h, v3.h[2]\n"
            "smlal2 v21.4s, v10.8h, v2.h[2]\n"
            "smlal2 v23.4s, v10.8h, v3.h[2]\n"
            "smlal v20.4s, v11.4h, v2.h[3]\n"
            "smlal v22.4s, v11.4h, v3.h[3]\n"
            "smlal2 v21.4s, v11.8h, v2.h[3]\n"
            "smlal2 v23.4s, v11.8h, v3.h[3]\n"
            "smlal v20.4s, v12.4h, v2.h[4]\n"
            "smlal v22.4s, v12.4h, v3.h[4]\n"
            "smlal2 v21.4s, v12.8h, v2.h[4]\n"
            "smlal2 v23.4s, v12.8h, v3.h[4]\n"
            "smlal v20.4s, v13.4h, v2.h[5]\n"
            "smlal v22.4s, v13.4h, v3.h[5]\n"
            "smlal2 v21.4s, v13.8h, v2.h[5]\n"
            "smlal2 v23.4s, v13.8h, v3.h[5]\n"

            "ld1 {v4.4s, v5.4s}, [%[b_ptr]], 32\n"
            "smlal v20.4s, v14.4h, v2.h[6]\n"
            "smlal v22.4s, v14.4h, v3.h[6]\n"
            "smlal2 v21.4s, v14.8h, v2.h[6]\n"
            "smlal2 v23.4s, v14.8h, v3.h[6]\n"
            "smlal v24.4s, v8.4h, v4.h[0]\n"
            "smlal v26.4s, v8.4h, v5.h[0]\n"
            "smlal2 v25.4s, v8.8h, v4.h[0]\n"
            "smlal2 v27.4s, v8.8h, v5.h[0]\n"
            "smlal v20.4s, v15.4h, v2.h[7]\n"
            "smlal2 v21.4s, v15.8h, v2.h[7]\n"
            "smlal v22.4s, v15.4h, v3.h[7]\n"
            "smlal2 v23.4s, v15.8h, v3.h[7]\n"
            "smlal v24.4s, v9.4h, v4.h[1]\n"
            "smlal v26.4s, v9.4h, v5.h[1]\n"
            "smlal2 v25.4s, v9.8h, v4.h[1]\n"
            "smlal2 v27.4s, v9.8h, v5.h[1]\n"
            "smlal v24.4s, v10.4h, v4.h[2]\n"
            "smlal v26.4s, v10.4h, v5.h[2]\n"
            "smlal2 v25.4s, v10.8h, v4.h[2]\n"
            "smlal2 v27.4s, v10.8h, v5.h[2]\n"
            "smlal v24.4s, v11.4h, v4.h[3]\n"
            "smlal v26.4s, v11.4h, v5.h[3]\n"
            "smlal2 v25.4s, v11.8h, v4.h[3]\n"
            "smlal2 v27.4s, v11.8h, v5.h[3]\n"
            "smlal v24.4s, v12.4h, v4.h[4]\n"
            "smlal v26.4s, v12.4h, v5.h[4]\n"
            "smlal2 v25.4s, v12.8h, v4.h[4]\n"
            "smlal2 v27.4s, v12.8h, v5.h[4]\n"
            "smlal v24.4s, v13.4h, v4.h[5]\n"
            "smlal v26.4s, v13.4h, v5.h[5]\n"
            "smlal2 v25.4s, v13.8h, v4.h[5]\n"
            "smlal2 v27.4s, v13.8h, v5.h[5]\n"

            "ld1 {v6.4s, v7.4s}, [%[b_ptr]], %x[LDB]\n"
            "smlal v24.4s, v14.4h, v4.h[6]\n"
            "smlal v26.4s, v14.4h, v5.h[6]\n"
            "smlal2 v25.4s, v14.8h, v4.h[6]\n"
            "smlal2 v27.4s, v14.8h, v5.h[6]\n"
            "smlal v28.4s, v8.4h, v6.h[0]\n"
            "smlal v30.4s, v8.4h, v7.h[0]\n"
            "smlal2 v29.4s, v8.8h, v6.h[0]\n"
            "smlal2 v31.4s, v8.8h, v7.h[0]\n"
            "smlal v28.4s, v9.4h, v6.h[1]\n"
            "smlal v30.4s, v9.4h, v7.h[1]\n"
            "smlal2 v29.4s, v9.8h, v6.h[1]\n"
            "smlal2 v31.4s, v9.8h, v7.h[1]\n"
            "smlal v28.4s, v10.4h, v6.h[2]\n"
            "smlal v30.4s, v10.4h, v7.h[2]\n"
            "smlal2 v29.4s, v10.8h, v6.h[2]\n"
            "smlal2 v31.4s, v10.8h, v7.h[2]\n"
            "smlal v28.4s, v11.4h, v6.h[3]\n"
            "smlal v30.4s, v11.4h, v7.h[3]\n"
            "smlal2 v29.4s, v11.8h, v6.h[3]\n"
            "smlal2 v31.4s, v11.8h, v7.h[3]\n"
            "smlal v28.4s, v12.4h, v6.h[4]\n"
            "smlal v30.4s, v12.4h, v7.h[4]\n"
            "smlal2 v29.4s, v12.8h, v6.h[4]\n"
            "smlal2 v31.4s, v12.8h, v7.h[4]\n"
            "smlal v28.4s, v13.4h, v6.h[5]\n"
            "smlal v30.4s, v13.4h, v7.h[5]\n"
            "smlal2 v29.4s, v13.8h, v6.h[5]\n"
            "smlal2 v31.4s, v13.8h, v7.h[5]\n"

            "subs %w[K], %w[K], #8\n"
            "bne 1b\n"

            "2:\n"
            "st1 {v16.4s, v17.4s, v18.4s, v19.4s}, [%[output]], 64\n"
            "smlal v24.4s, v15.4h, v4.h[7]\n"
            "smlal v28.4s, v14.4h, v6.h[6]\n"
            "smlal v30.4s, v14.4h, v7.h[6]\n"
            "smlal v26.4s, v15.4h, v5.h[7]\n"
            "smlal2 v25.4s, v15.8h, v4.h[7]\n"
            "smlal2 v27.4s, v15.8h, v5.h[7]\n"
            "smlal2 v29.4s, v14.8h, v6.h[6]\n"
            "smlal2 v31.4s, v14.8h, v7.h[6]\n"
            "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [%[output]], 64\n"
            "smlal v28.4s, v15.4h, v6.h[7]\n"
            "smlal v30.4s, v15.4h, v7.h[7]\n"
            "smlal2 v29.4s, v15.8h, v6.h[7]\n"
            "smlal2 v31.4s, v15.8h, v7.h[7]\n"
            "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [%[output]], 64\n"
            "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [%[output]], 64\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [K] "+r"(K),
              [output] "+r"(output), [LDB] "+r"(LDB)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
              "v29", "v30", "v31", "cc", "memory");
}

}  // anonymous namespace

MEGDNN_REG_GEMM_STRATEGY_IMPL_NOPACK(gemm_nopack_s16_8x8);

void gemm_nopack_s16_8x8::kern(const dt_int16* A, size_t LDA, const dt_int16* B,
                                size_t LDB, dt_int32* C, size_t LDC, size_t M,
                                size_t K, size_t N, const dt_int32*, void*,
                                bool trA, bool trB) const {
    constexpr static size_t MB = 8;
    constexpr static size_t KB = 8;
    constexpr static size_t NB = 8;
    constexpr static size_t CALCBLK = 4;

    megdnn_assert(!trA && !trB && M % MB == 0 && K % KB == 0);

    //! (m/8, k/8, 8, 8) * (k/8, n, 8) = (m/8, n, 8)
    for (size_t m = 0; m < M; m += MB) {
        dt_int32* output = C + (m / MB) * LDC;
        const dt_int16* cur_B = B;
        size_t n = 0;
        for (; n + NB - 1 < N; n += NB) {
            kern_8x8(A, cur_B, LDB, K, output);
            cur_B += KB * NB;
            output += MB * NB;
        }
        if (N - n >= 4) {
            kern_8x4(A, cur_B, LDB, K, output);
            cur_B += KB * CALCBLK;
            output += MB * CALCBLK;
            n += 4;
        }
        while (n < N) {
            kern_8x1(A, cur_B, LDB, K, output);
            cur_B += KB;
            output += MB;
            n++;
        }
        A += LDA;
    }
}

// vim: syntax=cpp.doxygen
