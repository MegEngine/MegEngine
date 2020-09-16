/**
 * \file dnn/src/aarch64/matrix_mul/asm/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <cmath>
#include <cstdint>
#include <type_traits>
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace aarch64 {

/* ======================== Prefetch ======================== */
#define ASM_PREFETCH(address) "PRFM PLDL1KEEP, " address "\n"
#define ASM_PREFETCHL2(address) "PRFM PLDL2KEEP, " address "\n"
#define ASM_PREFETCHW(address) "PRFM PSTL1KEEP, " address "\n"
#define ASM_PREFETCHWL2(address) "PRFM PSTL2KEEP, " address "\n"

static inline void prefetch_6x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 ASM_PREFETCH("[%[pfp], #256]")
                 ASM_PREFETCH("[%[pfp], #320]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_5x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 ASM_PREFETCH("[%[pfp], #256]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_4x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 ASM_PREFETCH("[%[pfp], #192]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_3x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 ASM_PREFETCH("[%[pfp], #128]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_2x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]")
                 ASM_PREFETCH("[%[pfp], #64]")
                 :
                 : [pfp] "r"(pfp)
                 : "memory");
    // clang-format on
}

static inline void prefetch_1x(const void* pfp) {
    // clang-format off
    asm volatile(ASM_PREFETCH("[%[pfp]]") : : [pfp] "r"(pfp) : "memory");
    // clang-format on
}

/* ======================== interleave pack A ======================== */

/**
 * interleave_INTERLEAVE_UNROLLK_BATCH_type
 *
 * BATCH means process BATCH * UNROLL_K cols once, BATCH * sizeof(TYPE) *
 * UNROLL_K = 16bytes(128bits, a vector size).
 *
 * the elements traverse order:
 * rep(j, 0, INTERLEAVE) rep(i, 0, UNROLL_K) *ouptr++ = inptr[j, i]
 */

template <typename T>
static inline void interleave_24x1_8_h_helper(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        const T*& inptr4, const T*& inptr5, const T*& inptr6, const T*& inptr7,
        T*& outptr, int skippf = 0) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            // Load up 8 elements (1 vector) from each of 8 sources.
            "cbnz    %w[skippf], 1f\n"
            ASM_PREFETCH("[%[inptr0], #128]")
            ASM_PREFETCH("[%[inptr1], #128]")
            ASM_PREFETCH("[%[inptr2], #128]")
            ASM_PREFETCH("[%[inptr3], #128]")
            "1:\n"

            "ldr    q0, [%[inptr0]], #16\n" // q0=A0A1A2A3A4A5A6A7
            "ldr    q4, [%[inptr4]], #16\n" // q8=E0E1E2E3E4E5E6E7
            "ldr    q2, [%[inptr2]], #16\n" // q4=C0C1C2C3...
            "ldr    q6, [%[inptr6]], #16\n"
            "zip1    v8.8h, v0.8h, v4.8h\n"  // q8=A0E0A1E1A2E2A3E3
            "zip2    v16.8h, v0.8h, v4.8h\n" // q16=A4E4A5E5A6E6A7E7
            "zip1    v9.8h, v2.8h, v6.8h\n"  // q9=C0G0C1G1C2G2C3G3
            "zip2    v17.8h, v2.8h, v6.8h\n" // q17=C4G4C5G5C6G6C7G7
            "ldr    q1, [%[inptr1]], #16\n"  // q1=B0B1B2B3B4B5B6B7
            "ldr    q5, [%[inptr5]], #16\n"
            "ldr    q3, [%[inptr3]], #16\n" // q3=D0D1D2D3....
            "ldr    q7, [%[inptr7]], #16\n"
            "zip1    v10.8h, v1.8h, v5.8h\n" // q18=B0F0B1F1B2F2B3F3
            "zip2    v18.8h, v1.8h, v5.8h\n" // q18=B4F4B5F5B6F6B7F7
            "zip1    v11.8h, v3.8h, v7.8h\n" // q19=D0H0D1H1D2H2D3H3
            "zip2    v19.8h, v3.8h, v7.8h\n" // q19=D4H4D5H5D6H6D7H7

            "zip1    v12.8h,  v8.8h,  v9.8h\n" // q20=A0C0E0G0A1C1E1G1
            "zip2    v20.8h,  v8.8h,  v9.8h\n"
            "zip1    v13.8h, v10.8h, v11.8h\n" // q21=B0D0F0H0B1I1F1H1
            "zip2    v21.8h, v10.8h, v11.8h\n"

            "cbnz    %w[skippf], 2f\n"
            ASM_PREFETCH("[%[inptr4], #112]")
            ASM_PREFETCH("[%[inptr5], #112]")
            ASM_PREFETCH("[%[inptr6], #112]")
            ASM_PREFETCH("[%[inptr7], #112]")
            "2:\n"

            "zip1    v22.8h, v16.8h, v17.8h\n"
            "zip2    v30.8h, v16.8h, v17.8h\n"
            "zip1    v23.8h, v18.8h, v19.8h\n"
            "zip2    v31.8h, v18.8h, v19.8h\n"

            "zip1    v14.8h, v12.8h, v13.8h\n"    // q22=A0B0C0D0E0F0G0H0
            "zip2    v15.8h, v12.8h, v13.8h\n"    // q23=A1B1C1D1E1F1G1H1
            "str    q14, [%[outptr]], #48\n"
            "str    q15, [%[outptr]], #48\n"

            "zip1    v0.8h, v20.8h, v21.8h\n"
            "zip2    v1.8h, v20.8h, v21.8h\n"
            "str    q0, [%[outptr]], #48\n"
            "str    q1, [%[outptr]], #48\n"

            "zip1    v2.8h, v22.8h, v23.8h\n"
            "zip2    v3.8h, v22.8h, v23.8h\n"
            "str    q2, [%[outptr]], #48\n"
            "str    q3, [%[outptr]], #48\n"

            "zip1    v4.8h, v30.8h, v31.8h\n"
            "zip2    v5.8h, v30.8h, v31.8h\n"
            "str    q4, [%[outptr]], #48\n"
            "str    q5, [%[outptr]], #48\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
              [outptr] "+r"(outptr)
            : [skippf] "r"(skippf)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
              "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
              "v31", "cc", "memory");
}

template <typename T>
static inline void interleave_16x1_8_h_helper(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        const T*& inptr4, const T*& inptr5, const T*& inptr6, const T*& inptr7,
        T*& outptr, int skippf = 0) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            // Load up 8 elements (1 vector) from each of 8 sources.
            "cbnz    %w[skippf], 1f\n"
            ASM_PREFETCH("[%[inptr0], #128]")
            ASM_PREFETCH("[%[inptr1], #128]")
            ASM_PREFETCH("[%[inptr2], #128]")
            ASM_PREFETCH("[%[inptr3], #128]")
            "1:\n"

            "ldr    q0, [%[inptr0]], #16\n" // q0=A0A1A2A3A4A5A6A7
            "ldr    q4, [%[inptr4]], #16\n" // q8=E0E1E2E3E4E5E6E7
            "ldr    q2, [%[inptr2]], #16\n" // q4=C0C1C2C3...
            "ldr    q6, [%[inptr6]], #16\n"
            "zip1    v8.8h, v0.8h, v4.8h\n"  // q8=A0E0A1E1A2E2A3E3
            "zip2    v16.8h, v0.8h, v4.8h\n" // q16=A4E4A5E5A6E6A7E7
            "zip1    v9.8h, v2.8h, v6.8h\n"  // q9=C0G0C1G1C2G2C3G3
            "zip2    v17.8h, v2.8h, v6.8h\n" // q17=C4G4C5G5C6G6C7G7
            "ldr    q1, [%[inptr1]], #16\n"  // q1=B0B1B2B3B4B5B6B7
            "ldr    q5, [%[inptr5]], #16\n"
            "ldr    q3, [%[inptr3]], #16\n" // q3=D0D1D2D3....
            "ldr    q7, [%[inptr7]], #16\n"
            "zip1    v10.8h, v1.8h, v5.8h\n" // q18=B0F0B1F1B2F2B3F3
            "zip2    v18.8h, v1.8h, v5.8h\n" // q18=B4F4B5F5B6F6B7F7
            "zip1    v11.8h, v3.8h, v7.8h\n" // q19=D0H0D1H1D2H2D3H3
            "zip2    v19.8h, v3.8h, v7.8h\n" // q19=D4H4D5H5D6H6D7H7

            "zip1    v12.8h,  v8.8h,  v9.8h\n" // q20=A0C0E0G0A1C1E1G1
            "zip2    v20.8h,  v8.8h,  v9.8h\n"
            "zip1    v13.8h, v10.8h, v11.8h\n" // q21=B0D0F0H0B1I1F1H1
            "zip2    v21.8h, v10.8h, v11.8h\n"

            "cbnz    %w[skippf], 2f\n"
            ASM_PREFETCH("[%[inptr4], #112]")
            ASM_PREFETCH("[%[inptr5], #112]")
            ASM_PREFETCH("[%[inptr6], #112]")
            ASM_PREFETCH("[%[inptr7], #112]")
            "2:\n"

            "zip1    v22.8h, v16.8h, v17.8h\n"
            "zip2    v30.8h, v16.8h, v17.8h\n"
            "zip1    v23.8h, v18.8h, v19.8h\n"
            "zip2    v31.8h, v18.8h, v19.8h\n"

            "zip1    v14.8h, v12.8h, v13.8h\n"    // q22=A0B0C0D0E0F0G0H0
            "zip2    v15.8h, v12.8h, v13.8h\n"    // q23=A1B1C1D1E1F1G1H1
            "str    q14, [%[outptr]], #32\n"
            "str    q15, [%[outptr]], #32\n"

            "zip1    v0.8h, v20.8h, v21.8h\n"
            "zip2    v1.8h, v20.8h, v21.8h\n"
            "str    q0, [%[outptr]], #32\n"
            "str    q1, [%[outptr]], #32\n"

            "zip1    v2.8h, v22.8h, v23.8h\n"
            "zip2    v3.8h, v22.8h, v23.8h\n"
            "str    q2, [%[outptr]], #32\n"
            "str    q3, [%[outptr]], #32\n"

            "zip1    v4.8h, v30.8h, v31.8h\n"
            "zip2    v5.8h, v30.8h, v31.8h\n"
            "str    q4, [%[outptr]], #32\n"
            "str    q5, [%[outptr]], #32\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
              [outptr] "+r"(outptr)
            : [skippf] "r"(skippf)
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
              "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
              "v31", "cc", "memory");
}

template <typename T>
static inline void interleave_8x1_8_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr, int skippf = 0) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
       // Load up 8 elements (1 vector) from each of 8 sources.
       "cbnz    %w[skippf], 1f\n"
       ASM_PREFETCH("[%[inptr0], #128]")
       ASM_PREFETCH("[%[inptr1], #128]")
       ASM_PREFETCH("[%[inptr2], #128]")
       ASM_PREFETCH("[%[inptr3], #128]")
       "1:\n"


       "ldr    q0, [%[inptr0]], #16\n" // q0=A0A1A2A3A4A5A6A7
       "ldr    q4, [%[inptr4]], #16\n" // q8=E0E1E2E3E4E5E6E7
       "ldr    q2, [%[inptr2]], #16\n" // q4=C0C1C2C3...
       "ldr    q6, [%[inptr6]], #16\n"
       "zip1    v8.8h, v0.8h, v4.8h\n"  // q8=A0E0A1E1A2E2A3E3
       "zip2    v16.8h, v0.8h, v4.8h\n" // q16=A4E4A5E5A6E6A7E7
       "zip1    v9.8h, v2.8h, v6.8h\n"  // q9=C0G0C1G1C2G2C3G3
       "zip2    v17.8h, v2.8h, v6.8h\n" // q17=C4G4C5G5C6G6C7G7
       "ldr    q1, [%[inptr1]], #16\n"  // q1=B0B1B2B3B4B5B6B7
       "ldr    q5, [%[inptr5]], #16\n"
       "ldr    q3, [%[inptr3]], #16\n" // q3=D0D1D2D3....
       "ldr    q7, [%[inptr7]], #16\n"
       "zip1    v10.8h, v1.8h, v5.8h\n" // q18=B0F0B1F1B2F2B3F3
       "zip2    v18.8h, v1.8h, v5.8h\n" // q18=B4F4B5F5B6F6B7F7
       "zip1    v11.8h, v3.8h, v7.8h\n" // q19=D0H0D1H1D2H2D3H3
       "zip2    v19.8h, v3.8h, v7.8h\n" // q19=D4H4D5H5D6H6D7H7

       "zip1    v12.8h,  v8.8h,  v9.8h\n" // q20=A0C0E0G0A1C1E1G1
       "zip2    v20.8h,  v8.8h,  v9.8h\n"
       "zip1    v13.8h, v10.8h, v11.8h\n" // q21=B0D0F0H0B1I1F1H1
       "zip2    v21.8h, v10.8h, v11.8h\n"

       "cbnz    %w[skippf], 2f\n"
       ASM_PREFETCH("[%[inptr4], #112]")
       ASM_PREFETCH("[%[inptr5], #112]")
       ASM_PREFETCH("[%[inptr6], #112]")
       ASM_PREFETCH("[%[inptr7], #112]")
       "2:\n"

       "zip1    v22.8h, v16.8h, v17.8h\n"
       "zip2    v30.8h, v16.8h, v17.8h\n"
       "zip1    v23.8h, v18.8h, v19.8h\n"
       "zip2    v31.8h, v18.8h, v19.8h\n"

       "zip1    v14.8h, v12.8h, v13.8h\n"    // q22=A0B0C0D0E0F0G0H0
       "zip2    v15.8h, v12.8h, v13.8h\n"    // q23=A1B1C1D1E1F1G1H1
       "stp    q14, q15, [%[outptr]], #32\n" // Write back first two elements

       "zip1    v0.8h, v20.8h, v21.8h\n"
       "zip2    v1.8h, v20.8h, v21.8h\n"
       "stp    q0, q1, [%[outptr]], #32\n" // Write back next two elements

       "zip1    v2.8h, v22.8h, v23.8h\n"
       "zip2    v3.8h, v22.8h, v23.8h\n"
       "stp    q2, q3, [%[outptr]], #32\n" // Write back next two elements

       "zip1    v4.8h, v30.8h, v31.8h\n"
       "zip2    v5.8h, v30.8h, v31.8h\n"
       "stp    q4, q5, [%[outptr]], #32\n" // Write back last two elements

       : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
         [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
         [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
         [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
         [outptr] "+r"(outptr)

       : [skippf] "r"(skippf)
       : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
         "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
         "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
         "v31", "cc", "memory");
}

template <typename T>
static inline void interleave_4x1_4_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldr d0, [%[inptr0]], #8\n"   // d0 = A0A1A2A3
            "ldr d1, [%[inptr1]], #8\n"   // d1 = B0B1B2B3
            "ldr d2, [%[inptr2]], #8\n"   // d2 = C0C1C2C3
            "ldr d3, [%[inptr3]], #8\n"   // d3 = D0D1D2D3
            "zip1 v4.4h, v0.4h, v2.4h\n"  // d4 = A0C0A1C1
            "zip2 v8.4h, v0.4h, v2.4h\n"  // d8 = A2C2A3C3
            "zip1 v5.4h, v1.4h, v3.4h\n"  // d5 = B0D0B1D1
            "zip2 v9.4h, v1.4h, v3.4h\n"  // d9 = B2D2B3D3

            "zip1 v6.4h, v4.4h, v5.4h\n"  // d6 = A0B0C0D0
            "zip2 v7.4h, v4.4h, v5.4h\n"  // d7 = A1B1C1D1
            "stp d6, d7, [%[outptr]], #16\n"

            "zip1 v10.4h, v8.4h, v9.4h\n"  // d10 = A2B2C2D2
            "zip2 v11.4h, v8.4h, v9.4h\n"  // d11 = A3B3C3D3
            "stp d10, d11, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "memory");
}

static inline void interleave_4x1_2_d(const int64_t*& inptr0,
                                      const int64_t*& inptr1,
                                      const int64_t*& inptr2,
                                      const int64_t*& inptr3,
                                      int64_t*& outptr) {
    asm volatile(
            "ld1 {v0.2d}, [%[inptr0]], #16\n"  // d0 = A0A1
            "ld1 {v1.2d}, [%[inptr1]], #16\n"  // d1 = B0B1
            "ld1 {v2.2d}, [%[inptr2]], #16\n"  // d2 = C0C1
            "ld1 {v3.2d}, [%[inptr3]], #16\n"  // d3 = D0D1

            "zip1 v4.2d, v0.2d, v1.2d\n"  // d8 = A0B0
            "zip2 v5.2d, v0.2d, v1.2d\n"  // d9 = A1B1
            "zip1 v6.2d, v2.2d, v3.2d\n"  // d10 = C0D0
            "zip2 v7.2d, v2.2d, v3.2d\n"  // d11 = C1D1

            "st1 {v4.2d}, [%[outptr]], #16\n"
            "st1 {v6.2d}, [%[outptr]], #16\n"
            "st1 {v5.2d}, [%[outptr]], #16\n"
            "st1 {v7.2d}, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
}

static inline void interleave_4x2_2_d(const int64_t*& inptr0,
                                      const int64_t*& inptr1,
                                      const int64_t*& inptr2,
                                      const int64_t*& inptr3,
                                      int64_t*& outptr) {
    asm volatile(
            "ld1 {v0.2d}, [%[inptr0]], #16\n"  // d0 = A0
            "ld1 {v1.2d}, [%[inptr0]], #16\n"  // d1 = A1
            "ld1 {v2.2d}, [%[inptr1]], #16\n"  // d2 = B0
            "ld1 {v3.2d}, [%[inptr1]], #16\n"  // d3 = B1
            "ld1 {v4.2d}, [%[inptr2]], #16\n"  // d4 = C0
            "ld1 {v5.2d}, [%[inptr2]], #16\n"  // d5 = C1
            "ld1 {v6.2d}, [%[inptr3]], #16\n"  // d6 = D0
            "ld1 {v7.2d}, [%[inptr3]], #16\n"  // d7 = D1

            "st1 {v0.2d}, [%[outptr]], #16\n"
            "st1 {v2.2d}, [%[outptr]], #16\n"
            "st1 {v4.2d}, [%[outptr]], #16\n"
            "st1 {v6.2d}, [%[outptr]], #16\n"
            "st1 {v1.2d}, [%[outptr]], #16\n"
            "st1 {v3.2d}, [%[outptr]], #16\n"
            "st1 {v5.2d}, [%[outptr]], #16\n"
            "st1 {v7.2d}, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
}

static inline void interleave_12x1_4_s(
        const int32_t*& inptr0, const int32_t*& inptr1, const int32_t*& inptr2,
        const int32_t*& inptr3, const int32_t*& inptr4, const int32_t*& inptr5,
        const int32_t*& inptr6, const int32_t*& inptr7, const int32_t*& inptr8,
        const int32_t*& inptr9, const int32_t*& inptr10,
        const int32_t*& inptr11, int32_t*& outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "ld1 {v2.4s}, [%[inptr2]], #16\n"  // d2 = C0C1C2C3
            "ld1 {v3.4s}, [%[inptr3]], #16\n"  // d3 = D0D1D2D3
            "zip1 v12.4s, v0.4s, v2.4s\n"      // d12 = A0C0A1C1
            "zip2 v13.4s, v0.4s, v2.4s\n"      // d13 = A2C2A3C3
            "zip1 v14.4s, v1.4s, v3.4s\n"      // d14 = B0D0B1D1
            "zip2 v15.4s, v1.4s, v3.4s\n"      // d15 = B2D2B3D3
            "zip1 v0.4s, v12.4s, v14.4s\n"     // d0 = A0B0C0D0
            "zip2 v1.4s, v12.4s, v14.4s\n"     // d1 = A1B1C1D1
            "zip1 v2.4s, v13.4s, v15.4s\n"     // d2 = A2B2C2D2
            "zip2 v3.4s, v13.4s, v15.4s\n"     // d3 = A3B3C3D3

            "ld1 {v4.4s}, [%[inptr4]], #16\n"  // d4 = E0E1E2E3
            "ld1 {v5.4s}, [%[inptr5]], #16\n"  // d5 = F0F1F2F3
            "ld1 {v6.4s}, [%[inptr6]], #16\n"  // d6 = G0G1G2G3
            "ld1 {v7.4s}, [%[inptr7]], #16\n"  // d7 = H0H1H2H3
            "zip1 v16.4s, v4.4s, v6.4s\n"      // d16 = E0G0E1G1
            "zip2 v17.4s, v4.4s, v6.4s\n"      // d17 = E2G2E3G3
            "zip1 v18.4s, v5.4s, v7.4s\n"      // d18 = F0H0F1H1
            "zip2 v19.4s, v5.4s, v7.4s\n"      // d19 = F2H2F3H3
            "zip1 v4.4s, v16.4s, v18.4s\n"     // d4 = E0F0G0H0
            "zip2 v5.4s, v16.4s, v18.4s\n"     // d5 = E1F1G1H1
            "zip1 v6.4s, v17.4s, v19.4s\n"     // d6 = E2F2G2H2
            "zip2 v7.4s, v17.4s, v19.4s\n"     // d7 = E3F3G3H3

            "ld1 {v8.4s}, [%[inptr8]], #16\n"    // d8 = I0I1I2I3
            "ld1 {v9.4s}, [%[inptr9]], #16\n"    // d9 = J0J1J2J3
            "ld1 {v10.4s}, [%[inptr10]], #16\n"  // d10 = K0K1K2K3
            "ld1 {v11.4s}, [%[inptr11]], #16\n"  // d11 = L0L1L2L3
            "zip1 v20.4s, v8.4s, v10.4s\n"       // d20 = I0K0I1K1
            "zip2 v21.4s, v8.4s, v10.4s\n"       // d21 = I2K2I3K3
            "zip1 v22.4s, v9.4s, v11.4s\n"       // d22 = J0L0J1L1
            "zip2 v23.4s, v9.4s, v11.4s\n"       // d23 = J2L2J3L3
            "zip1 v8.4s, v20.4s, v22.4s\n"       // d8 = I0J0K0L0
            "zip2 v9.4s, v20.4s, v22.4s\n"       // d9 = I1J1K1L1
            "zip1 v10.4s, v21.4s, v23.4s\n"      // d10 = I2J2K2L2
            "zip2 v11.4s, v21.4s, v23.4s\n"      // d11 = I3J3K3L3

            "st1 {v0.4s}, [%[outptr]], #16\n"
            "st1 {v4.4s}, [%[outptr]], #16\n"
            "st1 {v8.4s}, [%[outptr]], #16\n"
            "st1 {v1.4s}, [%[outptr]], #16\n"
            "st1 {v5.4s}, [%[outptr]], #16\n"
            "st1 {v9.4s}, [%[outptr]], #16\n"
            "st1 {v2.4s}, [%[outptr]], #16\n"
            "st1 {v6.4s}, [%[outptr]], #16\n"
            "st1 {v10.4s}, [%[outptr]], #16\n"
            "st1 {v3.4s}, [%[outptr]], #16\n"
            "st1 {v7.4s}, [%[outptr]], #16\n"
            "st1 {v11.4s}, [%[outptr]], #16\n"

            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [inptr8] "+r"(inptr8),
            [inptr9] "+r"(inptr9), [inptr10] "+r"(inptr10),
            [inptr11] "+r"(inptr11), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "cc", "memory");
}

template <typename T>
static inline void interleave_12x1_4_h(
        const T*& in0, const T*& in1, const T*& in2, const T*& in3,
        const T*& in4, const T*& in5, const T*& in6, const T*& in7,
        const T*& in8, const T*& in9, const T*& in10, const T*& in11, T*& out) {
    static_assert(
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value,
            "interleave_12x1_4_h only support uint16_t and int16_t");
    const int16_t*& inptr0 = reinterpret_cast<const int16_t*&>(in0);
    const int16_t*& inptr1 = reinterpret_cast<const int16_t*&>(in1);
    const int16_t*& inptr2 = reinterpret_cast<const int16_t*&>(in2);
    const int16_t*& inptr3 = reinterpret_cast<const int16_t*&>(in3);
    const int16_t*& inptr4 = reinterpret_cast<const int16_t*&>(in4);
    const int16_t*& inptr5 = reinterpret_cast<const int16_t*&>(in5);
    const int16_t*& inptr6 = reinterpret_cast<const int16_t*&>(in6);
    const int16_t*& inptr7 = reinterpret_cast<const int16_t*&>(in7);
    const int16_t*& inptr8 = reinterpret_cast<const int16_t*&>(in8);
    const int16_t*& inptr9 = reinterpret_cast<const int16_t*&>(in9);
    const int16_t*& inptr10 = reinterpret_cast<const int16_t*&>(in10);
    const int16_t*& inptr11 = reinterpret_cast<const int16_t*&>(in11);
    int16_t*& outptr = reinterpret_cast<int16_t*&>(out);
    asm volatile(
            "ld1 {v0.4h}, [%[inptr0]], #8\n"  // d0 = A0A1A2A3
            "ld1 {v1.4h}, [%[inptr1]], #8\n"  // d1 = B0B1B2B3
            "ld1 {v2.4h}, [%[inptr2]], #8\n"  // d2 = C0C1C2C3
            "ld1 {v3.4h}, [%[inptr3]], #8\n"  // d3 = D0D1D2D3
            "zip1 v12.4h, v0.4h, v2.4h\n"     // d12 = A0C0A1C1
            "zip2 v13.4h, v0.4h, v2.4h\n"     // d13 = A2C2A3C3
            "zip1 v14.4h, v1.4h, v3.4h\n"     // d14 = B0D0B1D1
            "zip2 v15.4h, v1.4h, v3.4h\n"     // d15 = B2D2B3D3
            "zip1 v0.4h, v12.4h, v14.4h\n"    // d0 = A0B0C0D0
            "zip2 v1.4h, v12.4h, v14.4h\n"    // d1 = A1B1C1D1
            "zip1 v2.4h, v13.4h, v15.4h\n"    // d2 = A2B2C2D2
            "zip2 v3.4h, v13.4h, v15.4h\n"    // d3 = A3B3C3D3

            "ld1 {v4.4h}, [%[inptr4]], #8\n"  // d4 = E0E1E2E3
            "ld1 {v5.4h}, [%[inptr5]], #8\n"  // d5 = F0F1F2F3
            "ld1 {v6.4h}, [%[inptr6]], #8\n"  // d6 = G0G1G2G3
            "ld1 {v7.4h}, [%[inptr7]], #8\n"  // d7 = H0H1H2H3
            "zip1 v16.4h, v4.4h, v6.4h\n"     // d16 = E0G0E1G1
            "zip2 v17.4h, v4.4h, v6.4h\n"     // d17 = E2G2E3G3
            "zip1 v18.4h, v5.4h, v7.4h\n"     // d18 = F0H0F1H1
            "zip2 v19.4h, v5.4h, v7.4h\n"     // d19 = F2H2F3H3
            "zip1 v4.4h, v16.4h, v18.4h\n"    // d4 = E0F0G0H0
            "zip2 v5.4h, v16.4h, v18.4h\n"    // d5 = E1F1G1H1
            "zip1 v6.4h, v17.4h, v19.4h\n"    // d6 = E2F2G2H2
            "zip2 v7.4h, v17.4h, v19.4h\n"    // d7 = E3F3G3H3

            "ld1 {v8.4h}, [%[inptr8]], #8\n"    // d8 = I0I1I2I3
            "ld1 {v9.4h}, [%[inptr9]], #8\n"    // d9 = J0J1J2J3
            "ld1 {v10.4h}, [%[inptr10]], #8\n"  // d10 = K0K1K2K3
            "ld1 {v11.4h}, [%[inptr11]], #8\n"  // d11 = L0L1L2L3
            "zip1 v20.4h, v8.4h, v10.4h\n"      // d20 = I0K0I1K1
            "zip2 v21.4h, v8.4h, v10.4h\n"      // d21 = I2K2I3K3
            "zip1 v22.4h, v9.4h, v11.4h\n"      // d22 = J0L0J1L1
            "zip2 v23.4h, v9.4h, v11.4h\n"      // d23 = J2L2J3L3
            "zip1 v8.4h, v20.4h, v22.4h\n"      // d8 = I0J0K0L0
            "zip2 v9.4h, v20.4h, v22.4h\n"      // d9 = I1J1K1L1
            "zip1 v10.4h, v21.4h, v23.4h\n"     // d10 = I2J2K2L2
            "zip2 v11.4h, v21.4h, v23.4h\n"     // d11 = I3J3K3L3

            "st1 {v0.4h}, [%[outptr]], #8\n"   // d0 = A0B0C0D0
            "st1 {v4.4h}, [%[outptr]], #8\n"   // d4 = E0F0G0H0
            "st1 {v8.4h}, [%[outptr]], #8\n"   // d8 = I0J0K0L0
            "st1 {v1.4h}, [%[outptr]], #8\n"   // d1 = A1B1C1D1
            "st1 {v5.4h}, [%[outptr]], #8\n"   // d5 = E1F1G1H1
            "st1 {v9.4h}, [%[outptr]], #8\n"   // d9 = I1J1K1L1
            "st1 {v2.4h}, [%[outptr]], #8\n"   // d2 = A2B2C2D2
            "st1 {v6.4h}, [%[outptr]], #8\n"   // d6 = E2F2G2H2
            "st1 {v10.4h}, [%[outptr]], #8\n"  // d10 = I2J2K2L2
            "st1 {v3.4h}, [%[outptr]], #8\n"   // d3 = A3B3C3D3
            "st1 {v7.4h}, [%[outptr]], #8\n"   // d7 = E3F3G3H3
            "st1 {v11.4h}, [%[outptr]], #8\n"  // d11 = I3J3K3L3

            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [inptr8] "+r"(inptr8),
            [inptr9] "+r"(inptr9), [inptr10] "+r"(inptr10),
            [inptr11] "+r"(inptr11), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "cc", "memory");
}

template <typename T>
static inline void interleave_12x4_4_b(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       const T*& inptr4, const T*& inptr5,
                                       const T*& inptr6, const T*& inptr7,
                                       const T*& inptr8, const T*& inptr9,
                                       const T*& inptr10, const T*& inptr11,
                                       T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_12x4_4_b only support uint8_t and int8_t");
    interleave_12x1_4_s(reinterpret_cast<const int32_t*&>(inptr0),
                        reinterpret_cast<const int32_t*&>(inptr1),
                        reinterpret_cast<const int32_t*&>(inptr2),
                        reinterpret_cast<const int32_t*&>(inptr3),
                        reinterpret_cast<const int32_t*&>(inptr4),
                        reinterpret_cast<const int32_t*&>(inptr5),
                        reinterpret_cast<const int32_t*&>(inptr6),
                        reinterpret_cast<const int32_t*&>(inptr7),
                        reinterpret_cast<const int32_t*&>(inptr8),
                        reinterpret_cast<const int32_t*&>(inptr9),
                        reinterpret_cast<const int32_t*&>(inptr10),
                        reinterpret_cast<const int32_t*&>(inptr11),
                        reinterpret_cast<int32_t*&>(outptr));
}

static inline void interleave_2x1_4_s(const int32_t*& inptr0,
                                      const int32_t*& inptr1,
                                      int32_t*& outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "st1 {v0.4s}, [%[outptr]], #16\n"
            "st1 {v1.4s}, [%[outptr]], #16\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "cc", "memory");
}

static inline void interleave_8x1_4_s(
        const int32_t*& inptr0, const int32_t*& inptr1, const int32_t*& inptr2,
        const int32_t*& inptr3, const int32_t*& inptr4, const int32_t*& inptr5,
        const int32_t*& inptr6, const int32_t*& inptr7, int32_t*& outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "ld1 {v2.4s}, [%[inptr2]], #16\n"  // d2 = C0C1C2C3
            "ld1 {v3.4s}, [%[inptr3]], #16\n"  // d3 = D0D1D2D3
            "zip1 v8.4s, v0.4s, v2.4s\n"       // d8 = A0C0A1C1
            "zip2 v9.4s, v0.4s, v2.4s\n"       // d9 = A2C2A3C3
            "zip1 v10.4s, v1.4s, v3.4s\n"      // d10 = B0D0B1D1
            "zip2 v11.4s, v1.4s, v3.4s\n"      // d11 = B2D2B3D3
            "zip1 v12.4s, v8.4s, v10.4s\n"     // d12 = A0B0C0D0
            "zip2 v13.4s, v8.4s, v10.4s\n"     // d13 = A1B1C1D1
            "zip1 v14.4s, v9.4s, v11.4s\n"     // d14 = A2B2C2D2
            "zip2 v15.4s, v9.4s, v11.4s\n"     // d15 = A3B3C3D3

            "ld1 {v4.4s}, [%[inptr4]], #16\n"  // d4 = E0E1E2E3
            "ld1 {v5.4s}, [%[inptr5]], #16\n"  // d5 = F0F1F2F3
            "ld1 {v6.4s}, [%[inptr6]], #16\n"  // d6 = G0G1G2G3
            "ld1 {v7.4s}, [%[inptr7]], #16\n"  // d7 = H0H1H2H3
            "zip1 v16.4s, v4.4s, v6.4s\n"      // d16 = E0G0E1G1
            "zip2 v17.4s, v4.4s, v6.4s\n"      // d17 = E2G2E3G3
            "zip1 v18.4s, v5.4s, v7.4s\n"      // d18 = F0H0F1H1
            "zip2 v19.4s, v5.4s, v7.4s\n"      // d19 = F2H2F3H3
            "zip1 v20.4s, v16.4s, v18.4s\n"    // d20 = E0F0G0H0
            "zip2 v21.4s, v16.4s, v18.4s\n"    // d21 = E1F1G1H1
            "zip1 v22.4s, v17.4s, v19.4s\n"    // d22 = E2F2G2H2
            "zip2 v23.4s, v17.4s, v19.4s\n"    // d23 = E3F3G3H3

            "st1 {v12.4s}, [%[outptr]], #16\n"
            "st1 {v20.4s}, [%[outptr]], #16\n"
            "st1 {v13.4s}, [%[outptr]], #16\n"
            "st1 {v21.4s}, [%[outptr]], #16\n"
            "st1 {v14.4s}, [%[outptr]], #16\n"
            "st1 {v22.4s}, [%[outptr]], #16\n"
            "st1 {v15.4s}, [%[outptr]], #16\n"
            "st1 {v23.4s}, [%[outptr]], #16\n"

            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "cc", "memory");
}

static inline void interleave_8x1_2_d(
        const int64_t*& inptr0, const int64_t*& inptr1, const int64_t*& inptr2,
        const int64_t*& inptr3, const int64_t*& inptr4, const int64_t*& inptr5,
        const int64_t*& inptr6, const int64_t*& inptr7, int64_t*& outptr) {
    asm volatile(
            "ld1 {v0.2d}, [%[inptr0]], #16\n"  // d0 = A0A1
            "ld1 {v1.2d}, [%[inptr1]], #16\n"  // d1 = B0B1
            "ld1 {v2.2d}, [%[inptr2]], #16\n"  // d2 = C0C1
            "ld1 {v3.2d}, [%[inptr3]], #16\n"  // d3 = D0D1
            "ld1 {v4.2d}, [%[inptr4]], #16\n"  // d4 = E0E1
            "ld1 {v5.2d}, [%[inptr5]], #16\n"  // d5 = F0F1
            "ld1 {v6.2d}, [%[inptr6]], #16\n"  // d6 = G0G1
            "ld1 {v7.2d}, [%[inptr7]], #16\n"  // d7 = H0H1

            "zip1 v8.2d, v0.2d, v1.2d\n"   // d8 = A0B0
            "zip2 v9.2d, v0.2d, v1.2d\n"   // d9 = A1B1
            "zip1 v10.2d, v2.2d, v3.2d\n"  // d10 = C0D0
            "zip2 v11.2d, v2.2d, v3.2d\n"  // d11 = C1D1
            "zip1 v12.2d, v4.2d, v5.2d\n"  // d12 = E0F0
            "zip2 v13.2d, v4.2d, v5.2d\n"  // d13 = E1F1
            "zip1 v14.2d, v6.2d, v7.2d\n"  // d14 = G0H0
            "zip2 v15.2d, v6.2d, v7.2d\n"  // d15 = G1H1

            "st1 {v8.2d}, [%[outptr]], #16\n"
            "st1 {v10.2d}, [%[outptr]], #16\n"
            "st1 {v12.2d}, [%[outptr]], #16\n"
            "st1 {v14.2d}, [%[outptr]], #16\n"
            "st1 {v9.2d}, [%[outptr]], #16\n"
            "st1 {v11.2d}, [%[outptr]], #16\n"
            "st1 {v13.2d}, [%[outptr]], #16\n"
            "st1 {v15.2d}, [%[outptr]], #16\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

static inline void interleave_8x2_2_d(
        const int64_t*& inptr0, const int64_t*& inptr1, const int64_t*& inptr2,
        const int64_t*& inptr3, const int64_t*& inptr4, const int64_t*& inptr5,
        const int64_t*& inptr6, const int64_t*& inptr7, int64_t*& outptr) {
    asm volatile(
            "ld1 {v0.2d}, [%[inptr0]], #16\n"   // d0 = A0
            "ld1 {v1.2d}, [%[inptr0]], #16\n"   // d1 = A1
            "ld1 {v2.2d}, [%[inptr1]], #16\n"   // d2 = B0
            "ld1 {v3.2d}, [%[inptr1]], #16\n"   // d3 = B1
            "ld1 {v4.2d}, [%[inptr2]], #16\n"   // d4 = C0
            "ld1 {v5.2d}, [%[inptr2]], #16\n"   // d5 = C1
            "ld1 {v6.2d}, [%[inptr3]], #16\n"   // d6 = D0
            "ld1 {v7.2d}, [%[inptr3]], #16\n"   // d7 = D1
            "ld1 {v8.2d}, [%[inptr4]], #16\n"   // d8 = E0
            "ld1 {v9.2d}, [%[inptr4]], #16\n"   // d9 = E1
            "ld1 {v10.2d}, [%[inptr5]], #16\n"  // d10 = F0
            "ld1 {v11.2d}, [%[inptr5]], #16\n"  // d11 = F1
            "ld1 {v12.2d}, [%[inptr6]], #16\n"  // d12 = G0
            "ld1 {v13.2d}, [%[inptr6]], #16\n"  // d13 = G1
            "ld1 {v14.2d}, [%[inptr7]], #16\n"  // d14 = H0
            "ld1 {v15.2d}, [%[inptr7]], #16\n"  // d15 = H1

            "st1 {v0.2d}, [%[outptr]], #16\n"
            "st1 {v2.2d}, [%[outptr]], #16\n"
            "st1 {v4.2d}, [%[outptr]], #16\n"
            "st1 {v6.2d}, [%[outptr]], #16\n"
            "st1 {v8.2d}, [%[outptr]], #16\n"
            "st1 {v10.2d}, [%[outptr]], #16\n"
            "st1 {v12.2d}, [%[outptr]], #16\n"
            "st1 {v14.2d}, [%[outptr]], #16\n"
            "st1 {v1.2d}, [%[outptr]], #16\n"
            "st1 {v3.2d}, [%[outptr]], #16\n"
            "st1 {v5.2d}, [%[outptr]], #16\n"
            "st1 {v7.2d}, [%[outptr]], #16\n"
            "st1 {v9.2d}, [%[outptr]], #16\n"
            "st1 {v11.2d}, [%[outptr]], #16\n"
            "st1 {v13.2d}, [%[outptr]], #16\n"
            "st1 {v15.2d}, [%[outptr]], #16\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

template <typename T>
static inline void interleave_2x4_4_b(const T*& inptr0, const T*& inptr1,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_2x4_4_b only support uint8_t and int8_t");
    interleave_2x1_4_s(reinterpret_cast<const int32_t*&>(inptr0),
                       reinterpret_cast<const int32_t*&>(inptr1),
                       reinterpret_cast<int32_t*&>(outptr));
}

template <typename T>
static inline void interleave_8x4_4_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_8x4_4_b only support uint8_t and int8_t");
    interleave_8x1_4_s(reinterpret_cast<const int32_t*&>(inptr0),
                       reinterpret_cast<const int32_t*&>(inptr1),
                       reinterpret_cast<const int32_t*&>(inptr2),
                       reinterpret_cast<const int32_t*&>(inptr3),
                       reinterpret_cast<const int32_t*&>(inptr4),
                       reinterpret_cast<const int32_t*&>(inptr5),
                       reinterpret_cast<const int32_t*&>(inptr6),
                       reinterpret_cast<const int32_t*&>(inptr7),
                       reinterpret_cast<int32_t*&>(outptr));
}

template <typename T>
static inline void interleave_8x4_1_h(const T*& in0, const T*& in1,
                                      const T*& in2, const T*& in3, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldr q0, [%[in0]], #16\n"  // A1A2A3A4A5A6A7A8
            "ldr q1, [%[in1]], #16\n"  // B1B2B3B4B5B6B7B8
            "ldr q2, [%[in2]], #16\n"  // C1C2C3C4C5C6C7C8
            "ldr q3, [%[in3]], #16\n"  // D1D2D3D4D5D6D7D8

            "trn1 v4.8h, v0.8h, v1.8h\n"  // A1B1A3B3A5B5A7B7
            "trn2 v5.8h, v0.8h, v1.8h\n"  // A2B2A4B4A6B6A8B8
            "trn1 v6.8h, v2.8h, v3.8h\n"  // C1D1C3D3C5D5C7D7
            "trn2 v7.8h, v2.8h, v3.8h\n"  // C2D2C4D4C6D6C8D8

            "zip1 v8.4s, v4.4s, v6.4s\n"   // A1B1C1D1A3B3C3D3
            "zip2 v9.4s, v4.4s, v6.4s\n"   // A5B5C5D5A7B7C7D7
            "zip1 v10.4s, v5.4s, v7.4s\n"  // A2B2C2D2A4B4C4D4
            "zip2 v11.4s, v5.4s, v7.4s\n"  // A6B6C6D6A8B8C8D8

            "zip1 v12.2d, v8.2d, v10.2d\n"  // A1B1C1D1A2B2C2D2
            "zip2 v13.2d, v8.2d, v10.2d\n"  // A3B3C3D3A4B4C4D4
            "zip1 v14.2d, v9.2d, v11.2d\n"  // A5B5C5D5A6B6C6D6
            "zip2 v15.2d, v9.2d, v11.2d\n"  // A7B7C7D7A8B8C8D8

            "st1 {v12.2d}, [%[out]], #16\n"
            "st1 {v13.2d}, [%[out]], #16\n"
            "st1 {v14.2d}, [%[out]], #16\n"
            "st1 {v15.2d}, [%[out]], #16\n"
            : [in0] "+r"(in0), [in1] "+r"(in1), [in2] "+r"(in2),
              [in3] "+r"(in3), [out] "+r"(out)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "memory");
}

template <typename T>
static inline void interleave_8x8_2_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_8x8_2_b only support uint8_t and int8_t");
    interleave_8x1_2_d(reinterpret_cast<const int64_t*&>(inptr0),
                       reinterpret_cast<const int64_t*&>(inptr1),
                       reinterpret_cast<const int64_t*&>(inptr2),
                       reinterpret_cast<const int64_t*&>(inptr3),
                       reinterpret_cast<const int64_t*&>(inptr4),
                       reinterpret_cast<const int64_t*&>(inptr5),
                       reinterpret_cast<const int64_t*&>(inptr6),
                       reinterpret_cast<const int64_t*&>(inptr7),
                       reinterpret_cast<int64_t*&>(outptr));
}

template <typename T>
static inline void interleave_8x8_2_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value,
            "interleave_8x8_2_h only support uint16_t and int16_t");
    interleave_8x2_2_d(reinterpret_cast<const int64_t*&>(inptr0),
                       reinterpret_cast<const int64_t*&>(inptr1),
                       reinterpret_cast<const int64_t*&>(inptr2),
                       reinterpret_cast<const int64_t*&>(inptr3),
                       reinterpret_cast<const int64_t*&>(inptr4),
                       reinterpret_cast<const int64_t*&>(inptr5),
                       reinterpret_cast<const int64_t*&>(inptr6),
                       reinterpret_cast<const int64_t*&>(inptr7),
                       reinterpret_cast<int64_t*&>(outptr));
}

template <typename T>
static inline void interleave_8x2_8_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_8x2_8_b only support uint8_t and int8_t");
    interleave_8x1_8_h(reinterpret_cast<const int16_t*&>(inptr0),
                       reinterpret_cast<const int16_t*&>(inptr1),
                       reinterpret_cast<const int16_t*&>(inptr2),
                       reinterpret_cast<const int16_t*&>(inptr3),
                       reinterpret_cast<const int16_t*&>(inptr4),
                       reinterpret_cast<const int16_t*&>(inptr5),
                       reinterpret_cast<const int16_t*&>(inptr6),
                       reinterpret_cast<const int16_t*&>(inptr7),
                       reinterpret_cast<int16_t*&>(outptr));
}

template <typename T>
static inline void interleave_8x8_1_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_8x8_1_b only support uint8_t and int8_t");
    asm volatile(
            "ld1 {v0.d}[0],  [%[inptr0]], 8\n"  // A1A2A3A4A5A6A7A8
            "ld1 {v0.d}[1],  [%[inptr1]], 8\n"  // B1B2B3B4B5B6B7B8
            "ld1 {v1.d}[0],  [%[inptr2]], 8\n"  // C1C2C3C4C5C6C7C8
            "ld1 {v1.d}[1],  [%[inptr3]], 8\n"  // D1D2D3D4D5D6D7D8
            "ld1 {v2.d}[0],  [%[inptr4]], 8\n"  // E1E2E3E4E5E6E7E8
            "ld1 {v2.d}[1],  [%[inptr5]], 8\n"  // F1F2F3F4F5F6F7F8
            "ld1 {v3.d}[0],  [%[inptr6]], 8\n"  // G1G2G3G4G5G6G7G8
            "ld1 {v3.d}[1],  [%[inptr7]], 8\n"  // H1H2H3H4H5H6H7H8

            "st1 {v0.2d},  [%[outptr]], 16\n"  // A1A2A3A4A5A6A7A8B1B2B3B4B5B6B7B8
            "st1 {v1.2d},  [%[outptr]], 16\n"  // C1C2C3C4C5C6C7C8D1D2D3D4D5D6D7D8
            "st1 {v2.2d},  [%[outptr]], 16\n"  // E1E2E3E4E5E6E7E8F1F2F3F4F5F6F7F8
            "st1 {v3.2d},  [%[outptr]], 16\n"  // G1G2G3G4G5G6G7G8H1H2H3H4H5H6H7H8
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "memory");
}


template <typename T>
static inline void interleave_8x4_1_b_with_shift(
        const T*& inptr0, const T*& inptr1, const T*& inptr2, const T*& inptr3,
        const T*& inptr4, const T*& inptr5, const T*& inptr6, const T*& inptr7,
        T* outptr) {
    static_assert(sizeof(T) == 1, "only support size == 1");
    asm volatile(
            "ld1 {v0.s}[0], [%[inptr0]], #4\n"
            "ld1 {v0.s}[1], [%[inptr1]], #4\n"
            "ld1 {v0.s}[2], [%[inptr2]], #4\n"
            "ld1 {v0.s}[3], [%[inptr3]], #4\n"
            "ld1 {v1.s}[0], [%[inptr4]], #4\n"
            "ld1 {v1.s}[1], [%[inptr5]], #4\n"
            "ld1 {v1.s}[2], [%[inptr6]], #4\n"
            "ld1 {v1.s}[3], [%[inptr7]], #4\n"
            "shl  v2.16b, v0.16b,        #4\n"
            "shl  v5.16b, v1.16b,        #4\n"
            "sshr v3.16b, v0.16b,        #4\n"  // hig
            "sshr v4.16b, v2.16b,        #4\n"  // low
            "sshr v6.16b, v1.16b,        #4\n"  // hig
            "sshr v7.16b, v5.16b,        #4\n"  // low
            "zip1 v8.16b, v4.16b,    v3.16b\n"
            "zip2 v9.16b, v4.16b,    v3.16b\n"
            "zip1 v10.16b, v7.16b,   v6.16b\n"
            "zip2 v11.16b, v7.16b,   v6.16b\n"
            "st1 {v8.16b-v11.16b},[%[outptr]],#64"
            : [ inptr0 ] "+r"(inptr0), [ inptr1 ] "+r"(inptr1),
              [ inptr2 ] "+r"(inptr2), [ inptr3 ] "+r"(inptr3),
              [ inptr4 ] "+r"(inptr4), [ inptr5 ] "+r"(inptr5),
              [ inptr6 ] "+r"(inptr6), [ inptr7 ] "+r"(inptr7),
              [ outptr ] "+r"(outptr)
            :
            : "v0", "v1","v2","v3","v4","v5","v6","v7","v8","v9","v10","v11","memory");
}

template <typename T>
static inline void interleave_8x8_1_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value,
            "interleave_8x8_1_h only support uint16_t and int16_t");
    asm volatile(
            "ld1 {v0.8h},  [%[inptr0]], #16\n"  // A1A2A3A4A5A6A7A8
            "ld1 {v1.8h},  [%[inptr1]], #16\n"  // B1B2B3B4B5B6B7B8
            "ld1 {v2.8h},  [%[inptr2]], #16\n"  // C1C2C3C4C5C6C7C8
            "ld1 {v3.8h},  [%[inptr3]], #16\n"  // D1D2D3D4D5D6D7D8
            "ld1 {v4.8h},  [%[inptr4]], #16\n"  // E1E2E3E4E5E6E7E8
            "ld1 {v5.8h},  [%[inptr5]], #16\n"  // F1F2F3F4F5F6F7F8
            "ld1 {v6.8h},  [%[inptr6]], #16\n"  // G1G2G3G4G5G6G7G8
            "ld1 {v7.8h},  [%[inptr7]], #16\n"  // H1H2H3H4H5H6H7H8

            "st1 {v0.8h},  [%[outptr]], #16\n"  // A1A2A3A4A5A6A7A8
            "st1 {v1.8h},  [%[outptr]], #16\n"  // B1B2B3B4B5B6B7B8
            "st1 {v2.8h},  [%[outptr]], #16\n"  // C1C2C3C4C5C6C7C8
            "st1 {v3.8h},  [%[outptr]], #16\n"  // D1D2D3D4D5D6D7D8
            "st1 {v4.8h},  [%[outptr]], #16\n"  // E1E2E3E4E5E6E7E8
            "st1 {v5.8h},  [%[outptr]], #16\n"  // F1F2F3F4F5F6F7F8
            "st1 {v6.8h},  [%[outptr]], #16\n"  // G1G2G3G4G5G6G7G8
            "st1 {v7.8h},  [%[outptr]], #16\n"  // H1H2H3H4H5H6H7H8
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

static inline void interleave_4x1_4_s(const int32_t*& inptr0,
                                      const int32_t*& inptr1,
                                      const int32_t*& inptr2,
                                      const int32_t*& inptr3,
                                      int32_t*& outptr) {
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "ld1 {v2.4s}, [%[inptr2]], #16\n"  // d2 = C0C1C2C3
            "ld1 {v3.4s}, [%[inptr3]], #16\n"  // d3 = D0D1D2D3
            "zip1 v8.4s, v0.4s, v2.4s\n"       // d8 = A0C0A1C1
            "zip2 v9.4s, v0.4s, v2.4s\n"       // d9 = A2C2A3C3
            "zip1 v10.4s, v1.4s, v3.4s\n"      // d10 = B0D0B1D1
            "zip2 v11.4s, v1.4s, v3.4s\n"      // d11 = B2D2B3D3
            "zip1 v12.4s, v8.4s, v10.4s\n"     // d12 = A0B0C0D0
            "zip2 v13.4s, v8.4s, v10.4s\n"     // d13 = A1B1C1D1
            "zip1 v14.4s, v9.4s, v11.4s\n"     // d14 = A2B2C2D2
            "zip2 v15.4s, v9.4s, v11.4s\n"     // d15 = A3B3C3D3

            "st1 {v12.4s}, [%[outptr]], #16\n"
            "st1 {v13.4s}, [%[outptr]], #16\n"
            "st1 {v14.4s}, [%[outptr]], #16\n"
            "st1 {v15.4s}, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

template <typename T>
static inline void interleave_4x8_1_s(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(sizeof(T) == 4, "only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s}, [%[inptr0]], #32\n"
            "ld1 {v2.4s, v3.4s}, [%[inptr1]], #32\n"
            "ld1 {v4.4s, v5.4s}, [%[inptr2]], #32\n"
            "ld1 {v6.4s, v7.4s}, [%[inptr3]], #32\n"
            "st1 {v0.4s, v1.4s}, [%[outptr]], #32\n"
            "st1 {v2.4s, v3.4s}, [%[outptr]], #32\n"
            "st1 {v4.4s, v5.4s}, [%[outptr]], #32\n"
            "st1 {v6.4s, v7.4s}, [%[outptr]], #32\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "cc", "memory");
}

template <typename T>
static inline void interleave_4x12_1_s(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 4, "only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s}, [%[inptr0]], #48\n"
            "ld1 {v4.4s, v5.4s, v6.4s}, [%[inptr1]], #48\n"
            "ld1 {v8.4s, v9.4s, v10.4s}, [%[inptr2]], #48\n"
            "ld1 {v12.4s, v13.4s, v14.4s}, [%[inptr3]], #48\n"
            "st1 {v0.4s, v1.4s, v2.4s}, [%[outptr]], #48\n"
            "st1 {v4.4s, v5.4s, v6.4s}, [%[outptr]], #48\n"
            "st1 {v8.4s, v9.4s, v10.4s}, [%[outptr]], #48\n"
            "st1 {v12.4s, v13.4s, v14.4s}, [%[outptr]], #48\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v4", "v5", "v6", "v8", "v9", "v10", "v12",
              "v13", "v14", "cc", "memory");
}

template <typename T>
static inline void interleave_4x16_1_b(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 1, "only support size == 1");
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"  // d0 = A0A1A2A3
            "ld1 {v1.4s}, [%[inptr1]], #16\n"  // d1 = B0B1B2B3
            "ld1 {v2.4s}, [%[inptr2]], #16\n"  // d2 = C0C1C2C3
            "ld1 {v3.4s}, [%[inptr3]], #16\n"  // d3 = D0D1D2D3
            "st1 {v0.4s}, [%[outptr]], #16\n"
            "st1 {v1.4s}, [%[outptr]], #16\n"
            "st1 {v2.4s}, [%[outptr]], #16\n"
            "st1 {v3.4s}, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "cc", "memory");
}


template <typename T>
static inline void interleave_4x16_1_s(const T*& inptr0, const T*& inptr1,
                                       const T*& inptr2, const T*& inptr3,
                                       T*& outptr) {
    static_assert(sizeof(T) == 4, "only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[inptr0]], #64\n"
            "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[inptr1]], #64\n"
            "ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[inptr2]], #64\n"
            "ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[inptr3]], #64\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[outptr]], #64\n"
            "st1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[outptr]], #64\n"
            "st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [%[outptr]], #64\n"
            "st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [%[outptr]], #64\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "cc", "memory");
}

template <typename T>
static inline void interleave_4x2_4_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x2_4_b only support uint8_t and int8_t");
    interleave_4x1_4_h(reinterpret_cast<const int16_t*&>(inptr0),
                       reinterpret_cast<const int16_t*&>(inptr1),
                       reinterpret_cast<const int16_t*&>(inptr2),
                       reinterpret_cast<const int16_t*&>(inptr3),
                       reinterpret_cast<int16_t*&>(outptr));
}

template <typename T>
static inline void interleave_4x4_4_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x4_4_b only support uint8_t and int8_t");
    interleave_4x1_4_s(reinterpret_cast<const int32_t*&>(inptr0),
                       reinterpret_cast<const int32_t*&>(inptr1),
                       reinterpret_cast<const int32_t*&>(inptr2),
                       reinterpret_cast<const int32_t*&>(inptr3),
                       reinterpret_cast<int32_t*&>(outptr));
}

template <typename T>
static inline void interleave_4x4_1_s(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_4x4_1_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"
            "ld1 {v1.4s}, [%[inptr1]], #16\n"
            "ld1 {v2.4s}, [%[inptr2]], #16\n"
            "ld1 {v3.4s}, [%[inptr3]], #16\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[outptr]], #64\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "cc", "memory");
}

template <typename T>
static inline void interleave_2x4_4_s(const T*& inptr0, const T*& inptr1,
                                      T* outptr) {
    static_assert(sizeof(T) == 4, "interleave_2x4_4_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[inptr0]], #64\n"
            "ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [%[inptr1]], #64\n"
            "stp q0, q4, [%[outptr]]\n"
            "stp q1, q5, [%[outptr], #32]\n"
            "stp q2, q6, [%[outptr], #64]\n"
            "stp q3, q7, [%[outptr], #96]\n"

            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

template <typename T>
static inline void interleave_1x4_4_s(const T*& inptr0, T* outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x4_4_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[inptr0]], #64\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[outptr]]\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "memory");
}

template <typename T>
static inline void interleave_4x8_2_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "interleave_4x8_2_b only support uint8_t and int8_t");
    interleave_4x1_2_d(reinterpret_cast<const int64_t*&>(inptr0),
                       reinterpret_cast<const int64_t*&>(inptr1),
                       reinterpret_cast<const int64_t*&>(inptr2),
                       reinterpret_cast<const int64_t*&>(inptr3),
                       reinterpret_cast<int64_t*&>(outptr));
}

template <typename T>
static inline void interleave_4x8_2_h(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T*& outptr) {
    static_assert(
            std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value,
            "interleave_4x8_2_h only support uint16_t and int16_t");
    interleave_4x2_2_d(reinterpret_cast<const int64_t*&>(inptr0),
                       reinterpret_cast<const int64_t*&>(inptr1),
                       reinterpret_cast<const int64_t*&>(inptr2),
                       reinterpret_cast<const int64_t*&>(inptr3),
                       reinterpret_cast<int64_t*&>(outptr));
}

template <typename T>
static inline void interleave_1x16_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x16_1_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[inptr0]], #64\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%[outptr]], #64\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "cc", "memory");
}
template <typename T>
static inline void interleave_1x12_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x12_1_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s, v2.4s}, [%[inptr0]], #48\n"
            "st1 {v0.4s, v1.4s, v2.4s}, [%[outptr]], #48\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "cc", "memory");
}

template <typename T>
static inline void interleave_1x8_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x8_1_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s, v1.4s}, [%[inptr0]], #32\n"
            "st1 {v0.4s, v1.4s}, [%[outptr]], #32\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "cc", "memory");
}

template <typename T>
static inline void interleave_1x4_1_s(const T*& inptr0, T*& outptr) {
    static_assert(sizeof(T) == 4, "interleave_1x4_1_s only support size == 4");
    asm volatile(
            "ld1 {v0.4s}, [%[inptr0]], #16\n"
            "st1 {v0.4s}, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "cc", "memory");
}

template <typename T>
static inline void interleave_helper(const T*& inptr, T*& outptr, int unroll_k,
                                     int ksize, T val = 0) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}

template <typename T>
static inline void interleave_1(const T*& inptr0, T*& outptr, int unroll_k,
                                int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_4(const T*& inptr0, const T*& inptr1,
                                const T*& inptr2, const T*& inptr3, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_8(const T*& inptr0, const T*& inptr1,
                                const T*& inptr2, const T*& inptr3,
                                const T*& inptr4, const T*& inptr5,
                                const T*& inptr6, const T*& inptr7, T*& outptr,
                                int unroll_k, int ksize, T val = 0) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        interleave_helper(inptr3, outptr, unroll_k, size, val);
        interleave_helper(inptr4, outptr, unroll_k, size, val);
        interleave_helper(inptr5, outptr, unroll_k, size, val);
        interleave_helper(inptr6, outptr, unroll_k, size, val);
        interleave_helper(inptr7, outptr, unroll_k, size, val);
    }
}

template <typename T>
static inline void interleave_12(const T*& inptr0, const T*& inptr1,
                                 const T*& inptr2, const T*& inptr3,
                                 const T*& inptr4, const T*& inptr5,
                                 const T*& inptr6, const T*& inptr7,
                                 const T*& inptr8, const T*& inptr9,
                                 const T*& inptr10, const T*& inptr11,
                                 T*& outptr, int unroll_k, int ksize) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = std::min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size);
        interleave_helper(inptr1, outptr, unroll_k, size);
        interleave_helper(inptr2, outptr, unroll_k, size);
        interleave_helper(inptr3, outptr, unroll_k, size);
        interleave_helper(inptr4, outptr, unroll_k, size);
        interleave_helper(inptr5, outptr, unroll_k, size);
        interleave_helper(inptr6, outptr, unroll_k, size);
        interleave_helper(inptr7, outptr, unroll_k, size);
        interleave_helper(inptr8, outptr, unroll_k, size);
        interleave_helper(inptr9, outptr, unroll_k, size);
        interleave_helper(inptr10, outptr, unroll_k, size);
        interleave_helper(inptr11, outptr, unroll_k, size);
    }
}
/* ======================== transpose pack B ======================== */
/**
 * transpose_INTERLEAVE_UNROLLK_BATCH_type
 *
 * BATCH means process BATCH * INTERLEAVE cols once, BATCH * sizeof(TYPE) *
 * INTERLEAVE = 16bytes(128bits, a vector size).
 *
 * the elements traverse order:
 * rep(j, 0, INTERLEAVE) rep(i, 0, UNROLL_K) *ouptr++ = inptr[i, j]
 */
template <typename T>
static inline void transpose_24x4_1_h(const T*& in0, const T*& in1,
                                      const T*& in2, const T*& in3, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
        "ldp q0, q1, [%[in0]], #32\n"
        "stp q0, q1, [%[out]]\n"
        "ldr q2, [%[in0]], #16\n"
        ASM_PREFETCH("[%[in0], #192]")
        "ldp q3, q4, [%[in1]], #32\n"
        "stp q2, q3, [%[out], #32]\n"
        "ldr q5, [%[in1]], #16\n"
        ASM_PREFETCH("[%[in1], #192]")
        "stp q4, q5, [%[out], #64]\n"
        "ldp q6, q7, [%[in2]], #32\n"
        "stp q6, q7, [%[out], #96]\n"
        "ldr q8, [%[in2]], #16\n"
        ASM_PREFETCH("[%[in2], #192]")
        "ldp q9, q10, [%[in3]], #32\n"
        "stp q8, q9, [%[out], #128]\n"
        "ldr q11, [%[in3]], #16\n"
        "stp q10, q11, [%[out], #160]\n"
        ASM_PREFETCH("[%[in3], #192]")

        : [in0] "+r"(in0), [in1] "+r"(in1), [in2] "+r"(in2),
          [in3] "+r"(in3), [out] "+r"(out)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "memory");
}

template <typename T>
static inline void transpose_16x4_1_h(const T*& in0, const T*& in1,
                                      const T*& in2, const T*& in3, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldp q0, q1, [%[in0]], #32\n"
            "stp q0, q1, [%[out]]\n"
            "ldp q2, q3, [%[in1]], #32\n"
            "stp q2, q3, [%[out], #32]\n"
            "ldp q4, q5, [%[in2]], #32\n"
            "stp q4, q5, [%[out], #64]\n"
            "ldp q6, q7, [%[in3]], #32\n"
            "stp q6, q7, [%[out], #96]\n"
            : [in0] "+r"(in0), [in1] "+r"(in1), [in2] "+r"(in2),
              [in3] "+r"(in3), [out] "+r"(out)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

template <typename T>
static inline void transpose_8x4_1_h(const T*& in0, const T*& in1,
                                     const T*& in2, const T*& in3, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldr q0, [%[in0]], #16\n"
            "str q0, [%[out]]\n"
            "ldr q1, [%[in1]], #16\n"
            "str q1, [%[out], #16]\n"
            "ldr q2, [%[in2]], #16\n"
            "str q2, [%[out], #32]\n"
            "ldr q3, [%[in3]], #16\n"
            "str q3, [%[out], #48]\n"
            : [in0] "+r"(in0), [in1] "+r"(in1), [in2] "+r"(in2),
              [in3] "+r"(in3), [out] "+r"(out)
            :
            : "v0", "v1", "v2", "v3", "memory");
}

template <typename T>
static inline void transpose_24x2_1_h(const T*& in0, const T*& in1, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
        "ldp q0, q1, [%[in0]], #32\n"
        "stp q0, q1, [%[out]]\n"
        "ldr q2, [%[in0]], #16\n"
        ASM_PREFETCH("[%[in0], #192]")
        "ldp q3, q4, [%[in1]], #32\n"
        "stp q2, q3, [%[out], #32]\n"
        "ldr q5, [%[in1]], #16\n"
        ASM_PREFETCH("[%[in1], #192]")
        "stp q4, q5, [%[out], #64]\n"
        : [in0] "+r"(in0), [in1] "+r"(in1), [out] "+r"(out)
        :
        : "v0", "v1", "v2", "v3", "v4", "v5", "memory");
}

template <typename T>
static inline void transpose_16x2_1_h(const T*& in0, const T*& in1, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldp q0, q1, [%[in0]], #32\n"
            "stp q0, q1, [%[out]]\n"
            "ldp q2, q3, [%[in1]], #32\n"
            "stp q2, q3, [%[out], #32]\n"
            : [in0] "+r"(in0), [in1] "+r"(in1), [out] "+r"(out)
            :
            : "v0", "v1", "v2", "v3", "memory");
}

template <typename T>
static inline void transpose_8x2_1_h(const T*& in0, const T*& in1, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldr q0, [%[in0]], #16\n"
            "str q0, [%[out]]\n"
            "ldr q1, [%[in1]], #16\n"
            "str q1, [%[out], #16]\n"
            : [in0] "+r"(in0), [in1] "+r"(in1), [out] "+r"(out)
            :
            : "v0", "v1", "memory");
}

template <typename T>
static inline void transpose_24x1_1_h(const T*& in0, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    // clang-format off
    asm volatile(
            "ldp q0, q1, [%[in0]], #32\n"
            "stp q0, q1, [%[out]] \n"
            "ldr q2, [%[in0]], #16 \n"
            ASM_PREFETCH("[%[in0], #192]")
            "str q2, [%[out], #32] \n"
            : [in0] "+r"(in0), [out] "+r"(out)
            :
            : "v0", "v1", "v2", "memory");
    // clang-format on
}

template <typename T>
static inline void transpose_16x1_1_h(const T*& in0, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldp q0, q1, [%[in0]], #32\n"
            "stp q0, q1, [%[out]]\n"
            : [in0] "+r"(in0), [out] "+r"(out)
            :
            : "v0", "v1", "memory");
}

template <typename T>
static inline void transpose_12x1_1_h(const T*& in0, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    // clang-format off
    asm volatile(
            "ld1 {v0.8h}, [%[in0]], #16\n"
            "ld1 {v1.4h}, [%[in0]], #8\n"
            "st1 {v0.8h}, [%[out]], #16\n"
            "st1 {v1.4h}, [%[out]], #8\n"
            : [in0] "+r"(in0), [out] "+r"(out)
            :
            : "v0", "v1", "memory");
    // clang-format on
}

template <typename T>
static inline void transpose_8x1_1_h(const T*& in0, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    asm volatile(
            "ldr q0, [%[in0]], #16\n"
            "str q0, [%[out]]\n"
            : [in0] "+r"(in0), [out] "+r"(out)
            :
            : "v0", "memory");
}

template <typename T>
static inline void transpose_4x1_1_h(const T*& in0, T* out) {
    static_assert(sizeof(T) == 2, "only support size == 2");
    // clang-format off
    asm volatile(
            "ld1 {v0.4h}, [%[in0]], #8\n"
            "st1 {v0.4h}, [%[out]], #8\n"
            : [in0] "+r"(in0), [out] "+r"(out)
            :
            : "v0", "memory");
    // clang-format on
}

template <typename T>
static inline void transpose_4x4_1_s(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T* outptr, int stride = 16) {
    static_assert(sizeof(T) == 4,
                  "transpose_4x4_1_s only support sizeof(T) == 4");

    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], 16\n"  // A0A1A2A3
            "ld1 {v1.4s},  [%[inptr1]], 16\n"  // B0B1B2B3
            "ld1 {v2.4s},  [%[inptr2]], 16\n"  // C0C1C2C3
            "ld1 {v3.4s},  [%[inptr3]], 16\n"  // D0D1D2D3

            "zip1 v4.4s, v0.4s, v1.4s\n"
            "zip1 v5.4s, v2.4s, v3.4s\n"
            "zip2 v6.4s, v0.4s, v1.4s\n"
            "zip2 v7.4s, v2.4s, v3.4s\n"

            "zip1 v8.2d, v4.2d, v5.2d\n"
            "zip1 v9.2d, v6.2d, v7.2d\n"
            "zip2 v10.2d, v4.2d, v5.2d\n"
            "zip2 v11.2d, v6.2d, v7.2d\n"

            "st1 {v8.4s},  [%[outptr]], %x[stride]\n"
            "st1 {v10.4s},  [%[outptr]], %x[stride]\n"
            "st1 {v9.4s},  [%[outptr]], %x[stride]\n"
            "st1 {v11.4s},  [%[outptr]], %x[stride]\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "memory");
}

template <typename T>
static inline void transpose_1x12_4_s(const T*& inptr0, T* outptr) {
    static_assert(sizeof(T) == 4,
                  "transpose_1x12_4_s only support sizeof(T) == 4");

    asm volatile(
            "ld4 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[inptr0]], #64\n"
            "ld4 {v4.4s, v5.4s, v6.4s, v7.4s},  [%[inptr0]], #64\n"
            "ld4 {v8.4s, v9.4s, v10.4s, v11.4s},[%[inptr0]], #64\n"

            "stp q0, q4, [%[outptr]] \n"
            "stp q8, q1, [%[outptr], #32] \n"
            "stp q5, q9, [%[outptr], #64] \n"
            "stp q2, q6, [%[outptr], #96] \n"
            "stp q10, q3, [%[outptr], #128] \n"
            "stp q7, q11, [%[outptr], #160] \n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "memory");
}

template <typename T>
static inline void transpose_1x4_4_s(const T*& inptr0, T* outptr) {
    static_assert(sizeof(T) == 4,
                  "transpose_1x4_4_s only support sizeof(T) == 4");

    asm volatile(
            "ld4 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[inptr0]], #64\n"
            "st1 {v0.4s, v1.4s, v2.4s, v3.4s},  [%[outptr]]\n"
            : [inptr0] "+r"(inptr0), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "memory");
}

template <typename T>
static inline void transpose_8x4_1_s(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     const T*& inptr4, const T*& inptr5,
                                     const T*& inptr6, const T*& inptr7,
                                     T* outptr) {
    static_assert(sizeof(T) == 4,
                  "transpose_8x4_1_s only support sizeof(T) == 4");

    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], 16\n"  // A0A1A2A3
            "ld1 {v1.4s},  [%[inptr1]], 16\n"  // B0B1B2B3
            "ld1 {v2.4s},  [%[inptr2]], 16\n"  // C0C1C2C3
            "ld1 {v3.4s},  [%[inptr3]], 16\n"  // D0D1D2D3
            "ld1 {v4.4s},  [%[inptr4]], 16\n"  // E0E1E2E3
            "ld1 {v5.4s},  [%[inptr5]], 16\n"  // F0F1F2F3
            "ld1 {v6.4s},  [%[inptr6]], 16\n"  // G0G1G2G3
            "ld1 {v7.4s},  [%[inptr7]], 16\n"  // H0H1H2H3

            "zip1 v8.4s, v0.4s, v1.4s\n"   // A0B0A1B1
            "zip2 v9.4s, v0.4s, v1.4s\n"   // A2B2A3B3
            "zip1 v10.4s, v2.4s, v3.4s\n"  // C0D0C1D1
            "zip2 v11.4s, v2.4s, v3.4s\n"  // C2D2C3D3
            "zip1 v12.4s, v4.4s, v5.4s\n"  // E0F0E1F1
            "zip2 v13.4s, v4.4s, v5.4s\n"  // E2F2E3F3
            "zip1 v14.4s, v6.4s, v7.4s\n"  // G0H0G1H1
            "zip2 v15.4s, v6.4s, v7.4s\n"  // G2H2G3H3

            "zip1 v0.2d, v8.2d, v10.2d\n"  // A0B0C0D0
            "zip2 v2.2d, v8.2d, v10.2d\n"  // A1B1C1D1

            "zip1 v4.2d, v9.2d, v11.2d\n"  // A2B2C2D2
            "zip2 v6.2d, v9.2d, v11.2d\n"  // A3B3C3D3

            "zip1 v1.2d, v12.2d, v14.2d\n"  // E0F0G0H0
            "zip2 v3.2d, v12.2d, v14.2d\n"  // E1F1G1H1

            "zip1 v5.2d, v13.2d, v15.2d\n"  // E2F2G2H2
            "zip2 v7.2d, v13.2d, v15.2d\n"  // E3F3G3H3

            "st1 {v0.4s,v1.4s,v2.4s,v3.4s},  [%[outptr]], #64\n"
            "st1 {v4.4s,v5.4s,v6.4s,v7.4s},  [%[outptr]], #64\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "memory");
}

template <typename T>
static inline void transpose_12x4_1_s(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      const T*& inptr4, const T*& inptr5,
                                      const T*& inptr6, const T*& inptr7,
                                      const T*& inptr8, const T*& inptr9,
                                      const T*& inptr10, const T*& inptr11,
                                      T* outptr) {
    static_assert(sizeof(T) == 4,
                  "transpose_12x4_1_s only support sizeof(T) == 4");
    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], 16\n"    // A0A1A2A3
            "ld1 {v1.4s},  [%[inptr1]], 16\n"    // B0B1B2B3
            "ld1 {v2.4s},  [%[inptr2]], 16\n"    // C0C1C2C3
            "ld1 {v3.4s},  [%[inptr3]], 16\n"    // D0D1D2D3
            "ld1 {v4.4s},  [%[inptr4]], 16\n"    // E0E1E2E3
            "ld1 {v5.4s},  [%[inptr5]], 16\n"    // F0F1F2F3
            "ld1 {v6.4s},  [%[inptr6]], 16\n"    // G0G1G2G3
            "ld1 {v7.4s},  [%[inptr7]], 16\n"    // H0H1H2H3
            "ld1 {v16.4s},  [%[inptr8]], 16\n"   // I0I1I2I3
            "ld1 {v17.4s},  [%[inptr9]], 16\n"   // J0J1J2J3
            "ld1 {v18.4s},  [%[inptr10]], 16\n"  // K0K1K2K3
            "ld1 {v19.4s},  [%[inptr11]], 16\n"  // L0L1L2L3

            "zip1 v8.4s, v0.4s, v1.4s\n"   // A0B0A1B1
            "zip2 v9.4s, v0.4s, v1.4s\n"   // A2B2A3B3
            "zip1 v10.4s, v2.4s, v3.4s\n"  // C0D0C1D1
            "zip2 v11.4s, v2.4s, v3.4s\n"  // C2D2C3D3

            "zip1 v12.4s, v4.4s, v5.4s\n"  // E0F0E1F1
            "zip2 v13.4s, v4.4s, v5.4s\n"  // E2F2E3F3
            "zip1 v14.4s, v6.4s, v7.4s\n"  // G0H0G1H1
            "zip2 v15.4s, v6.4s, v7.4s\n"  // G2H2G3H3

            "zip1 v20.4s, v16.4s, v17.4s\n"  // I0J0I1J1
            "zip2 v21.4s, v16.4s, v17.4s\n"  // I2J2I3J3
            "zip1 v22.4s, v18.4s, v19.4s\n"  // K0L0K1L1
            "zip2 v23.4s, v18.4s, v19.4s\n"  // K2L2K3L3

            "zip1 v0.2d, v8.2d, v10.2d\n"  // A0B0C0D0
            "zip2 v3.2d, v8.2d, v10.2d\n"  // A1B1C1D1

            "zip1 v6.2d, v9.2d, v11.2d\n"   // A2B2C2D2
            "zip2 v24.2d, v9.2d, v11.2d\n"  // A3B3C3D3

            "zip1 v1.2d, v12.2d, v14.2d\n"  // E0F0G0H0
            "zip2 v4.2d, v12.2d, v14.2d\n"  // E1F1G1H1

            "zip1 v7.2d, v13.2d, v15.2d\n"   // E2F2G2H2
            "zip2 v25.2d, v13.2d, v15.2d\n"  // E3F3G3H3

            "zip1 v2.2d, v20.2d, v22.2d\n"  // I0J0K0L0
            "zip2 v5.2d, v20.2d, v22.2d\n"  // I1J1K1L1

            "zip1 v8.2d, v21.2d, v23.2d\n"   // I2J2K2L2
            "zip2 v26.2d, v21.2d, v23.2d\n"  // I3J3K3L3

            "st1 {v0.4s,v1.4s,v2.4s},  [%[outptr]], #48\n"
            "st1 {v3.4s,v4.4s,v5.4s},  [%[outptr]], #48\n"
            "st1 {v6.4s,v7.4s,v8.4s},  [%[outptr]], #48\n"
            "st1 {v24.4s,v25.4s,v26.4s},  [%[outptr]], #48\n"
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [inptr8] "+r"(inptr8),
            [inptr9] "+r"(inptr9), [inptr10] "+r"(inptr10),
            [inptr11] "+r"(inptr11), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "v20", "v21", "v22", "v23", "v24", "v25", "v26", "memory");
}

template <typename T>
static inline void transpose_12x4_1_b(const T*& inptr0, const T*& inptr1,
                                      const T*& inptr2, const T*& inptr3,
                                      T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_12x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "ldr q0,  [%[inptr0]], #12\n"  // A1A2A3A4A5A6A7A8A9A10A11A12A13A14A15A16
            "ldr q1,  [%[inptr1]], #12\n"  // B1B2B3B4B5B6B7B8B9B10B11B12B13B14B15B16
            "ldr q2,  [%[inptr2]], #12\n"  // C1C2C3C4C5C6C7C8C9C10C11C12C13C14C15C16
            //! \warning the last inptr3 may less than 16bytes, so we should
            //! split read it
            "ldr d3,  [%[inptr3]], #8\n"  // D1D2D3D4D5D6D7D8D9D10D11D12D13D14D15D16
            "ldr w1, [%[inptr3]], #4\n"
            "ins v3.s[2], w1\n"

            "trn1 v4.16b, v0.16b, v1.16b\n"  // v4: A1B1A3B3....
            "trn2 v5.16b, v0.16b, v1.16b\n"  // v5: A2B2A4B4....
            "trn1 v6.16b, v2.16b, v3.16b\n"  // v6: C1D1C3D3....
            "trn2 v7.16b, v2.16b, v3.16b\n"  // v7: C2D2C4D4....

            "trn1 v8.8h, v4.8h, v6.8h\n"   // v8: A1B1C1D1A5B5C5D5...
            "trn2 v9.8h, v4.8h, v6.8h\n"   // v9: A3B3C3D3A7B7C7D7...
            "trn1 v10.8h, v5.8h, v7.8h\n"  // v10: A2B2C2D2A6B6C6D6...
            "trn2 v11.8h, v5.8h, v7.8h\n"  // v11: A4B4C4D4A8B8C8D8...

            //! ABCD=E then
            //! v8: E1E5E9E13 v10: E2E6E10E14 v9: E3E7E11E15 v11:
            //! E4E8E12E16
            "zip1 v12.4s, v8.4s, v10.4s\n"   // v12: E1E2E5E6
            "zip2 v13.4s, v8.4s, v10.4s\n"   // v13: E9E10E13E14
            "zip1 v14.4s, v9.4s, v11.4s\n"   // v14: E3E4E7E8
            "zip2 v15.4s, v9.4s, v11.4s\n"   // v15: E11E12E15E16
            "zip1 v17.2d, v12.2d, v14.2d\n"  // v17: E1E2E3E4
            "zip2 v18.2d, v12.2d, v14.2d\n"  // v18: E5E6E7E8
            "zip1 v19.2d, v13.2d, v15.2d\n"  // v19: E8E10E11E12
            "zip2 v20.2d, v13.2d, v15.2d\n"  // v19: E13E14E15E16

            "stp q17, q18, [%[outptr]], #32\n"
            "str q19, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "w1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
              "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
              "v19", "memory");
}

template <typename T>
static inline void transpose_8x4_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "ld1 {v0.d}[0],  [%[inptr0]], #8\n"  // A1A2A3A4A5A6A7A8
            "ld1 {v1.d}[0],  [%[inptr1]], #8\n"  // B1B2B3B4B5B6B7B8
            "ld1 {v0.d}[1],  [%[inptr2]], #8\n"  // C1C2C3C4C5C6C7C8
            "ld1 {v1.d}[1],  [%[inptr3]], #8\n"  // D1D2D3D4D5D6D7D8

            "zip1 v2.16b, v0.16b, v1.16b\n"  // A1B1A2B2A3B3A4B4A5B5A6B6A7B7A8B8
            "zip2 v3.16b, v0.16b, v1.16b\n"  // C1D1C2D2C3D3C4D4C5D5C6D6C7D7C8D8

            "zip1 v4.8h, v2.8h, v3.8h\n"  // A1B1C1D1A2B2C2D2A3B3C3D3A4B4C4D4
            "zip2 v5.8h, v2.8h, v3.8h\n"  // A5B5C5D5A6B6C6D6A7B7C7D7A8B8C8D8

            "st1 {v4.2d}, [%[outptr]], #16\n"
            "st1 {v5.2d}, [%[outptr]], #16\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "memory");
}

template <typename T>
static inline void transpose_4x8_1_b_with_shift(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     const T*& inptr4, const T*& inptr5,
                                     const T*& inptr6, const T*& inptr7,
                                     T*& outptr) {

    static int8x16_t shuffle_idx = {0, 4, 8,  12, 1, 5, 9,  13,
                                    2, 6, 10, 14, 3, 7, 11, 15};
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "ld1 {v0.s}[0],  [%[inptr0]], #4\n"  // A1A2A3A4
            "ld1 {v0.s}[1],  [%[inptr1]], #4\n"  // B1B2B3B4
            "ld1 {v0.s}[2],  [%[inptr2]], #4\n"  // C1C2C3C4
            "ld1 {v0.s}[3],  [%[inptr3]], #4\n"  // D1D2D3D4
            "ld1 {v1.s}[0],  [%[inptr4]], #4\n"  // E1E2E3E4
            "ld1 {v1.s}[1],  [%[inptr5]], #4\n"  // F1F2F3F4
            "ld1 {v1.s}[2],  [%[inptr6]], #4\n"  // G1G2G3G4
            "ld1 {v1.s}[3],  [%[inptr7]], #4\n"  // H1H2H3H4

            "tbl v2.16b, {v0.16b}, %[shuffle_idx].16b \n" // A1B1C1D1A2B2C2D2A3B3C3D3A4B4C4D4
            "tbl v3.16b, {v1.16b}, %[shuffle_idx].16b \n" // E1F1G1H1E2F2G2H2E3F3G3H3E4F4G4H4

            "zip1 v4.4s, v2.4s, v3.4s\n"  // A1B1C1D1E1F1G1H1 A2B2C2D2E2F2G2H2
            "zip2 v5.4s, v2.4s, v3.4s\n"  // A3B3C3D3E3F3G3H3 A4B4C4D4E4F4G4H4

            "shl  v6.16b, v4.16b,        #4\n"
            "sshr v7.16b, v4.16b,        #4\n"  // hig
            "sshr v8.16b, v6.16b,        #4\n"  // low
            "shl  v9.16b, v5.16b,        #4\n"
            "sshr v10.16b, v5.16b,        #4\n"  // hig
            "sshr v11.16b, v9.16b,        #4\n"  // low
            "zip1 v0.2d,v8.2d,v7.2d\n"
            "zip2 v1.2d,v8.2d,v7.2d\n"
            "zip1 v2.2d,v11.2d,v10.2d\n"
            "zip2 v3.2d,v11.2d,v10.2d\n"
            "st1 {v0.2d-v3.2d},[%[outptr]],#64\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [shuffle_idx]"+w"(shuffle_idx),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5","v6","v7","v8","v9","v10","v11","memory");
}
template <typename T>
static inline void transpose_8x8_1_b(const T*& inptr0, const T*& inptr1,
                                     const T*& inptr2, const T*& inptr3,
                                     const T*& inptr4, const T*& inptr5,
                                     const T*& inptr6, const T*& inptr7,
                                     T* outptr) {
    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x8_1_b only support uint8_t and int8_t");
    asm volatile(
            "ld1 {v0.8b},  [%[inptr0]], #8\n"  // A1A2A3A4A5A6A7A8
            "ld1 {v1.8b},  [%[inptr1]], #8\n"  // B1B2B3B4B5B6B7B8
            "ld1 {v2.8b},  [%[inptr2]], #8\n"  // C1C2C3C4C5C6C7C8
            "ld1 {v3.8b},  [%[inptr3]], #8\n"  // D1D2D3D4D5D6D7D8
            "ld1 {v4.8b},  [%[inptr4]], #8\n"  // E1E2E3E4E5E6E7E8
            "ld1 {v5.8b},  [%[inptr5]], #8\n"  // F1F2F3F4F5F6F7F8
            "ld1 {v6.8b},  [%[inptr6]], #8\n"  // G1G2G3G4G5G6G7G8
            "ld1 {v7.8b},  [%[inptr7]], #8\n"  // H1H2H3H4H5H6H7H8

            "zip1 v8.16b, v0.16b, v1.16b\n"   // A1B1A2B2A3B3A4B4
                                              // A5B5A6B6A7B7A8B8
            "zip1 v9.16b, v2.16b, v3.16b\n"   // C1D1C2D2C3D3C4D4
                                              // C5D5C6D6C7D7C8D8
            "zip1 v10.16b, v4.16b, v5.16b\n"  // E1F1E2F2E3F3E4F4
                                              // E5F5E6F6E7F7E8F8
            "zip1 v11.16b, v6.16b, v7.16b\n"  // G1H1G2H2G3H3G4H4
                                              // G5H5G6H6G7H7G8H8

            "zip1 v12.8h, v8.8h, v9.8h\n"    // A1B1C1D1A2B2C2D2
                                             // A3B3C3D3A4B4C4D4
            "zip1 v13.8h, v10.8h, v11.8h\n"  // E1F1G1H1E2F2G2H2
                                             // E3F3G3H3E4F4G4H4
            "zip2 v14.8h, v8.8h, v9.8h\n"    // A5B5C5D5A6B6C6D6
                                             // A7B7C7D7A8B8C8D8
            "zip2 v15.8h, v10.8h, v11.8h\n"  // E5F5G5H5E6F6G6H6
                                             // E7F7G7H7E8F8G8H8

            "zip1 v16.4s, v12.4s, v13.4s\n"  // A1B1C1D1E1F1G1H1
                                             // A2B2C2D2E2F2G2H2
            "zip1 v18.4s, v14.4s, v15.4s\n"  // A5B5C5D5E5F5G5H5
                                             // A6B6C6D6E6F6G6H6
            "zip2 v17.4s, v12.4s, v13.4s\n"  // A3B3C3D3E3F3G3H3
                                             // A4B4C4D4E4F4G4H4
            "zip2 v19.4s, v14.4s, v15.4s\n"  // A7B7C7D7E7F7G7H7
                                             // A8B8C8D8E8F8G8H8

            "st1 {v16.16b},  [%[outptr]], #16\n"  // A1B1C1D1E1F1G1H1
                                                  // A2B2C2D2E2F2G2H2
            "st1 {v17.16b},  [%[outptr]], #16\n"  // A3B3C3D3E3F3G3H3
                                                  // A4B4C4D4E4F4G4H4
            "st1 {v18.16b},  [%[outptr]], #16\n"  // A5B5C5D5E5F5G5H5
                                                  // A6B6C6D6E6F6G6H6
            "st1 {v19.16b},  [%[outptr]], #16\n"  // A7B7C7D7E7F7G7H7
                                                  // A8B8C8D8E8F8G8H8
            :
            [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1), [inptr2] "+r"(inptr2),
            [inptr3] "+r"(inptr3), [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
            [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7), [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
              "memory");
}

template <typename T>
static inline void transpose_4x16_1_b_helper(const T*& inptr0, const T*& inptr1,
                                             const T*& inptr2, const T*& inptr3,
                                             const T*& inptr4, const T*& inptr5,
                                             const T*& inptr6, const T*& inptr7,
                                             T* outptr) {
    static_assert(sizeof(T) == 1, "only support size == 1");
    static int8x16_t shuffle_idx = {0, 4, 8,  12, 1, 5, 9,  13,
                                    2, 6, 10, 14, 3, 7, 11, 15};
    asm volatile(
            "ld1 {v0.s}[0], [%[inptr0]], #4\n"
            "ld1 {v0.s}[1], [%[inptr1]], #4\n"
            "ld1 {v0.s}[2], [%[inptr2]], #4\n"
            "ld1 {v0.s}[3], [%[inptr3]], #4\n"
            "ld1 {v1.s}[0], [%[inptr4]], #4\n"
            "ld1 {v1.s}[1], [%[inptr5]], #4\n"
            "ld1 {v1.s}[2], [%[inptr6]], #4\n"
            "ld1 {v1.s}[3], [%[inptr7]], #4\n"

            "tbl v2.16b, {v0.16b}, %[shuffle_idx].16b\n"
            "tbl v3.16b, {v1.16b}, %[shuffle_idx].16b\n"

            "zip1 v4.4s, v2.4s, v3.4s\n"
            "zip2 v5.4s, v2.4s, v3.4s\n"

            "dup v6.2d, v4.d[1]\n"
            "dup v7.2d, v5.d[1]\n"

            "str d4, [%[outptr]], #16\n"
            "str d6, [%[outptr]], #16\n"
            "str d5, [%[outptr]], #16\n"
            "str d7, [%[outptr]], #16\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [inptr4] "+r"(inptr4), [inptr5] "+r"(inptr5),
              [inptr6] "+r"(inptr6), [inptr7] "+r"(inptr7),
              [outptr] "+r"(outptr), [shuffle_idx] "+w"(shuffle_idx)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory");
}

template <typename T>
static inline void transpose_4(const T*& inptr0, const T*& inptr1,
                               const T*& inptr2, const T*& inptr3, T* outptr,
                               int interleave, int size, T val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
    }
}

template <typename T>
static inline void transpose_8(const T*& inptr0, const T*& inptr1,
                               const T*& inptr2, const T*& inptr3,
                               const T*& inptr4, const T*& inptr5,
                               const T*& inptr6, const T*& inptr7, T* outptr,
                               int interleave, int size, T val = 0) {
    megdnn_assert(size <= interleave);
    int i = 0;
    for (; i < size; i++) {
        *outptr++ = *inptr0++;
        *outptr++ = *inptr1++;
        *outptr++ = *inptr2++;
        *outptr++ = *inptr3++;
        *outptr++ = *inptr4++;
        *outptr++ = *inptr5++;
        *outptr++ = *inptr6++;
        *outptr++ = *inptr7++;
    }
    for (; i < interleave; i++) {
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
        *outptr++ = val;
    }
}
/***************************** Transpose then interleave ********************/

//! pack form {1, 4(icb), 4(ic), 4(oc)} to {1, 1, 4(oc), 16(ic)}
template <typename T>
static inline void transpose_interleave_4x4_4_b(const T*& inptr0,
                                                const T*& inptr1,
                                                const T*& inptr2,
                                                const T*& inptr3, T* outptr,
                                                int stride = 64) {
    static_assert(sizeof(T) == 1,
                  "transpose_interleave_4x4_4_b only support sizeof(T) == 1");

    asm volatile(
            "ld4 {v0.16b, v1.16b, v2.16b, v3.16b},[%[inptr0]], 64\n"
            "ld4 {v4.16b, v5.16b, v6.16b, v7.16b},[%[inptr1]], 64\n"
            "ld4 {v8.16b, v9.16b, v10.16b, v11.16b},[%[inptr2]], 64\n"
            "ld4 {v12.16b, v13.16b, v14.16b, v15.16b},[%[inptr3]], 64\n"

            "st1 {v0.16b, v1.16b, v2.16b, v3.16b},[%[outptr]], %x[stride]\n"
            "st1 {v4.16b, v5.16b, v6.16b, v7.16b},[%[outptr]], %x[stride]\n"
            "st1 {v8.16b, v9.16b, v10.16b, v11.16b},[%[outptr]], %x[stride]\n"
            "st1 {v12.16b, v13.16b, v14.16b, v15.16b},[%[outptr]], %x[stride]\n"
            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [inptr2] "+r"(inptr2), [inptr3] "+r"(inptr3),
              [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
              "v11", "v12", "v14", "v15", "memory");
}

template <typename T>
static inline void transpose_interleave_1x4_4_b(const T*& inptr0, T* outptr,
                                                int stride = 64) {
    static_assert(sizeof(T) == 1,
                  "transpose_interleave_1x4_4_b only support sizeof(T) == 1");

    asm volatile(
            "ld4 {v0.16b, v1.16b, v2.16b, v3.16b},[%[inptr0]], 64\n"
            "st1 {v0.16b, v1.16b, v2.16b, v3.16b},[%[outptr]], %x[stride]\n"
            :
            [inptr0] "+r"(inptr0), [outptr] "+r"(outptr), [stride] "+r"(stride)
            :
            : "v0", "v1", "v2", "v3", "v4", "memory");
}

static inline void interleave_4x4_16x4_s8_s16(const int8_t* inptr0,
                                              const int8_t* inptr1,
                                              const int8_t* inptr2,
                                              const int8_t* inptr3,
                                              int16_t* outptr) {
    int8x16_t row0 = vld1q_s8(inptr0);
    int16x8_t row0_01 = vmovl_low_s8(row0);
    int16x8_t row0_23 = vmovl_high_s8(row0);
    int16x4_t row0_0 = vget_low_s16(row0_01);
    int16x4_t row0_1 = vget_high_s16(row0_01);
    int16x4_t row0_2 = vget_low_s16(row0_23);
    int16x4_t row0_3 = vget_high_s16(row0_23);

    int8x16_t row1 = vld1q_s8(inptr1);
    int16x8_t row1_01 = vmovl_low_s8(row1);
    int16x8_t row1_23 = vmovl_high_s8(row1);
    int16x4_t row1_0 = vget_low_s16(row1_01);
    int16x4_t row1_1 = vget_high_s16(row1_01);
    int16x4_t row1_2 = vget_low_s16(row1_23);
    int16x4_t row1_3 = vget_high_s16(row1_23);

    int8x16_t row2 = vld1q_s8(inptr2);
    int16x8_t row2_01 = vmovl_low_s8(row2);
    int16x8_t row2_23 = vmovl_high_s8(row2);
    int16x4_t row2_0 = vget_low_s16(row2_01);
    int16x4_t row2_1 = vget_high_s16(row2_01);
    int16x4_t row2_2 = vget_low_s16(row2_23);
    int16x4_t row2_3 = vget_high_s16(row2_23);

    int8x16_t row3 = vld1q_s8(inptr3);
    int16x8_t row3_01 = vmovl_low_s8(row3);
    int16x8_t row3_23 = vmovl_high_s8(row3);
    int16x4_t row3_0 = vget_low_s16(row3_01);
    int16x4_t row3_1 = vget_high_s16(row3_01);
    int16x4_t row3_2 = vget_low_s16(row3_23);
    int16x4_t row3_3 = vget_high_s16(row3_23);

    vst1_s16(outptr, row0_0);
    vst1_s16(outptr + 1 * 4, row1_0);
    vst1_s16(outptr + 2 * 4, row2_0);
    vst1_s16(outptr + 3 * 4, row3_0);
    vst1_s16(outptr + 4 * 4, row0_1);
    vst1_s16(outptr + 5 * 4, row1_1);
    vst1_s16(outptr + 6 * 4, row2_1);
    vst1_s16(outptr + 7 * 4, row3_1);
    vst1_s16(outptr + 8 * 4, row0_2);
    vst1_s16(outptr + 9 * 4, row1_2);
    vst1_s16(outptr + 10 * 4, row2_2);
    vst1_s16(outptr + 11 * 4, row3_2);
    vst1_s16(outptr + 12 * 4, row0_3);
    vst1_s16(outptr + 13 * 4, row1_3);
    vst1_s16(outptr + 14 * 4, row2_3);
    vst1_s16(outptr + 15 * 4, row3_3);
};
static inline void interleave_4x4_8x4_s8_s16(const int8_t* inptr0,
                                             const int8_t* inptr1,
                                             int16_t* outptr) {
    int8x16_t row0 = vld1q_s8(inptr0);
    int16x8_t row0_01 = vmovl_low_s8(row0);
    int16x8_t row0_23 = vmovl_high_s8(row0);
    int16x4_t row0_0 = vget_low_s16(row0_01);
    int16x4_t row0_1 = vget_high_s16(row0_01);
    int16x4_t row0_2 = vget_low_s16(row0_23);
    int16x4_t row0_3 = vget_high_s16(row0_23);

    int8x16_t row1 = vld1q_s8(inptr1);
    int16x8_t row1_01 = vmovl_low_s8(row1);
    int16x8_t row1_23 = vmovl_high_s8(row1);
    int16x4_t row1_0 = vget_low_s16(row1_01);
    int16x4_t row1_1 = vget_high_s16(row1_01);
    int16x4_t row1_2 = vget_low_s16(row1_23);
    int16x4_t row1_3 = vget_high_s16(row1_23);

    vst1_s16(outptr, row0_0);
    vst1_s16(outptr + 1 * 4, row1_0);
    vst1_s16(outptr + 2 * 4, row0_1);
    vst1_s16(outptr + 3 * 4, row1_1);
    vst1_s16(outptr + 4 * 4, row0_2);
    vst1_s16(outptr + 5 * 4, row1_2);
    vst1_s16(outptr + 6 * 4, row0_3);
    vst1_s16(outptr + 7 * 4, row1_3);
};

static inline void memcpy_s8_s16(const int8_t* inptr, int16_t* outptr,
                                 int count) {
    for (; count >= 32; count -= 32) {
        int8x8_t in0 = vld1_s8(inptr);
        int8x8_t in1 = vld1_s8(inptr + 1 * 8);
        int8x8_t in2 = vld1_s8(inptr + 2 * 8);
        int8x8_t in3 = vld1_s8(inptr + 3 * 8);
        vst1q_s16(outptr, vmovl_s8(in0));
        vst1q_s16(outptr + 1 * 8, vmovl_s8(in1));
        vst1q_s16(outptr + 2 * 8, vmovl_s8(in2));
        vst1q_s16(outptr + 3 * 8, vmovl_s8(in3));
        inptr += 32;
        outptr += 32;
    }
    for (; count >= 8; count -= 8) {
        int8x8_t in0 = vld1_s8(inptr);
        vst1q_s16(outptr, vmovl_s8(in0));
        inptr += 8;
        outptr += 8;
    }
    for (; count > 0; --count) {
        *outptr++ = (int16_t)(*inptr++);
    }
}

static inline void transpos_12x4_s8(const int8_t* inptr0, int8_t* outptr) {
    static const uint8_t src_idx_buffer[16] = {0, 4, 8,  12, 1, 5, 9,  13,
                                               2, 6, 10, 14, 3, 7, 11, 15};
    static const uint8x16_t vtbl = vld1q_u8(&src_idx_buffer[0]);
    int8x8x4_t input = vld4_s8(inptr0);
    int8x16_t input2 = vqtbl1q_s8(vld1q_s8(inptr0 + 4 * 8), vtbl);

    vst1_s8(outptr, input.val[0]);
    vst1q_lane_s32(reinterpret_cast<int32_t*>(outptr + 8),
                   vreinterpretq_s32_s8(input2), 0);
    vst1_s8(outptr + 1 * 12, input.val[1]);
    vst1q_lane_s32(reinterpret_cast<int32_t*>(outptr + 1 * 12 + 8),
                   vreinterpretq_s32_s8(input2), 1);
    vst1_s8(outptr + 2 * 12, input.val[2]);
    vst1q_lane_s32(reinterpret_cast<int32_t*>(outptr + 2 * 12 + 8),
                   vreinterpretq_s32_s8(input2), 2);
    vst1_s8(outptr + 3 * 12, input.val[3]);
    vst1q_lane_s32(reinterpret_cast<int32_t*>(outptr + 3 * 12 + 8),
                   vreinterpretq_s32_s8(input2), 3);
}


template <typename T>
static inline void interleave_8x8_mk4_b(const T*& inptr0, const T*& inptr1,
                                     T*& outptr) {

    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "ld1 {v0.4s},  [%[inptr0]], #16\n"
            "ld1 {v1.4s},  [%[inptr1]], #16\n"
            "ld1 {v2.4s},  [%[inptr0]], #16\n"
            "ld1 {v3.4s},  [%[inptr1]], #16\n"

            "zip1 v4.4s, v0.4s, v1.4s \n"
            "zip2 v5.4s, v0.4s, v1.4s \n"

            "zip1 v6.4s, v2.4s, v3.4s\n"
            "zip2 v7.4s, v2.4s, v3.4s\n"

            "st1 {v4.4s},[%[outptr]],#16\n"
            "st1 {v5.4s},[%[outptr]],#16\n"
            "st1 {v6.4s},[%[outptr]],#16\n"
            "st1 {v7.4s},[%[outptr]],#16\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5","v6","v7","memory");
}

template <typename T>
static inline void transpose_8x8_mk4_b(const T*& inptr0, const T*& inptr1,
                                     T* outptr) {

    static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value,
            "transpose_8x4_1_b only support uint8_t and int8_t");
    asm volatile(
            "ld4 {v0.8b-v3.8b},  [%[inptr0]], #32\n"
            "ld4 {v4.8b-v7.8b},  [%[inptr1]], #32\n"
            "st1 {v0.2s},[%[outptr]],#8\n"
            "st1 {v1.2s},[%[outptr]],#8\n"
            "st1 {v2.2s},[%[outptr]],#8\n"
            "st1 {v3.2s},[%[outptr]],#8\n"
            "st1 {v4.2s},[%[outptr]],#8\n"
            "st1 {v5.2s},[%[outptr]],#8\n"
            "st1 {v6.2s},[%[outptr]],#8\n"
            "st1 {v7.2s},[%[outptr]],#8\n"

            : [inptr0] "+r"(inptr0), [inptr1] "+r"(inptr1),
              [outptr] "+r"(outptr)
            :
            : "v0", "v1", "v2", "v3", "v4", "v5","v6","v7","memory");
}

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
