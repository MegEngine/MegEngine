/**
 * \file dnn/src/aarch64/conv_bias/fp16/stride2_kern.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <cstddef>
#include "src/arm_common/simd_macro/neon_helper_fp16.h"
#include "src/common/utils.h"

namespace megdnn {
namespace aarch64 {
namespace fp16 {
namespace conv_stride2 {

static void do_conv_2x2_stride2(const __fp16* src, const __fp16* filter,
                                __fp16* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 3;
    size_t mod4_left = width & 3;

    rep(ic, IC) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;

        const __fp16* k0 = filter;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        rep(h, OH) {
            asm volatile(
                    "dup v28.8h, %5.h[0] \n"
                    "dup v29.8h, %5.h[1] \n"
                    "dup v30.8h, %5.h[2] \n"
                    "dup v31.8h, %5.h[3] \n"
                    "cmp %4, #2 \n"
                    "mov x1, xzr \n"
                    // mod4_left == 3
                    "bgt 0f \n"
                    // mod4_left == 2
                    "beq 1f \n"
                    "cmp %4, #1 \n"
                    // mod4_left == 1
                    "beq 2f \n"
                    // mod4_left == 0
                    "b 3f \n"

                    // mod4_left == 3
                    "0: \n"
                    "ld1 {v0.8h, v1.8h, v2.8h}, [%1] \n"

                    "ld2 {v3.8h, v4.8h}, [%2], #32 \n"
                    "ld2 {v9.8h, v10.8h}, [%3], #32 \n"
                    "ld2 {v5.8h, v6.8h}, [%2], #32 \n"
                    "ld2 {v11.8h, v12.8h}, [%3], #32 \n"
                    "ld2 {v7.8h, v8.8h}, [%2], #32 \n"
                    "ld2 {v13.8h, v14.8h}, [%3], #32 \n"
                    "fmla v0.8h, v3.8h, v28.8h \n"
                    "fmla v1.8h, v5.8h, v28.8h \n"
                    "fmla v2.8h, v7.8h, v28.8h \n"
                    "fmla v0.8h, v4.8h, v29.8h \n"
                    "fmla v1.8h, v6.8h, v29.8h \n"
                    "fmla v2.8h, v8.8h, v29.8h \n"

                    "fmla v0.8h, v9.8h, v30.8h \n"
                    "fmla v1.8h, v11.8h, v30.8h \n"
                    "fmla v2.8h, v13.8h, v30.8h \n"
                    "fmla v0.8h, v10.8h, v31.8h \n"
                    "fmla v1.8h, v12.8h, v31.8h \n"
                    "fmla v2.8h, v14.8h, v31.8h \n"

                    "add x1, x1, #3 \n"
                    "st1 {v0.8h, v1.8h, v2.8h}, [%1], #48 \n"
                    "b 3f \n"

                    // mod4_left == 2
                    "1: \n"
                    "ld1 {v0.8h, v1.8h}, [%1] \n"

                    "ld2 {v2.8h, v3.8h}, [%2], #32 \n"
                    "ld2 {v6.8h, v7.8h}, [%3], #32 \n"
                    "ld2 {v4.8h, v5.8h}, [%2], #32 \n"
                    "ld2 {v8.8h, v9.8h}, [%3], #32 \n"
                    "fmla v0.8h, v2.8h, v28.8h \n"
                    "fmla v1.8h, v4.8h, v28.8h \n"
                    "fmla v0.8h, v3.8h, v29.8h \n"
                    "fmla v1.8h, v5.8h, v29.8h \n"

                    "fmla v0.8h, v6.8h, v30.8h \n"
                    "fmla v1.8h, v8.8h, v30.8h \n"
                    "fmla v0.8h, v7.8h, v31.8h \n"
                    "fmla v1.8h, v9.8h, v31.8h \n"

                    "add x1, x1, #2 \n"
                    "st1 {v0.8h, v1.8h}, [%1], #32 \n"
                    "b 3f \n"

                    // mod4_left == 1
                    "2: \n"
                    "ld1 {v0.8h}, [%1] \n"

                    "ld2 {v1.8h, v2.8h}, [%2], #32 \n"
                    "ld2 {v3.8h, v4.8h}, [%3], #32 \n"
                    "fmla v0.8h, v1.8h, v28.8h \n"
                    "fmla v0.8h, v2.8h, v29.8h \n"

                    "fmla v0.8h, v3.8h, v30.8h \n"
                    "fmla v0.8h, v4.8h, v31.8h \n"

                    "add x1, x1, #1 \n"
                    "st1 {v0.8h}, [%1], #16 \n"
                    "b 3f \n"

                    // mod4_left == 0
                    "3: \n"
                    "cmp %0, x1 \n"
                    "beq 5f \n"
                    "4: \n"
                    "ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [%1] \n"

                    "ld2 {v4.8h, v5.8h}, [%2], #32 \n"
                    "ld2 {v12.8h, v13.8h}, [%3], #32 \n"
                    "ld2 {v6.8h, v7.8h}, [%2], #32 \n"
                    "ld2 {v14.8h, v15.8h}, [%3], #32 \n"
                    "ld2 {v8.8h, v9.8h}, [%2], #32 \n"
                    "ld2 {v16.8h, v17.8h}, [%3], #32 \n"
                    "ld2 {v10.8h, v11.8h}, [%2], #32 \n"
                    "ld2 {v18.8h, v19.8h}, [%3], #32 \n"
                    "fmla v0.8h, v4.8h, v28.8h \n"
                    "fmla v1.8h, v6.8h, v28.8h \n"
                    "fmla v2.8h, v8.8h, v28.8h \n"
                    "fmla v3.8h, v10.8h, v28.8h \n"
                    "fmla v0.8h, v5.8h, v29.8h \n"
                    "fmla v1.8h, v7.8h, v29.8h \n"
                    "fmla v2.8h, v9.8h, v29.8h \n"
                    "fmla v3.8h, v11.8h, v29.8h \n"

                    "fmla v0.8h, v12.8h, v30.8h \n"
                    "fmla v1.8h, v14.8h, v30.8h \n"
                    "fmla v2.8h, v16.8h, v30.8h \n"
                    "fmla v3.8h, v18.8h, v30.8h \n"
                    "fmla v0.8h, v13.8h, v31.8h \n"
                    "fmla v1.8h, v15.8h, v31.8h \n"
                    "fmla v2.8h, v17.8h, v31.8h \n"
                    "fmla v3.8h, v19.8h, v31.8h \n"

                    "add x1, x1, #4 \n"
                    "cmp %0, x1 \n"
                    "st1 {v0.8h, v1.8h, v2.8h, v3.8h}, [%1], #64 \n"
                    "bne 4b \n"

                    "5: \n"
                    : "+r"(width), "+r"(outptr), "+r"(r0), "+r"(r1)
                    : "r"(mod4_left), "w"(_k0123)
                    : "cc", "memory", "x1", "v0", "v1", "v2", "v3", "v4", "v5",
                      "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                      "v15", "v16", "v17", "v18", "v19", "v28", "v29", "v30",
                      "v31");

            r0 += tail_step;
            r1 += tail_step;
        }

        filter += 4;
    }
}

static void do_conv_3x3_stride2(const __fp16* src, const __fp16* filter,
                                __fp16* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 3;
    size_t mod3_left = width % 3;

    rep(ic, IC) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;
        const __fp16* r2 = src_ptr + IW * 2;

        const __fp16* k0 = filter;
        const __fp16* k1 = filter + 3;
        const __fp16* k2 = filter + 5;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        MEGDNN_SIMD_TYPE _k3456 = MEGDNN_SIMD_LOADU(k1);
        MEGDNN_SIMD_TYPE _k5678 = MEGDNN_SIMD_LOADU(k2);
        rep(h, OH) {
            asm volatile(
                    "dup v21.8h, %6.h[0] \n"
                    "dup v22.8h, %6.h[1] \n"
                    "dup v23.8h, %6.h[2] \n"
                    "dup v24.8h, %6.h[3] \n"
                    "dup v25.8h, %7.h[1] \n"
                    "dup v26.8h, %7.h[2] \n"
                    "dup v27.8h, %7.h[3] \n"
                    "dup v28.8h, %8.h[2] \n"
                    "dup v29.8h, %8.h[3] \n"
                    "cmp %5, #1 \n"
                    "mov x1, xzr \n"
                    "bgt 0f \n"  // mod3_left == 2
                    "beq 1f \n"  // mod3_left == 1
                    "blt 2f \n"  // mod3_left == 0

                    "0:  \n"
                    "ld1 {v0.8h, v1.8h}, [%1] \n"

                    "ld2 {v2.8h, v3.8h}, [%2], #32 \n"
                    "ld2 {v9.8h, v10.8h}, [%3], #32 \n"
                    "ld2 {v4.8h, v5.8h}, [%2], #32 \n"
                    "ld2 {v11.8h, v12.8h}, [%3], #32 \n"
                    "fmla v0.8h, v2.8h, v21.8h \n"
                    "fmla v1.8h, v4.8h, v21.8h \n"
                    "fmla v0.8h, v3.8h, v22.8h \n"
                    "fmla v1.8h, v5.8h, v22.8h \n"
                    "ld1 {v6.8h}, [%2] \n"
                    "ld1 {v13.8h}, [%3] \n"
                    "ext v7.16b, v2.16b, v4.16b, #2 \n"
                    "ext v8.16b, v4.16b, v6.16b, #2 \n"
                    "fmla v0.8h, v7.8h, v23.8h \n"
                    "fmla v1.8h, v8.8h, v23.8h \n"

                    "ld2 {v2.8h, v3.8h}, [%4], #32 \n"
                    "fmla v0.8h, v9.8h, v24.8h \n"
                    "fmla v1.8h, v11.8h, v24.8h \n"
                    "fmla v0.8h, v10.8h, v25.8h \n"
                    "fmla v1.8h, v12.8h, v25.8h \n"
                    "ld2 {v4.8h, v5.8h}, [%4], #32 \n"
                    "ext v14.16b, v9.16b, v11.16b, #2 \n"
                    "ext v15.16b, v11.16b, v13.16b, #2 \n"
                    "fmla v0.8h, v14.8h, v26.8h \n"
                    "fmla v1.8h, v15.8h, v26.8h \n"

                    "ld1 {v6.8h}, [%4] \n"
                    "fmla v0.8h, v2.8h, v27.8h \n"
                    "fmla v1.8h, v4.8h, v27.8h \n"
                    "fmla v0.8h, v3.8h, v28.8h \n"
                    "fmla v1.8h, v5.8h, v28.8h \n"
                    "ext v7.16b, v2.16b, v4.16b, #2 \n"
                    "ext v8.16b, v4.16b, v6.16b, #2 \n"
                    "fmla v0.8h, v7.8h, v29.8h \n"
                    "fmla v1.8h, v8.8h, v29.8h \n"

                    "add x1, x1, #2 \n"
                    "cmp %0, x1 \n"

                    "st1 {v0.8h, v1.8h}, [%1], #32 \n"
                    "bne 2f \n"  // if width != 2 jump to 2
                    "b 3f \n"    // jump end

                    "1:           \n"
                    "ld1 {v0.8h}, [%1] \n"
                    "ld2 {v1.8h, v2.8h}, [%2], #32 \n"

                    "ld2 {v5.8h, v6.8h}, [%3], #32 \n"
                    "ld1 {v3.8h}, [%2] \n"
                    "fmla v0.8h, v1.8h, v21.8h \n"
                    "ext v7.16b, v1.16b, v3.16b, #2 \n"
                    "fmla v0.8h, v2.8h, v22.8h \n"
                    "ld1 {v1.8h}, [%3] \n"
                    "fmla v0.8h, v7.8h, v23.8h \n"
                    "ld2 {v3.8h, v4.8h}, [%4], #32 \n"

                    "fmla v0.8h, v5.8h, v24.8h \n"
                    "ext v7.16b, v5.16b, v1.16b, #2 \n"
                    "fmla v0.8h, v6.8h, v25.8h \n"
                    "ld1 {v5.8h}, [%4] \n"
                    "fmla v0.8h, v7.8h, v26.8h \n"

                    "fmla v0.8h, v3.8h, v27.8h \n"
                    "fmla v0.8h, v4.8h, v28.8h \n"
                    "ext v7.16b, v3.16b, v5.16b, #2 \n"
                    "fmla v0.8h, v7.8h, v29.8h \n"

                    "st1 {v0.8h}, [%1], #16 \n"

                    "add x1, x1, #1 \n"
                    "cmp %0, x1 \n"
                    "beq 3f \n"

                    "2: \n"
                    "ld1 {v0.8h, v1.8h, v2.8h}, [%1] \n"

                    "ld2 {v3.8h, v4.8h}, [%2], #32 \n"
                    "ld2 {v11.8h, v12.8h}, [%3], #32 \n"
                    "ld2 {v5.8h, v6.8h}, [%2], #32 \n"
                    "ld2 {v13.8h, v14.8h}, [%3], #32 \n"
                    "ld2 {v7.8h, v8.8h}, [%2], #32 \n"
                    "ld2 {v15.8h, v16.8h}, [%3], #32 \n"
                    "fmla v0.8h, v3.8h, v21.8h \n"
                    "fmla v1.8h, v5.8h, v21.8h \n"
                    "fmla v2.8h, v7.8h, v21.8h \n"
                    "ld1 {v9.8h}, [%2] \n"
                    "ld1 {v17.8h}, [%3] \n"
                    "fmla v0.8h, v4.8h, v22.8h \n"
                    "fmla v1.8h, v6.8h, v22.8h \n"
                    "fmla v2.8h, v8.8h, v22.8h \n"
                    "ext v10.16b, v3.16b, v5.16b, #2 \n"
                    "ext v4.16b, v5.16b, v7.16b, #2 \n"
                    "ext v6.16b, v7.16b, v9.16b, #2 \n"
                    "fmla v0.8h, v10.8h, v23.8h \n"
                    "fmla v1.8h, v4.8h, v23.8h \n"
                    "fmla v2.8h, v6.8h, v23.8h \n"

                    "ld2 {v3.8h, v4.8h}, [%4], #32 \n"
                    "fmla v0.8h, v11.8h, v24.8h \n"
                    "fmla v1.8h, v13.8h, v24.8h \n"
                    "fmla v2.8h, v15.8h, v24.8h \n"
                    "ld2 {v5.8h, v6.8h}, [%4], #32 \n"
                    "fmla v0.8h, v12.8h, v25.8h \n"
                    "fmla v1.8h, v14.8h, v25.8h \n"
                    "fmla v2.8h, v16.8h, v25.8h \n"
                    "ld2 {v7.8h, v8.8h}, [%4], #32 \n"
                    "ext v18.16b, v11.16b, v13.16b, #2 \n"
                    "ext v12.16b, v13.16b, v15.16b, #2 \n"
                    "ext v14.16b, v15.16b, v17.16b, #2 \n"
                    "fmla v0.8h, v18.8h, v26.8h \n"
                    "fmla v1.8h, v12.8h, v26.8h \n"
                    "fmla v2.8h, v14.8h, v26.8h \n"

                    "ld1 {v9.8h}, [%4] \n"
                    "fmla v0.8h, v3.8h, v27.8h \n"
                    "fmla v1.8h, v5.8h, v27.8h \n"
                    "fmla v2.8h, v7.8h, v27.8h \n"
                    "fmla v0.8h, v4.8h, v28.8h \n"
                    "fmla v1.8h, v6.8h, v28.8h \n"
                    "fmla v2.8h, v8.8h, v28.8h \n"
                    "ext v10.16b, v3.16b, v5.16b, #2 \n"
                    "ext v4.16b, v5.16b, v7.16b, #2 \n"
                    "ext v6.16b, v7.16b, v9.16b, #2 \n"
                    "fmla v0.8h, v10.8h, v29.8h \n"
                    "fmla v1.8h, v4.8h, v29.8h \n"
                    "fmla v2.8h, v6.8h, v29.8h \n"

                    "add x1, x1, #3 \n"
                    "cmp %0, x1 \n"

                    "st1 {v0.8h, v1.8h, v2.8h}, [%1], #48 \n"
                    "bne 2b \n"  // if
                    "3: \n"
                    : "+r"(width), "+r"(outptr), "+r"(r0), "+r"(r1), "+r"(r2)
                    : "r"(mod3_left), "w"(_k0123), "w"(_k3456), "w"(_k5678)
                    : "cc", "memory", "x1", "v0", "v1", "v2", "v3", "v4", "v5",
                      "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                      "v15", "v16", "v17", "v18", "v21", "v22", "v23", "v24",
                      "v25", "v26", "v27", "v28", "v29");

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
        }

        filter += 9;
    }
}

static void do_conv_5x5_stride2(const __fp16* src, const __fp16* filter,
                                __fp16* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 3;
    size_t mod2_left = width & 1;

    rep(ic, IC) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;
        const __fp16* r2 = src_ptr + IW * 2;
        const __fp16* r3 = src_ptr + IW * 3;
        const __fp16* r4 = src_ptr + IW * 4;

        register MEGDNN_SIMD_TYPE _k0123 asm("v0") = MEGDNN_SIMD_LOADU(filter);
        register MEGDNN_SIMD_TYPE _k4567 asm("v1") =
                MEGDNN_SIMD_LOADU(filter + 4);
        register MEGDNN_SIMD_TYPE _k891011 asm("v2") =
                MEGDNN_SIMD_LOADU(filter + 8);
        register MEGDNN_SIMD_TYPE _k12131415 asm("v3") =
                MEGDNN_SIMD_LOADU(filter + 12);
        register MEGDNN_SIMD_TYPE _k16171819 asm("v4") =
                MEGDNN_SIMD_LOADU(filter + 16);
        register MEGDNN_SIMD_TYPE _k20212223 asm("v5") =
                MEGDNN_SIMD_LOADU(filter + 20);
        register MEGDNN_SIMD_TYPE _k24242424 asm("v6") =
                MEGDNN_SIMD_SET1(filter[24]);

        for (size_t i = 0; i < OH; i++) {
            asm volatile(
                    "cmp %14, #0 \n"
                    "mov x1, xzr \n"
                    "beq 1f \n"

                    // mod2_left == 1
                    "0: \n"
                    "ld1 {v7.8h}, [%1] \n"

                    // v8.8h: 0 2 4 6 v9.8h: 1 3 5 7
                    "ld2 {v8.8h, v9.8h}, [%2], #32 \n"
                    "ld2 {v15.8h, v16.8h}, [%3], #32 \n"
                    "fmla v7.8h, v8.8h, %7.h[0] \n"
                    "fmla v7.8h, v9.8h, %7.h[1] \n"
                    "ld2 {v10.8h, v11.8h}, [%2] \n"
                    "ld2 {v17.8h, v18.8h}, [%3] \n"
                    // v12.8h: 2 4 6 8
                    "ext v12.16b, v8.16b, v10.16b, #2 \n"
                    // v13.8h: 3 5 7 9
                    "ext v13.16b, v9.16b, v11.16b, #2 \n"
                    "fmla v7.8h, v12.8h, %7.h[2] \n"
                    "fmla v7.8h, v13.8h, %7.h[3] \n"
                    // v14.8h: 4 6 8 10
                    "ext v14.16b, v8.16b, v10.16b, #4 \n"
                    "fmla v7.8h, v14.8h, %8.h[0] \n"

                    "ld2 {v8.8h, v9.8h}, [%4], #32 \n"
                    "fmla v7.8h, v15.8h, %8.h[1] \n"
                    "fmla v7.8h, v16.8h, %8.h[2] \n"
                    "ld2 {v10.8h, v11.8h}, [%4] \n"
                    "ext v19.16b, v15.16b, v17.16b, #2 \n"
                    "ext v20.16b, v16.16b, v18.16b, #2 \n"
                    "fmla v7.8h, v19.8h, %8.h[3] \n"
                    "fmla v7.8h, v20.8h, %9.h[0] \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "fmla v7.8h, v21.8h, %9.h[1] \n"

                    "ld2 {v15.8h, v16.8h}, [%5], #32 \n"
                    "fmla v7.8h, v8.8h, %9.h[2] \n"
                    "fmla v7.8h, v9.8h, %9.h[3] \n"
                    "ld2 {v17.8h, v18.8h}, [%5] \n"
                    "ext v12.16b, v8.16b, v10.16b, #2 \n"
                    "ext v13.16b, v9.16b, v11.16b, #2 \n"
                    "fmla v7.8h, v12.8h, %10.h[0] \n"
                    "fmla v7.8h, v13.8h, %10.h[1] \n"
                    "ext v14.16b, v8.16b, v10.16b, #4 \n"
                    "fmla v7.8h, v14.8h, %10.h[2] \n"

                    "ld2 {v8.8h, v9.8h}, [%6], #32 \n"
                    "fmla v7.8h, v15.8h, %10.h[3] \n"
                    "fmla v7.8h, v16.8h, %11.h[0] \n"
                    "ld2 {v10.8h, v11.8h}, [%6] \n"
                    "ext v19.16b, v15.16b, v17.16b, #2 \n"
                    "ext v20.16b, v16.16b, v18.16b, #2 \n"
                    "fmla v7.8h, v19.8h, %11.h[1] \n"
                    "fmla v7.8h, v20.8h, %11.h[2] \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "fmla v7.8h, v21.8h, %11.h[3] \n"

                    "fmla v7.8h, v8.8h, %12.h[0] \n"
                    "fmla v7.8h, v9.8h, %12.h[1] \n"
                    "ext v12.16b, v8.16b, v10.16b, #2 \n"
                    "ext v13.16b, v9.16b, v11.16b, #2 \n"
                    "fmla v7.8h, v12.8h, %12.h[2] \n"
                    "fmla v7.8h, v13.8h, %12.h[3] \n"
                    "ext v14.16b, v8.16b, v10.16b, #4 \n"
                    "fmla v7.8h, v14.8h, %13.h[0] \n"

                    "add x1, x1, #1 \n"
                    "st1 {v7.8h}, [%1], #16 \n"

                    "1: \n"
                    "cmp %0, x1 \n"
                    "beq 3f \n"

                    // mod2_left == 0
                    "2: \n"
                    "ld1 {v7.8h, v8.8h}, [%1] \n"

                    // v9.8h: 0 2 4 6 v10.8h: 1 3 5 7
                    "ld2 {v9.8h, v10.8h}, [%2], #32 \n"
                    "ld2 {v21.8h, v22.8h}, [%3], #32 \n"
                    // v11.8h: 8 10 12 14 v12.8h: 9 11 13 15
                    "ld2 {v11.8h, v12.8h}, [%2], #32 \n"
                    "ld2 {v23.8h, v24.8h}, [%3], #32 \n"
                    // v13.8h: 16 18 20 22 v14.8h: 17 19 21 23
                    "ld2 {v13.8h, v14.8h}, [%2] \n"
                    "ld2 {v25.8h, v26.8h}, [%3] \n"
                    // v15.8h: 2 4 6 8
                    "ext v15.16b, v9.16b, v11.16b, #2 \n"
                    // v16.8h: 3 5 7 9
                    "ext v16.16b, v10.16b, v12.16b, #2 \n"
                    // v17.8h: 4 6 8 10
                    "ext v17.16b, v9.16b, v11.16b, #4 \n"
                    // v18.8h: 10 12 14 16
                    "ext v18.16b, v11.16b, v13.16b, #2 \n"
                    // v19.8h: 11 13 15 17
                    "ext v19.16b, v12.16b, v14.16b, #2 \n"
                    // v20.8h: 12 14 16 18
                    "ext v20.16b, v11.16b, v13.16b, #4 \n"
                    "fmla v7.8h, v9.8h, %7.h[0] \n"
                    "fmla v7.8h, v10.8h, %7.h[1] \n"
                    "fmla v7.8h, v15.8h, %7.h[2] \n"
                    "fmla v7.8h, v16.8h, %7.h[3] \n"
                    "fmla v7.8h, v17.8h, %8.h[0] \n"
                    "fmla v8.8h, v11.8h, %7.h[0] \n"
                    "fmla v8.8h, v12.8h, %7.h[1] \n"
                    "fmla v8.8h, v18.8h, %7.h[2] \n"
                    "fmla v8.8h, v19.8h, %7.h[3] \n"
                    "fmla v8.8h, v20.8h, %8.h[0] \n"

                    "ld2 {v9.8h, v10.8h}, [%4], #32 \n"
                    "ext v27.16b, v21.16b, v23.16b, #2 \n"
                    "ext v28.16b, v22.16b, v24.16b, #2 \n"
                    "ext v29.16b, v21.16b, v23.16b, #4 \n"
                    "fmla v7.8h, v21.8h, %8.h[1] \n"
                    "fmla v7.8h, v22.8h, %8.h[2] \n"
                    "fmla v7.8h, v27.8h, %8.h[3] \n"
                    "fmla v7.8h, v28.8h, %9.h[0] \n"
                    "fmla v7.8h, v29.8h, %9.h[1] \n"
                    "ld2 {v11.8h, v12.8h}, [%4], #32 \n"
                    "ext v30.16b, v23.16b, v25.16b, #2 \n"
                    "ext v31.16b, v24.16b, v26.16b, #2 \n"
                    "ext v21.16b, v23.16b, v25.16b, #4 \n"
                    "ld2 {v13.8h, v14.8h}, [%4] \n"
                    "fmla v8.8h, v23.8h, %8.h[1] \n"
                    "fmla v8.8h, v24.8h, %8.h[2] \n"
                    "fmla v8.8h, v30.8h, %8.h[3] \n"
                    "fmla v8.8h, v31.8h, %9.h[0] \n"
                    "fmla v8.8h, v21.8h, %9.h[1] \n"

                    "ld2 {v21.8h, v22.8h}, [%5], #32 \n"
                    "ext v15.16b, v9.16b, v11.16b, #2 \n"
                    "ext v16.16b, v10.16b, v12.16b, #2 \n"
                    "ext v17.16b, v9.16b, v11.16b, #4 \n"
                    "ext v18.16b, v11.16b, v13.16b, #2 \n"
                    "ext v19.16b, v12.16b, v14.16b, #2 \n"
                    "ext v20.16b, v11.16b, v13.16b, #4 \n"
                    "ld2 {v23.8h, v24.8h}, [%5], #32 \n"
                    "fmla v7.8h, v9.8h, %9.h[2] \n"
                    "fmla v7.8h, v10.8h, %9.h[3] \n"
                    "fmla v7.8h, v15.8h, %10.h[0] \n"
                    "fmla v7.8h, v16.8h, %10.h[1] \n"
                    "fmla v7.8h, v17.8h, %10.h[2] \n"
                    "ld2 {v25.8h, v26.8h}, [%5] \n"
                    "fmla v8.8h, v11.8h, %9.h[2] \n"
                    "fmla v8.8h, v12.8h, %9.h[3] \n"
                    "fmla v8.8h, v18.8h, %10.h[0] \n"
                    "fmla v8.8h, v19.8h, %10.h[1] \n"
                    "fmla v8.8h, v20.8h, %10.h[2] \n"

                    "ld2 {v9.8h, v10.8h}, [%6], #32 \n"
                    "ext v27.16b, v21.16b, v23.16b, #2 \n"
                    "ext v28.16b, v22.16b, v24.16b, #2 \n"
                    "ext v29.16b, v21.16b, v23.16b, #4 \n"
                    "fmla v7.8h, v21.8h, %10.h[3] \n"
                    "fmla v7.8h, v22.8h, %11.h[0] \n"
                    "fmla v7.8h, v27.8h, %11.h[1] \n"
                    "fmla v7.8h, v28.8h, %11.h[2] \n"
                    "fmla v7.8h, v29.8h, %11.h[3] \n"
                    "ld2 {v11.8h, v12.8h}, [%6], #32 \n"
                    "ext v30.16b, v23.16b, v25.16b, #2 \n"
                    "ext v31.16b, v24.16b, v26.16b, #2 \n"
                    "ext v21.16b, v23.16b, v25.16b, #4 \n"
                    "ld2 {v13.8h, v14.8h}, [%6] \n"
                    "fmla v8.8h, v23.8h, %10.h[3] \n"
                    "fmla v8.8h, v24.8h, %11.h[0] \n"
                    "fmla v8.8h, v30.8h, %11.h[1] \n"
                    "fmla v8.8h, v31.8h, %11.h[2] \n"
                    "fmla v8.8h, v21.8h, %11.h[3] \n"

                    "ext v15.16b, v9.16b, v11.16b, #2 \n"
                    "ext v16.16b, v10.16b, v12.16b, #2 \n"
                    "ext v17.16b, v9.16b, v11.16b, #4 \n"
                    "ext v18.16b, v11.16b, v13.16b, #2 \n"
                    "ext v19.16b, v12.16b, v14.16b, #2 \n"
                    "ext v20.16b, v11.16b, v13.16b, #4 \n"
                    "fmla v7.8h, v9.8h, %12.h[0] \n"
                    "fmla v7.8h, v10.8h, %12.h[1] \n"
                    "fmla v7.8h, v15.8h, %12.h[2] \n"
                    "fmla v7.8h, v16.8h, %12.h[3] \n"
                    "fmla v7.8h, v17.8h, %13.8h \n"
                    "fmla v8.8h, v11.8h, %12.h[0] \n"
                    "fmla v8.8h, v12.8h, %12.h[1] \n"
                    "fmla v8.8h, v18.8h, %12.h[2] \n"
                    "fmla v8.8h, v19.8h, %12.h[3] \n"
                    "fmla v8.8h, v20.8h, %13.8h \n"

                    "add x1, x1, #2 \n"
                    "cmp %0, x1 \n"
                    "st1 {v7.8h, v8.8h}, [%1], #32 \n"
                    "bne 2b \n"
                    "3: \n"

                    : "+r"(width), "+r"(outptr), "+r"(r0), "+r"(r1), "+r"(r2),
                      "+r"(r3), "+r"(r4)
                    : "w"(_k0123), "w"(_k4567), "w"(_k891011), "w"(_k12131415),
                      "w"(_k16171819), "w"(_k20212223), "w"(_k24242424),
                      "r"(mod2_left)
                    : "cc", "memory", "x1", "v7", "v8", "v9", "v10", "v11",
                      "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                      "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27",
                      "v28", "v29", "v30", "v31");

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
        }

        filter += 25;
    }
}

static void do_conv_7x7_stride2(const __fp16* src, const __fp16* filter,
                                __fp16* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 3;

    rep(ic, IC) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;
        const __fp16* r2 = src_ptr + IW * 2;
        const __fp16* r3 = src_ptr + IW * 3;
        const __fp16* r4 = src_ptr + IW * 4;
        const __fp16* r5 = src_ptr + IW * 5;
        const __fp16* r6 = src_ptr + IW * 6;

        register MEGDNN_SIMD_TYPE _k0123 asm("v0") = MEGDNN_SIMD_LOADU(filter);
        register MEGDNN_SIMD_TYPE _k4567 asm("v1") =
                MEGDNN_SIMD_LOADU(filter + 4);
        register MEGDNN_SIMD_TYPE _k891011 asm("v2") =
                MEGDNN_SIMD_LOADU(filter + 8);
        register MEGDNN_SIMD_TYPE _k12131415 asm("v3") =
                MEGDNN_SIMD_LOADU(filter + 12);
        register MEGDNN_SIMD_TYPE _k16171819 asm("v4") =
                MEGDNN_SIMD_LOADU(filter + 16);
        register MEGDNN_SIMD_TYPE _k20212223 asm("v5") =
                MEGDNN_SIMD_LOADU(filter + 20);
        register MEGDNN_SIMD_TYPE _k24252627 asm("v6") =
                MEGDNN_SIMD_LOADU(filter + 24);
        register MEGDNN_SIMD_TYPE _k28293031 asm("v7") =
                MEGDNN_SIMD_LOADU(filter + 28);
        register MEGDNN_SIMD_TYPE _k32333435 asm("v8") =
                MEGDNN_SIMD_LOADU(filter + 32);
        register MEGDNN_SIMD_TYPE _k36373839 asm("v9") =
                MEGDNN_SIMD_LOADU(filter + 36);
        register MEGDNN_SIMD_TYPE _k40414243 asm("v10") =
                MEGDNN_SIMD_LOADU(filter + 40);
        register MEGDNN_SIMD_TYPE _k44454647 asm("v11") =
                MEGDNN_SIMD_LOADU(filter + 44);
        register MEGDNN_SIMD_TYPE _k48484848 asm("v12") =
                MEGDNN_SIMD_SET1(filter[48]);

        for (size_t i = 0; i < OH; i++) {
            asm volatile(
                    "and x1, %8, #1 \n"
                    "cmp x1, #0 \n"
                    "mov x1, xzr \n"
                    "beq 1f \n"

                    // mod2_left == 1
                    "0: \n"
                    "ld1 {v13.8h}, [%0] \n"

                    // v14.8h: 0 2 4 6 v15.8h: 1 3 5 7
                    "ld2 {v14.8h, v15.8h}, [%1], #32 \n"
                    "ld2 {v23.8h, v24.8h}, [%2], #32 \n"
                    "ld2 {v16.8h, v17.8h}, [%1] \n"
                    "ld2 {v25.8h, v26.8h}, [%2] \n"
                    // v18.8h: 2 4 6 8
                    "ext v18.16b, v14.16b, v16.16b, #2 \n"
                    // v19.8h: 3 5 7 9
                    "ext v19.16b, v15.16b, v17.16b, #2 \n"
                    // v20.8h: 4 6 8 10
                    "ext v20.16b, v14.16b, v16.16b, #4 \n"
                    // v21.8h: 5 7 9 11
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    // v22.8h: 6 8 10 12
                    "ext v22.16b, v14.16b, v16.16b, #6 \n"
                    "fmla v13.8h, v14.8h, %9.h[0] \n"
                    "fmla v13.8h, v15.8h, %9.h[1] \n"
                    "fmla v13.8h, v18.8h, %9.h[2] \n"
                    "fmla v13.8h, v19.8h, %9.h[3] \n"
                    "fmla v13.8h, v20.8h, %10.h[0] \n"
                    "fmla v13.8h, v21.8h, %10.h[1] \n"
                    "fmla v13.8h, v22.8h, %10.h[2] \n"

                    "ld2 {v14.8h, v15.8h}, [%3], #32 \n"
                    "ext v27.16b, v23.16b, v25.16b, #2 \n"
                    "ext v28.16b, v24.16b, v26.16b, #2 \n"
                    "ext v29.16b, v23.16b, v25.16b, #4 \n"
                    "ext v30.16b, v24.16b, v26.16b, #4 \n"
                    "ext v31.16b, v23.16b, v25.16b, #6 \n"
                    "ld2 {v16.8h, v17.8h}, [%3] \n"
                    "fmla v13.8h, v23.8h, %10.h[3] \n"
                    "fmla v13.8h, v24.8h, %11.h[0] \n"
                    "fmla v13.8h, v27.8h, %11.h[1] \n"
                    "fmla v13.8h, v28.8h, %11.h[2] \n"
                    "fmla v13.8h, v29.8h, %11.h[3] \n"
                    "fmla v13.8h, v30.8h, %12.h[0] \n"
                    "fmla v13.8h, v31.8h, %12.h[1] \n"

                    "ld2 {v23.8h, v24.8h}, [%4], #32 \n"
                    "ext v18.16b, v14.16b, v16.16b, #2 \n"
                    "ext v19.16b, v15.16b, v17.16b, #2 \n"
                    "ext v20.16b, v14.16b, v16.16b, #4 \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "ext v22.16b, v14.16b, v16.16b, #6 \n"
                    "ld2 {v25.8h, v26.8h}, [%4] \n"
                    "fmla v13.8h, v14.8h, %12.h[2] \n"
                    "fmla v13.8h, v15.8h, %12.h[3] \n"
                    "fmla v13.8h, v18.8h, %13.h[0] \n"
                    "fmla v13.8h, v19.8h, %13.h[1] \n"
                    "fmla v13.8h, v20.8h, %13.h[2] \n"
                    "fmla v13.8h, v21.8h, %13.h[3] \n"
                    "fmla v13.8h, v22.8h, %14.h[0] \n"

                    "ld2 {v14.8h, v15.8h}, [%5], #32 \n"
                    "ext v27.16b, v23.16b, v25.16b, #2 \n"
                    "ext v28.16b, v24.16b, v26.16b, #2 \n"
                    "ext v29.16b, v23.16b, v25.16b, #4 \n"
                    "ext v30.16b, v24.16b, v26.16b, #4 \n"
                    "ext v31.16b, v23.16b, v25.16b, #6 \n"
                    "ld2 {v16.8h, v17.8h}, [%5] \n"
                    "fmla v13.8h, v23.8h, %14.h[1] \n"
                    "fmla v13.8h, v24.8h, %14.h[2] \n"
                    "fmla v13.8h, v27.8h, %14.h[3] \n"
                    "fmla v13.8h, v28.8h, %15.h[0] \n"
                    "fmla v13.8h, v29.8h, %15.h[1] \n"
                    "fmla v13.8h, v30.8h, %15.h[2] \n"
                    "fmla v13.8h, v31.8h, %15.h[3] \n"

                    "ld2 {v23.8h, v24.8h}, [%6], #32 \n"
                    "ext v18.16b, v14.16b, v16.16b, #2 \n"
                    "ext v19.16b, v15.16b, v17.16b, #2 \n"
                    "ext v20.16b, v14.16b, v16.16b, #4 \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "ext v22.16b, v14.16b, v16.16b, #6 \n"
                    "ld2 {v25.8h, v26.8h}, [%6] \n"
                    "fmla v13.8h, v14.8h, %16.h[0] \n"
                    "fmla v13.8h, v15.8h, %16.h[1] \n"
                    "fmla v13.8h, v18.8h, %16.h[2] \n"
                    "fmla v13.8h, v19.8h, %16.h[3] \n"
                    "fmla v13.8h, v20.8h, %17.h[0] \n"
                    "fmla v13.8h, v21.8h, %17.h[1] \n"
                    "fmla v13.8h, v22.8h, %17.h[2] \n"

                    "ld2 {v14.8h, v15.8h}, [%7], #32 \n"
                    "ext v27.16b, v23.16b, v25.16b, #2 \n"
                    "ext v28.16b, v24.16b, v26.16b, #2 \n"
                    "ext v29.16b, v23.16b, v25.16b, #4 \n"
                    "ext v30.16b, v24.16b, v26.16b, #4 \n"
                    "ext v31.16b, v23.16b, v25.16b, #6 \n"
                    "ld2 {v16.8h, v17.8h}, [%7] \n"
                    "fmla v13.8h, v23.8h, %17.h[3] \n"
                    "fmla v13.8h, v24.8h, %18.h[0] \n"
                    "fmla v13.8h, v27.8h, %18.h[1] \n"
                    "fmla v13.8h, v28.8h, %18.h[2] \n"
                    "fmla v13.8h, v29.8h, %18.h[3] \n"
                    "fmla v13.8h, v30.8h, %19.h[0] \n"
                    "fmla v13.8h, v31.8h, %19.h[1] \n"

                    "ext v18.16b, v14.16b, v16.16b, #2 \n"
                    "ext v19.16b, v15.16b, v17.16b, #2 \n"
                    "ext v20.16b, v14.16b, v16.16b, #4 \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "ext v22.16b, v14.16b, v16.16b, #6 \n"
                    "fmla v13.8h, v14.8h, %19.h[2] \n"
                    "fmla v13.8h, v15.8h, %19.h[3] \n"
                    "fmla v13.8h, v18.8h, %20.h[0] \n"
                    "fmla v13.8h, v19.8h, %20.h[1] \n"
                    "fmla v13.8h, v20.8h, %20.h[2] \n"
                    "fmla v13.8h, v21.8h, %20.h[3] \n"
                    "fmla v13.8h, v22.8h, %21.8h \n"

                    "add x1, x1, #1 \n"
                    "st1 {v13.8h}, [%0], #16 \n"

                    "1: \n"
                    "cmp %8, x1 \n"
                    "beq 3f \n"

                    // mod2_left == 0
                    "2: \n"
                    "ld1 {v13.8h, v14.8h}, [%0] \n"

                    // v15.8h: 0 2 4 6 v16.8h: 1 3 5 7
                    "ld2 {v15.8h, v16.8h}, [%1], #32 \n"
                    // v17.8h: 8 10 12 14 v16.8h: 9 11 13 15
                    "ld2 {v17.8h, v18.8h}, [%1], #32 \n"
                    // v19.8h: 16 18 20 22 v20.8h: 17 19 21 23
                    "ld2 {v19.8h, v20.8h}, [%1] \n"
                    // v21.8h: 2 4 6 8
                    "ext v21.16b, v15.16b, v17.16b, #2 \n"
                    // v22.8h: 3 5 7 9
                    "ext v22.16b, v16.16b, v18.16b, #2 \n"
                    // v23.8h: 4 6 8 10
                    "ext v23.16b, v15.16b, v17.16b, #4 \n"
                    // v24.8h: 5 7 9 11
                    "ext v24.16b, v16.16b, v18.16b, #4 \n"
                    // v25.8h: 6 8 10 12
                    "ext v25.16b, v15.16b, v17.16b, #6 \n"
                    "fmla v13.8h, v15.8h, %9.h[0] \n"
                    "fmla v13.8h, v16.8h, %9.h[1] \n"
                    "fmla v13.8h, v21.8h, %9.h[2] \n"
                    "fmla v13.8h, v22.8h, %9.h[3] \n"
                    "fmla v13.8h, v23.8h, %10.h[0] \n"
                    "fmla v13.8h, v24.8h, %10.h[1] \n"
                    "fmla v13.8h, v25.8h, %10.h[2] \n"
                    // v15.8h: 10 12 14 16
                    "ext v15.16b, v17.16b, v19.16b, #2 \n"
                    // v16.8h: 11 13 15 17
                    "ext v16.16b, v18.16b, v20.16b, #2 \n"
                    // v21.8h: 12 14 16 18
                    "ext v21.16b, v17.16b, v19.16b, #4 \n"
                    // v22.8h: 13 15 17 19
                    "ext v22.16b, v18.16b, v20.16b, #4 \n"
                    // v23.8h: 14 16 18 19
                    "ext v23.16b, v17.16b, v19.16b, #6 \n"
                    "fmla v14.8h, v17.8h, %9.h[0] \n"
                    "fmla v14.8h, v18.8h, %9.h[1] \n"
                    "fmla v14.8h, v15.8h, %9.h[2] \n"
                    "fmla v14.8h, v16.8h, %9.h[3] \n"
                    "fmla v14.8h, v21.8h, %10.h[0] \n"
                    "fmla v14.8h, v22.8h, %10.h[1] \n"
                    "fmla v14.8h, v23.8h, %10.h[2] \n"

                    "ld2 {v26.8h, v27.8h}, [%2], #32 \n"
                    "ld2 {v28.8h, v29.8h}, [%2], #32 \n"
                    "ld2 {v30.8h, v31.8h}, [%2] \n"
                    "ext v21.16b, v26.16b, v28.16b, #2 \n"
                    "ext v22.16b, v27.16b, v29.16b, #2 \n"
                    "ext v23.16b, v26.16b, v28.16b, #4 \n"
                    "ext v24.16b, v27.16b, v29.16b, #4 \n"
                    "ext v25.16b, v26.16b, v28.16b, #6 \n"
                    "fmla v13.8h, v26.8h, %10.h[3] \n"
                    "fmla v13.8h, v27.8h, %11.h[0] \n"
                    "fmla v13.8h, v21.8h, %11.h[1] \n"
                    "fmla v13.8h, v22.8h, %11.h[2] \n"
                    "fmla v13.8h, v23.8h, %11.h[3] \n"
                    "fmla v13.8h, v24.8h, %12.h[0] \n"
                    "fmla v13.8h, v25.8h, %12.h[1] \n"
                    "ext v26.16b, v28.16b, v30.16b, #2 \n"
                    "ext v27.16b, v29.16b, v31.16b, #2 \n"
                    "ext v21.16b, v28.16b, v30.16b, #4 \n"
                    "ext v22.16b, v29.16b, v31.16b, #4 \n"
                    "ext v23.16b, v28.16b, v30.16b, #6 \n"
                    "fmla v14.8h, v28.8h, %10.h[3] \n"
                    "fmla v14.8h, v29.8h, %11.h[0] \n"
                    "fmla v14.8h, v26.8h, %11.h[1] \n"
                    "fmla v14.8h, v27.8h, %11.h[2] \n"
                    "fmla v14.8h, v21.8h, %11.h[3] \n"
                    "fmla v14.8h, v22.8h, %12.h[0] \n"
                    "fmla v14.8h, v23.8h, %12.h[1] \n"

                    "ld2 {v15.8h, v16.8h}, [%3], #32 \n"
                    "ld2 {v17.8h, v18.8h}, [%3], #32 \n"
                    "ld2 {v19.8h, v20.8h}, [%3] \n"
                    "ext v21.16b, v15.16b, v17.16b, #2 \n"
                    "ext v22.16b, v16.16b, v18.16b, #2 \n"
                    "ext v23.16b, v15.16b, v17.16b, #4 \n"
                    "ext v24.16b, v16.16b, v18.16b, #4 \n"
                    "ext v25.16b, v15.16b, v17.16b, #6 \n"
                    "fmla v13.8h, v15.8h, %12.h[2] \n"
                    "fmla v13.8h, v16.8h, %12.h[3] \n"
                    "fmla v13.8h, v21.8h, %13.h[0] \n"
                    "fmla v13.8h, v22.8h, %13.h[1] \n"
                    "fmla v13.8h, v23.8h, %13.h[2] \n"
                    "fmla v13.8h, v24.8h, %13.h[3] \n"
                    "fmla v13.8h, v25.8h, %14.h[0] \n"
                    "ext v15.16b, v17.16b, v19.16b, #2 \n"
                    "ext v16.16b, v18.16b, v20.16b, #2 \n"
                    "ext v21.16b, v17.16b, v19.16b, #4 \n"
                    "ext v22.16b, v18.16b, v20.16b, #4 \n"
                    "ext v23.16b, v17.16b, v19.16b, #6 \n"
                    "fmla v14.8h, v17.8h, %12.h[2] \n"
                    "fmla v14.8h, v18.8h, %12.h[3] \n"
                    "fmla v14.8h, v15.8h, %13.h[0] \n"
                    "fmla v14.8h, v16.8h, %13.h[1] \n"
                    "fmla v14.8h, v21.8h, %13.h[2] \n"
                    "fmla v14.8h, v22.8h, %13.h[3] \n"
                    "fmla v14.8h, v23.8h, %14.h[0] \n"

                    "ld2 {v26.8h, v27.8h}, [%4], #32 \n"
                    "ld2 {v28.8h, v29.8h}, [%4], #32 \n"
                    "ld2 {v30.8h, v31.8h}, [%4] \n"
                    "ext v21.16b, v26.16b, v28.16b, #2 \n"
                    "ext v22.16b, v27.16b, v29.16b, #2 \n"
                    "ext v23.16b, v26.16b, v28.16b, #4 \n"
                    "ext v24.16b, v27.16b, v29.16b, #4 \n"
                    "ext v25.16b, v26.16b, v28.16b, #6 \n"
                    "fmla v13.8h, v26.8h, %14.h[1] \n"
                    "fmla v13.8h, v27.8h, %14.h[2] \n"
                    "fmla v13.8h, v21.8h, %14.h[3] \n"
                    "fmla v13.8h, v22.8h, %15.h[0] \n"
                    "fmla v13.8h, v23.8h, %15.h[1] \n"
                    "fmla v13.8h, v24.8h, %15.h[2] \n"
                    "fmla v13.8h, v25.8h, %15.h[3] \n"
                    "ext v26.16b, v28.16b, v30.16b, #2 \n"
                    "ext v27.16b, v29.16b, v31.16b, #2 \n"
                    "ext v21.16b, v28.16b, v30.16b, #4 \n"
                    "ext v22.16b, v29.16b, v31.16b, #4 \n"
                    "ext v23.16b, v28.16b, v30.16b, #6 \n"
                    "fmla v14.8h, v28.8h, %14.h[1] \n"
                    "fmla v14.8h, v29.8h, %14.h[2] \n"
                    "fmla v14.8h, v26.8h, %14.h[3] \n"
                    "fmla v14.8h, v27.8h, %15.h[0] \n"
                    "fmla v14.8h, v21.8h, %15.h[1] \n"
                    "fmla v14.8h, v22.8h, %15.h[2] \n"
                    "fmla v14.8h, v23.8h, %15.h[3] \n"

                    "ld2 {v15.8h, v16.8h}, [%5], #32 \n"
                    "ld2 {v17.8h, v18.8h}, [%5], #32 \n"
                    "ld2 {v19.8h, v20.8h}, [%5] \n"
                    "ext v21.16b, v15.16b, v17.16b, #2 \n"
                    "ext v22.16b, v16.16b, v18.16b, #2 \n"
                    "ext v23.16b, v15.16b, v17.16b, #4 \n"
                    "ext v24.16b, v16.16b, v18.16b, #4 \n"
                    "ext v25.16b, v15.16b, v17.16b, #6 \n"
                    "fmla v13.8h, v15.8h, %16.h[0] \n"
                    "fmla v13.8h, v16.8h, %16.h[1] \n"
                    "fmla v13.8h, v21.8h, %16.h[2] \n"
                    "fmla v13.8h, v22.8h, %16.h[3] \n"
                    "fmla v13.8h, v23.8h, %17.h[0] \n"
                    "fmla v13.8h, v24.8h, %17.h[1] \n"
                    "fmla v13.8h, v25.8h, %17.h[2] \n"
                    "ext v15.16b, v17.16b, v19.16b, #2 \n"
                    "ext v16.16b, v18.16b, v20.16b, #2 \n"
                    "ext v21.16b, v17.16b, v19.16b, #4 \n"
                    "ext v22.16b, v18.16b, v20.16b, #4 \n"
                    "ext v23.16b, v17.16b, v19.16b, #6 \n"
                    "fmla v14.8h, v17.8h, %16.h[0] \n"
                    "fmla v14.8h, v18.8h, %16.h[1] \n"
                    "fmla v14.8h, v15.8h, %16.h[2] \n"
                    "fmla v14.8h, v16.8h, %16.h[3] \n"
                    "fmla v14.8h, v21.8h, %17.h[0] \n"
                    "fmla v14.8h, v22.8h, %17.h[1] \n"
                    "fmla v14.8h, v23.8h, %17.h[2] \n"

                    "ld2 {v26.8h, v27.8h}, [%6], #32 \n"
                    "ld2 {v28.8h, v29.8h}, [%6], #32 \n"
                    "ld2 {v30.8h, v31.8h}, [%6] \n"
                    "ext v21.16b, v26.16b, v28.16b, #2 \n"
                    "ext v22.16b, v27.16b, v29.16b, #2 \n"
                    "ext v23.16b, v26.16b, v28.16b, #4 \n"
                    "ext v24.16b, v27.16b, v29.16b, #4 \n"
                    "ext v25.16b, v26.16b, v28.16b, #6 \n"
                    "fmla v13.8h, v26.8h, %17.h[3] \n"
                    "fmla v13.8h, v27.8h, %18.h[0] \n"
                    "fmla v13.8h, v21.8h, %18.h[1] \n"
                    "fmla v13.8h, v22.8h, %18.h[2] \n"
                    "fmla v13.8h, v23.8h, %18.h[3] \n"
                    "fmla v13.8h, v24.8h, %19.h[0] \n"
                    "fmla v13.8h, v25.8h, %19.h[1] \n"
                    "ext v26.16b, v28.16b, v30.16b, #2 \n"
                    "ext v27.16b, v29.16b, v31.16b, #2 \n"
                    "ext v21.16b, v28.16b, v30.16b, #4 \n"
                    "ext v22.16b, v29.16b, v31.16b, #4 \n"
                    "ext v23.16b, v28.16b, v30.16b, #6 \n"
                    "fmla v14.8h, v28.8h, %17.h[3] \n"
                    "fmla v14.8h, v29.8h, %18.h[0] \n"
                    "fmla v14.8h, v26.8h, %18.h[1] \n"
                    "fmla v14.8h, v27.8h, %18.h[2] \n"
                    "fmla v14.8h, v21.8h, %18.h[3] \n"
                    "fmla v14.8h, v22.8h, %19.h[0] \n"
                    "fmla v14.8h, v23.8h, %19.h[1] \n"

                    "ld2 {v15.8h, v16.8h}, [%7], #32 \n"
                    "ld2 {v17.8h, v18.8h}, [%7], #32 \n"
                    "ld2 {v19.8h, v20.8h}, [%7] \n"
                    "ext v21.16b, v15.16b, v17.16b, #2 \n"
                    "ext v22.16b, v16.16b, v18.16b, #2 \n"
                    "ext v23.16b, v15.16b, v17.16b, #4 \n"
                    "ext v24.16b, v16.16b, v18.16b, #4 \n"
                    "ext v25.16b, v15.16b, v17.16b, #6 \n"
                    "fmla v13.8h, v15.8h, %19.h[2] \n"
                    "fmla v13.8h, v16.8h, %19.h[3] \n"
                    "fmla v13.8h, v21.8h, %20.h[0] \n"
                    "fmla v13.8h, v22.8h, %20.h[1] \n"
                    "fmla v13.8h, v23.8h, %20.h[2] \n"
                    "fmla v13.8h, v24.8h, %20.h[3] \n"
                    "fmla v13.8h, v25.8h, %21.8h \n"
                    "ext v15.16b, v17.16b, v19.16b, #2 \n"
                    "ext v16.16b, v18.16b, v20.16b, #2 \n"
                    "ext v21.16b, v17.16b, v19.16b, #4 \n"
                    "ext v22.16b, v18.16b, v20.16b, #4 \n"
                    "ext v23.16b, v17.16b, v19.16b, #6 \n"
                    "fmla v14.8h, v17.8h, %19.h[2] \n"
                    "fmla v14.8h, v18.8h, %19.h[3] \n"
                    "fmla v14.8h, v15.8h, %20.h[0] \n"
                    "fmla v14.8h, v16.8h, %20.h[1] \n"
                    "fmla v14.8h, v21.8h, %20.h[2] \n"
                    "fmla v14.8h, v22.8h, %20.h[3] \n"
                    "fmla v14.8h, v23.8h, %21.8h \n"

                    "add x1, x1, #2 \n"
                    "st1 {v13.8h, v14.8h}, [%0], #32 \n"
                    "cmp %8, x1 \n"
                    "bne 2b \n"
                    "3: \n"

                    : "+r"(outptr), "+r"(r0), "+r"(r1), "+r"(r2), "+r"(r3),
                      "+r"(r4), "+r"(r5), "+r"(r6)
                    : "r"(width), "w"(_k0123), "w"(_k4567), "w"(_k891011),
                      "w"(_k12131415), "w"(_k16171819), "w"(_k20212223),
                      "w"(_k24252627), "w"(_k28293031), "w"(_k32333435),
                      "w"(_k36373839), "w"(_k40414243), "w"(_k44454647),
                      "w"(_k48484848)
                    : "cc", "memory", "x1", "v13", "v14", "v15", "v16", "v17",
                      "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25",
                      "v26", "v27", "v28", "v29", "v30", "v31");

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
            r5 += tail_step;
            r6 += tail_step;
        }
        filter += 49;
    }
}

}  // namespace conv_stride2
}  // namespace fp16
}  // namespace aarch64
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
