/**
 * \file dnn/src/aarch64/conv_bias/fp32/stride2_kern.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>
#include "src/arm_common/simd_macro/neon_helper.h"
#include "src/common/utils.h"

namespace megdnn {
namespace aarch64 {
namespace fp32{
namespace conv_stride2 {


//! For the detail tune process, refer to `expr/conv_aarch64_stride2/main.cpp`

// refer to function do_conv_2x2_stride2_asm_unroll4
static void do_conv_2x2_stride2(const float* src, const float* filter,
                                float* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 2;
    size_t mod4_left = width & 3;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;

        const float* k0 = filter;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        rep(h, OH) {
            asm volatile(
                    "dup v28.4s, %5.s[0] \n"
                    "dup v29.4s, %5.s[1] \n"
                    "dup v30.4s, %5.s[2] \n"
                    "dup v31.4s, %5.s[3] \n"
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
                    "ld1 {v0.4s, v1.4s, v2.4s}, [%1] \n"

                    "ld2 {v3.4s, v4.4s}, [%2], #32 \n"
                    "ld2 {v9.4s, v10.4s}, [%3], #32 \n"
                    "ld2 {v5.4s, v6.4s}, [%2], #32 \n"
                    "ld2 {v11.4s, v12.4s}, [%3], #32 \n"
                    "ld2 {v7.4s, v8.4s}, [%2], #32 \n"
                    "ld2 {v13.4s, v14.4s}, [%3], #32 \n"
                    "fmla v0.4s, v3.4s, v28.4s \n"
                    "fmla v1.4s, v5.4s, v28.4s \n"
                    "fmla v2.4s, v7.4s, v28.4s \n"
                    "fmla v0.4s, v4.4s, v29.4s \n"
                    "fmla v1.4s, v6.4s, v29.4s \n"
                    "fmla v2.4s, v8.4s, v29.4s \n"

                    "fmla v0.4s, v9.4s, v30.4s \n"
                    "fmla v1.4s, v11.4s, v30.4s \n"
                    "fmla v2.4s, v13.4s, v30.4s \n"
                    "fmla v0.4s, v10.4s, v31.4s \n"
                    "fmla v1.4s, v12.4s, v31.4s \n"
                    "fmla v2.4s, v14.4s, v31.4s \n"

                    "add x1, x1, #3 \n"
                    "st1 {v0.4s, v1.4s, v2.4s}, [%1], #48 \n"
                    "b 3f \n"

                    // mod4_left == 2
                    "1: \n"
                    "ld1 {v0.4s, v1.4s}, [%1] \n"

                    "ld2 {v2.4s, v3.4s}, [%2], #32 \n"
                    "ld2 {v6.4s, v7.4s}, [%3], #32 \n"
                    "ld2 {v4.4s, v5.4s}, [%2], #32 \n"
                    "ld2 {v8.4s, v9.4s}, [%3], #32 \n"
                    "fmla v0.4s, v2.4s, v28.4s \n"
                    "fmla v1.4s, v4.4s, v28.4s \n"
                    "fmla v0.4s, v3.4s, v29.4s \n"
                    "fmla v1.4s, v5.4s, v29.4s \n"

                    "fmla v0.4s, v6.4s, v30.4s \n"
                    "fmla v1.4s, v8.4s, v30.4s \n"
                    "fmla v0.4s, v7.4s, v31.4s \n"
                    "fmla v1.4s, v9.4s, v31.4s \n"

                    "add x1, x1, #2 \n"
                    "st1 {v0.4s, v1.4s}, [%1], #32 \n"
                    "b 3f \n"

                    // mod4_left == 1
                    "2: \n"
                    "ld1 {v0.4s}, [%1] \n"

                    "ld2 {v1.4s, v2.4s}, [%2], #32 \n"
                    "ld2 {v3.4s, v4.4s}, [%3], #32 \n"
                    "fmla v0.4s, v1.4s, v28.4s \n"
                    "fmla v0.4s, v2.4s, v29.4s \n"

                    "fmla v0.4s, v3.4s, v30.4s \n"
                    "fmla v0.4s, v4.4s, v31.4s \n"

                    "add x1, x1, #1 \n"
                    "st1 {v0.4s}, [%1], #16 \n"
                    "b 3f \n"

                    // mod4_left == 0
                    "3: \n"
                    "cmp %0, x1 \n"
                    "beq 5f \n"
                    "4: \n"
                    "ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%1] \n"

                    "ld2 {v4.4s, v5.4s}, [%2], #32 \n"
                    "ld2 {v12.4s, v13.4s}, [%3], #32 \n"
                    "ld2 {v6.4s, v7.4s}, [%2], #32 \n"
                    "ld2 {v14.4s, v15.4s}, [%3], #32 \n"
                    "ld2 {v8.4s, v9.4s}, [%2], #32 \n"
                    "ld2 {v16.4s, v17.4s}, [%3], #32 \n"
                    "ld2 {v10.4s, v11.4s}, [%2], #32 \n"
                    "ld2 {v18.4s, v19.4s}, [%3], #32 \n"
                    "fmla v0.4s, v4.4s, v28.4s \n"
                    "fmla v1.4s, v6.4s, v28.4s \n"
                    "fmla v2.4s, v8.4s, v28.4s \n"
                    "fmla v3.4s, v10.4s, v28.4s \n"
                    "fmla v0.4s, v5.4s, v29.4s \n"
                    "fmla v1.4s, v7.4s, v29.4s \n"
                    "fmla v2.4s, v9.4s, v29.4s \n"
                    "fmla v3.4s, v11.4s, v29.4s \n"

                    "fmla v0.4s, v12.4s, v30.4s \n"
                    "fmla v1.4s, v14.4s, v30.4s \n"
                    "fmla v2.4s, v16.4s, v30.4s \n"
                    "fmla v3.4s, v18.4s, v30.4s \n"
                    "fmla v0.4s, v13.4s, v31.4s \n"
                    "fmla v1.4s, v15.4s, v31.4s \n"
                    "fmla v2.4s, v17.4s, v31.4s \n"
                    "fmla v3.4s, v19.4s, v31.4s \n"

                    "add x1, x1, #4 \n"
                    "cmp %0, x1 \n"
                    "st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [%1], #64 \n"
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

// refer to function do_conv_3x3_stride2_asm_unroll3
static void do_conv_3x3_stride2(const float* src, const float* filter,
                                float* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 2;
    size_t mod3_left = width % 3;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;

        const float* k0 = filter;
        const float* k1 = filter + 3;
        const float* k2 = filter + 5;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        MEGDNN_SIMD_TYPE _k3456 = MEGDNN_SIMD_LOADU(k1);
        MEGDNN_SIMD_TYPE _k5678 = MEGDNN_SIMD_LOADU(k2);
        rep(h, OH) {
            asm volatile(
                    "dup v21.4s, %6.s[0] \n"
                    "dup v22.4s, %6.s[1] \n"
                    "dup v23.4s, %6.s[2] \n"
                    "dup v24.4s, %6.s[3] \n"
                    "dup v25.4s, %7.s[1] \n"
                    "dup v26.4s, %7.s[2] \n"
                    "dup v27.4s, %7.s[3] \n"
                    "dup v28.4s, %8.s[2] \n"
                    "dup v29.4s, %8.s[3] \n"
                    "cmp %5, #1 \n"
                    "mov x1, xzr \n"
                    "bgt 0f \n"  // mod3_left == 2
                    "beq 1f \n"  // mod3_left == 1
                    "blt 2f \n"  // mod3_left == 0

                    "0:  \n"
                    "ld1 {v0.4s, v1.4s}, [%1] \n"

                    "ld2 {v2.4s, v3.4s}, [%2], #32 \n"
                    "ld2 {v9.4s, v10.4s}, [%3], #32 \n"
                    "ld2 {v4.4s, v5.4s}, [%2], #32 \n"
                    "ld2 {v11.4s, v12.4s}, [%3], #32 \n"
                    "fmla v0.4s, v2.4s, v21.4s \n"
                    "fmla v1.4s, v4.4s, v21.4s \n"
                    "fmla v0.4s, v3.4s, v22.4s \n"
                    "fmla v1.4s, v5.4s, v22.4s \n"
                    "ld1 {v6.4s}, [%2] \n"
                    "ld1 {v13.4s}, [%3] \n"
                    "ext v7.16b, v2.16b, v4.16b, #4 \n"
                    "ext v8.16b, v4.16b, v6.16b, #4 \n"
                    "fmla v0.4s, v7.4s, v23.4s \n"
                    "fmla v1.4s, v8.4s, v23.4s \n"

                    "ld2 {v2.4s, v3.4s}, [%4], #32 \n"
                    "fmla v0.4s, v9.4s, v24.4s \n"
                    "fmla v1.4s, v11.4s, v24.4s \n"
                    "fmla v0.4s, v10.4s, v25.4s \n"
                    "fmla v1.4s, v12.4s, v25.4s \n"
                    "ld2 {v4.4s, v5.4s}, [%4], #32 \n"
                    "ext v14.16b, v9.16b, v11.16b, #4 \n"
                    "ext v15.16b, v11.16b, v13.16b, #4 \n"
                    "fmla v0.4s, v14.4s, v26.4s \n"
                    "fmla v1.4s, v15.4s, v26.4s \n"

                    "ld1 {v6.4s}, [%4] \n"
                    "fmla v0.4s, v2.4s, v27.4s \n"
                    "fmla v1.4s, v4.4s, v27.4s \n"
                    "fmla v0.4s, v3.4s, v28.4s \n"
                    "fmla v1.4s, v5.4s, v28.4s \n"
                    "ext v7.16b, v2.16b, v4.16b, #4 \n"
                    "ext v8.16b, v4.16b, v6.16b, #4 \n"
                    "fmla v0.4s, v7.4s, v29.4s \n"
                    "fmla v1.4s, v8.4s, v29.4s \n"

                    "add x1, x1, #2 \n"
                    "cmp %0, x1 \n"

                    "st1 {v0.4s, v1.4s}, [%1], #32 \n"
                    "bne 2f \n"  // if width != 2 jump to 2
                    "b 3f \n"    // jump end

                    "1:           \n"
                    "ld1 {v0.4s}, [%1] \n"              // load dst 0, 1, 2, 3
                    "ld2 {v1.4s, v2.4s}, [%2], #32 \n"  // 0, 2, 4, 6

                    "ld2 {v5.4s, v6.4s}, [%3], #32 \n"
                    "ld1 {v3.4s}, [%2] \n"        // load src 8 12 ...
                    "fmla v0.4s, v1.4s, v21.4s \n"       // src[i] * k[i]
                    "ext v7.16b, v1.16b, v3.16b, #4 \n"  // 2, 4, 6, 8
                    "fmla v0.4s, v2.4s, v22.4s \n"
                    "ld1 {v1.4s}, [%3] \n"  // load src 8 12 ...
                    "fmla v0.4s, v7.4s, v23.4s \n"
                    "ld2 {v3.4s, v4.4s}, [%4], #32 \n"

                    "fmla v0.4s, v5.4s, v24.4s \n"
                    "ext v7.16b, v5.16b, v1.16b, #4 \n"  // 2, 4, 6, 8
                    "fmla v0.4s, v6.4s, v25.4s \n"
                    "ld1 {v5.4s}, [%4] \n"  // load src 8 12 ...
                    "fmla v0.4s, v7.4s, v26.4s \n"

                    "fmla v0.4s, v3.4s, v27.4s \n"
                    "fmla v0.4s, v4.4s, v28.4s \n"
                    "ext v7.16b, v3.16b, v5.16b, #4 \n"  // 2, 4, 6, 8
                    "fmla v0.4s, v7.4s, v29.4s \n"

                    "st1 {v0.4s}, [%1], #16 \n"

                    "add x1, x1, #1 \n"
                    "cmp %0, x1 \n"
                    "beq 3f \n"

                    "2: \n"
                    "ld1 {v0.4s, v1.4s, v2.4s}, [%1] \n"

                    "ld2 {v3.4s, v4.4s}, [%2], #32 \n"
                    "ld2 {v11.4s, v12.4s}, [%3], #32 \n"
                    "ld2 {v5.4s, v6.4s}, [%2], #32 \n"
                    "ld2 {v13.4s, v14.4s}, [%3], #32 \n"
                    "ld2 {v7.4s, v8.4s}, [%2], #32 \n"
                    "ld2 {v15.4s, v16.4s}, [%3], #32 \n"
                    "fmla v0.4s, v3.4s, v21.4s \n"
                    "fmla v1.4s, v5.4s, v21.4s \n"
                    "fmla v2.4s, v7.4s, v21.4s \n"
                    "ld1 {v9.4s}, [%2] \n"
                    "ld1 {v17.4s}, [%3] \n"
                    "fmla v0.4s, v4.4s, v22.4s \n"
                    "fmla v1.4s, v6.4s, v22.4s \n"
                    "fmla v2.4s, v8.4s, v22.4s \n"
                    "ext v10.16b, v3.16b, v5.16b, #4 \n"
                    "ext v4.16b, v5.16b, v7.16b, #4 \n"
                    "ext v6.16b, v7.16b, v9.16b, #4 \n"
                    "fmla v0.4s, v10.4s, v23.4s \n"
                    "fmla v1.4s, v4.4s, v23.4s \n"
                    "fmla v2.4s, v6.4s, v23.4s \n"

                    "ld2 {v3.4s, v4.4s}, [%4], #32 \n"
                    "fmla v0.4s, v11.4s, v24.4s \n"
                    "fmla v1.4s, v13.4s, v24.4s \n"
                    "fmla v2.4s, v15.4s, v24.4s \n"
                    "ld2 {v5.4s, v6.4s}, [%4], #32 \n"
                    "fmla v0.4s, v12.4s, v25.4s \n"
                    "fmla v1.4s, v14.4s, v25.4s \n"
                    "fmla v2.4s, v16.4s, v25.4s \n"
                    "ld2 {v7.4s, v8.4s}, [%4], #32 \n"
                    "ext v18.16b, v11.16b, v13.16b, #4 \n"
                    "ext v12.16b, v13.16b, v15.16b, #4 \n"
                    "ext v14.16b, v15.16b, v17.16b, #4 \n"
                    "fmla v0.4s, v18.4s, v26.4s \n"
                    "fmla v1.4s, v12.4s, v26.4s \n"
                    "fmla v2.4s, v14.4s, v26.4s \n"

                    "ld1 {v9.4s}, [%4] \n"
                    "fmla v0.4s, v3.4s, v27.4s \n"
                    "fmla v1.4s, v5.4s, v27.4s \n"
                    "fmla v2.4s, v7.4s, v27.4s \n"
                    "fmla v0.4s, v4.4s, v28.4s \n"
                    "fmla v1.4s, v6.4s, v28.4s \n"
                    "fmla v2.4s, v8.4s, v28.4s \n"
                    "ext v10.16b, v3.16b, v5.16b, #4 \n"
                    "ext v4.16b, v5.16b, v7.16b, #4 \n"
                    "ext v6.16b, v7.16b, v9.16b, #4 \n"
                    "fmla v0.4s, v10.4s, v29.4s \n"
                    "fmla v1.4s, v4.4s, v29.4s \n"
                    "fmla v2.4s, v6.4s, v29.4s \n"

                    "add x1, x1, #3 \n"
                    "cmp %0, x1 \n"

                    "st1 {v0.4s, v1.4s, v2.4s}, [%1], #48 \n"
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

// refer to function do_conv_5x5_stride2_asm_unroll2
static void do_conv_5x5_stride2(const float* src, const float* filter,
                                float* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 2;
    size_t mod2_left = width & 1;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(filter);
        MEGDNN_SIMD_TYPE _k4567 = MEGDNN_SIMD_LOADU(filter + 4);
        MEGDNN_SIMD_TYPE _k891011 = MEGDNN_SIMD_LOADU(filter + 8);
        MEGDNN_SIMD_TYPE _k12131415 = MEGDNN_SIMD_LOADU(filter + 12);
        MEGDNN_SIMD_TYPE _k16171819 = MEGDNN_SIMD_LOADU(filter + 16);
        MEGDNN_SIMD_TYPE _k20212223 = MEGDNN_SIMD_LOADU(filter + 20);
        MEGDNN_SIMD_TYPE _k24242424 = MEGDNN_SIMD_SET1(filter[24]);

        for (size_t i = 0; i < OH; i++) {
            asm volatile(
                    "cmp %14, #0 \n"
                    "mov x1, xzr \n"
                    "beq 1f \n"

                    // mod2_left == 1
                    "0: \n"
                    "ld1 {v0.4s}, [%1] \n"

                    // v1.4s: 0 2 4 6 v2.4s: 1 3 5 7
                    "ld2 {v1.4s, v2.4s}, [%2], #32 \n"
                    "ld2 {v8.4s, v9.4s}, [%3], #32 \n"
                    "fmla v0.4s, v1.4s, %7.s[0] \n"
                    "fmla v0.4s, v2.4s, %7.s[1] \n"
                    "ld2 {v3.4s, v4.4s}, [%2] \n"
                    "ld2 {v10.4s, v11.4s}, [%3] \n"
                    // v5.4s: 2 4 6 8
                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    // v6.4s: 3 5 7 9
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    "fmla v0.4s, v5.4s, %7.s[2] \n"
                    "fmla v0.4s, v6.4s, %7.s[3] \n"
                    // v7.4s: 4 6 8 10
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    "fmla v0.4s, v7.4s, %8.s[0] \n"

                    "ld2 {v1.4s, v2.4s}, [%4], #32 \n"
                    "fmla v0.4s, v8.4s, %8.s[1] \n"
                    "fmla v0.4s, v9.4s, %8.s[2] \n"
                    "ld2 {v3.4s, v4.4s}, [%4] \n"
                    "ext v12.16b, v8.16b, v10.16b, #4 \n"
                    "ext v13.16b, v9.16b, v11.16b, #4 \n"
                    "fmla v0.4s, v12.4s, %8.s[3] \n"
                    "fmla v0.4s, v13.4s, %9.s[0] \n"
                    "ext v14.16b, v8.16b, v10.16b, #8 \n"
                    "fmla v0.4s, v14.4s, %9.s[1] \n"

                    "ld2 {v8.4s, v9.4s}, [%5], #32 \n"
                    "fmla v0.4s, v1.4s, %9.s[2] \n"
                    "fmla v0.4s, v2.4s, %9.s[3] \n"
                    "ld2 {v10.4s, v11.4s}, [%5] \n"
                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    "fmla v0.4s, v5.4s, %10.s[0] \n"
                    "fmla v0.4s, v6.4s, %10.s[1] \n"
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    "fmla v0.4s, v7.4s, %10.s[2] \n"

                    "ld2 {v1.4s, v2.4s}, [%6], #32 \n"
                    "fmla v0.4s, v8.4s, %10.s[3] \n"
                    "fmla v0.4s, v9.4s, %11.s[0] \n"
                    "ld2 {v3.4s, v4.4s}, [%6] \n"
                    "ext v12.16b, v8.16b, v10.16b, #4 \n"
                    "ext v13.16b, v9.16b, v11.16b, #4 \n"
                    "fmla v0.4s, v12.4s, %11.s[1] \n"
                    "fmla v0.4s, v13.4s, %11.s[2] \n"
                    "ext v14.16b, v8.16b, v10.16b, #8 \n"
                    "fmla v0.4s, v14.4s, %11.s[3] \n"

                    "fmla v0.4s, v1.4s, %12.s[0] \n"
                    "fmla v0.4s, v2.4s, %12.s[1] \n"
                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    "fmla v0.4s, v5.4s, %12.s[2] \n"
                    "fmla v0.4s, v6.4s, %12.s[3] \n"
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    "fmla v0.4s, v7.4s, %13.s[0] \n"

                    "add x1, x1, #1 \n"
                    "st1 {v0.4s}, [%1], #16 \n"

                    "1: \n"
                    "cmp %0, x1 \n"
                    "beq 3f \n"

                    // mod2_left == 0
                    "2: \n"
                    "ld1 {v0.4s, v1.4s}, [%1] \n"

                    // v2.4s: 0 2 4 6 v3.4s: 1 3 5 7
                    "ld2 {v2.4s, v3.4s}, [%2], #32 \n"
                    "ld2 {v14.4s, v15.4s}, [%3], #32 \n"
                    // v4.4s: 8 10 12 14 v5.4s: 9 11 13 15
                    "ld2 {v4.4s, v5.4s}, [%2], #32 \n"
                    "ld2 {v16.4s, v17.4s}, [%3], #32 \n"
                    // v6.4s: 16 18 20 22 v7.4s: 17 19 21 23
                    "ld2 {v6.4s, v7.4s}, [%2] \n"
                    "ld2 {v18.4s, v19.4s}, [%3] \n"
                    // v8.4s: 2 4 6 8
                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    // v9.4s: 3 5 7 9
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    // v10.4s: 4 6 8 10
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    // v11.4s: 10 12 14 16
                    "ext v11.16b, v4.16b, v6.16b, #4 \n"
                    // v12.4s: 11 13 15 17
                    "ext v12.16b, v5.16b, v7.16b, #4 \n"
                    // v13.4s: 12 14 16 18
                    "ext v13.16b, v4.16b, v6.16b, #8 \n"
                    "fmla v0.4s, v2.4s, %7.s[0] \n"
                    "fmla v0.4s, v3.4s, %7.s[1] \n"
                    "fmla v0.4s, v8.4s, %7.s[2] \n"
                    "fmla v0.4s, v9.4s, %7.s[3] \n"
                    "fmla v0.4s, v10.4s, %8.s[0] \n"
                    "fmla v1.4s, v4.4s, %7.s[0] \n"
                    "fmla v1.4s, v5.4s, %7.s[1] \n"
                    "fmla v1.4s, v11.4s, %7.s[2] \n"
                    "fmla v1.4s, v12.4s, %7.s[3] \n"
                    "fmla v1.4s, v13.4s, %8.s[0] \n"

                    "ld2 {v2.4s, v3.4s}, [%4], #32 \n"
                    "ext v20.16b, v14.16b, v16.16b, #4 \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "ext v22.16b, v14.16b, v16.16b, #8 \n"
                    "fmla v0.4s, v14.4s, %8.s[1] \n"
                    "fmla v0.4s, v15.4s, %8.s[2] \n"
                    "fmla v0.4s, v20.4s, %8.s[3] \n"
                    "fmla v0.4s, v21.4s, %9.s[0] \n"
                    "fmla v0.4s, v22.4s, %9.s[1] \n"
                    "ld2 {v4.4s, v5.4s}, [%4], #32 \n"
                    "ext v23.16b, v16.16b, v18.16b, #4 \n"
                    "ext v24.16b, v17.16b, v19.16b, #4 \n"
                    "ext v14.16b, v16.16b, v18.16b, #8 \n"
                    "ld2 {v6.4s, v7.4s}, [%4] \n"
                    "fmla v1.4s, v16.4s, %8.s[1] \n"
                    "fmla v1.4s, v17.4s, %8.s[2] \n"
                    "fmla v1.4s, v23.4s, %8.s[3] \n"
                    "fmla v1.4s, v24.4s, %9.s[0] \n"
                    "fmla v1.4s, v14.4s, %9.s[1] \n"

                    "ld2 {v14.4s, v15.4s}, [%5], #32 \n"
                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    "ext v11.16b, v4.16b, v6.16b, #4 \n"
                    "ext v12.16b, v5.16b, v7.16b, #4 \n"
                    "ext v13.16b, v4.16b, v6.16b, #8 \n"
                    "ld2 {v16.4s, v17.4s}, [%5], #32 \n"
                    "fmla v0.4s, v2.4s, %9.s[2] \n"
                    "fmla v0.4s, v3.4s, %9.s[3] \n"
                    "fmla v0.4s, v8.4s, %10.s[0] \n"
                    "fmla v0.4s, v9.4s, %10.s[1] \n"
                    "fmla v0.4s, v10.4s, %10.s[2] \n"
                    "ld2 {v18.4s, v19.4s}, [%5] \n"
                    "fmla v1.4s, v4.4s, %9.s[2] \n"
                    "fmla v1.4s, v5.4s, %9.s[3] \n"
                    "fmla v1.4s, v11.4s, %10.s[0] \n"
                    "fmla v1.4s, v12.4s, %10.s[1] \n"
                    "fmla v1.4s, v13.4s, %10.s[2] \n"

                    "ld2 {v2.4s, v3.4s}, [%6], #32 \n"
                    "ext v20.16b, v14.16b, v16.16b, #4 \n"
                    "ext v21.16b, v15.16b, v17.16b, #4 \n"
                    "ext v22.16b, v14.16b, v16.16b, #8 \n"
                    "fmla v0.4s, v14.4s, %10.s[3] \n"
                    "fmla v0.4s, v15.4s, %11.s[0] \n"
                    "fmla v0.4s, v20.4s, %11.s[1] \n"
                    "fmla v0.4s, v21.4s, %11.s[2] \n"
                    "fmla v0.4s, v22.4s, %11.s[3] \n"
                    "ld2 {v4.4s, v5.4s}, [%6], #32 \n"
                    "ext v23.16b, v16.16b, v18.16b, #4 \n"
                    "ext v24.16b, v17.16b, v19.16b, #4 \n"
                    "ext v14.16b, v16.16b, v18.16b, #8 \n"
                    "ld2 {v6.4s, v7.4s}, [%6] \n"
                    "fmla v1.4s, v16.4s, %10.s[3] \n"
                    "fmla v1.4s, v17.4s, %11.s[0] \n"
                    "fmla v1.4s, v23.4s, %11.s[1] \n"
                    "fmla v1.4s, v24.4s, %11.s[2] \n"
                    "fmla v1.4s, v14.4s, %11.s[3] \n"

                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    "ext v11.16b, v4.16b, v6.16b, #4 \n"
                    "ext v12.16b, v5.16b, v7.16b, #4 \n"
                    "ext v13.16b, v4.16b, v6.16b, #8 \n"
                    "fmla v0.4s, v2.4s, %12.s[0] \n"
                    "fmla v0.4s, v3.4s, %12.s[1] \n"
                    "fmla v0.4s, v8.4s, %12.s[2] \n"
                    "fmla v0.4s, v9.4s, %12.s[3] \n"
                    "fmla v0.4s, v10.4s, %13.4s \n"
                    "fmla v1.4s, v4.4s, %12.s[0] \n"
                    "fmla v1.4s, v5.4s, %12.s[1] \n"
                    "fmla v1.4s, v11.4s, %12.s[2] \n"
                    "fmla v1.4s, v12.4s, %12.s[3] \n"
                    "fmla v1.4s, v13.4s, %13.4s \n"

                    "add x1, x1, #2 \n"
                    "cmp %0, x1 \n"
                    "st1 {v0.4s, v1.4s}, [%1], #32 \n"
                    "bne 2b \n"
                    "3: \n"

                    : "+r"(width), "+r"(outptr), "+r"(r0), "+r"(r1), "+r"(r2),
                      "+r"(r3), "+r"(r4)
                    : "w"(_k0123), "w"(_k4567), "w"(_k891011), "w"(_k12131415),
                      "w"(_k16171819), "w"(_k20212223), "w"(_k24242424),
                      "r"(mod2_left)
                    : "cc", "memory", "x1", "v0", "v1", "v2", "v3", "v4", "v5",
                      "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                      "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                      "v23", "v24");

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
        }

        filter += 25;
    }
}

// refer to function do_conv_7x7_stride2_asm_unroll2
static void do_conv_7x7_stride2(const float* src, const float* filter,
                                float* dst, size_t IH, size_t IW, size_t OH,
                                size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;
    size_t width = OW >> 2;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;
        const float* r5 = src_ptr + IW * 5;
        const float* r6 = src_ptr + IW * 6;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(filter);
        MEGDNN_SIMD_TYPE _k4567 = MEGDNN_SIMD_LOADU(filter + 4);
        MEGDNN_SIMD_TYPE _k891011 = MEGDNN_SIMD_LOADU(filter + 8);
        MEGDNN_SIMD_TYPE _k12131415 = MEGDNN_SIMD_LOADU(filter + 12);
        MEGDNN_SIMD_TYPE _k16171819 = MEGDNN_SIMD_LOADU(filter + 16);
        MEGDNN_SIMD_TYPE _k20212223 = MEGDNN_SIMD_LOADU(filter + 20);
        MEGDNN_SIMD_TYPE _k24252627 = MEGDNN_SIMD_LOADU(filter + 24);
        MEGDNN_SIMD_TYPE _k28293031 = MEGDNN_SIMD_LOADU(filter + 28);
        MEGDNN_SIMD_TYPE _k32333435 = MEGDNN_SIMD_LOADU(filter + 32);
        MEGDNN_SIMD_TYPE _k36373839 = MEGDNN_SIMD_LOADU(filter + 36);
        MEGDNN_SIMD_TYPE _k40414243 = MEGDNN_SIMD_LOADU(filter + 40);
        MEGDNN_SIMD_TYPE _k44454647 = MEGDNN_SIMD_LOADU(filter + 44);
        MEGDNN_SIMD_TYPE _k48484848 = MEGDNN_SIMD_SET1(filter[48]);

        for (size_t i = 0; i < OH; i++) {
            asm volatile(
                    "and x1, %8, #1 \n"
                    "cmp x1, #0 \n"
                    "mov x1, xzr \n"
                    "beq 1f \n"

                    // mod2_left == 1
                    "0: \n"
                    "ld1 {v0.4s}, [%0] \n"

                    // v1.4s: 0 2 4 6 v2.4s: 1 3 5 7
                    "ld2 {v1.4s, v2.4s}, [%1], #32 \n"
                    "ld2 {v10.4s, v11.4s}, [%2], #32 \n"
                    "ld2 {v3.4s, v4.4s}, [%1] \n"
                    "ld2 {v12.4s, v13.4s}, [%2] \n"
                    // v5.4s: 2 4 6 8
                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    // v6.4s: 3 5 7 9
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    // v7.4s: 4 6 8 10
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    // v8.4s: 5 7 9 11
                    "ext v8.16b, v2.16b, v4.16b, #8 \n"
                    // v9.4s: 6 8 10 12
                    "ext v9.16b, v1.16b, v3.16b, #12 \n"
                    "fmla v0.4s, v1.4s, %9.s[0] \n"
                    "fmla v0.4s, v2.4s, %9.s[1] \n"
                    "fmla v0.4s, v5.4s, %9.s[2] \n"
                    "fmla v0.4s, v6.4s, %9.s[3] \n"
                    "fmla v0.4s, v7.4s, %10.s[0] \n"
                    "fmla v0.4s, v8.4s, %10.s[1] \n"
                    "fmla v0.4s, v9.4s, %10.s[2] \n"

                    "ld2 {v1.4s, v2.4s}, [%3], #32 \n"
                    "ext v14.16b, v10.16b, v12.16b, #4 \n"
                    "ext v15.16b, v11.16b, v13.16b, #4 \n"
                    "ext v16.16b, v10.16b, v12.16b, #8 \n"
                    "ext v17.16b, v11.16b, v13.16b, #8 \n"
                    "ext v18.16b, v10.16b, v12.16b, #12 \n"
                    "ld2 {v3.4s, v4.4s}, [%3] \n"
                    "fmla v0.4s, v10.4s, %10.s[3] \n"
                    "fmla v0.4s, v11.4s, %11.s[0] \n"
                    "fmla v0.4s, v14.4s, %11.s[1] \n"
                    "fmla v0.4s, v15.4s, %11.s[2] \n"
                    "fmla v0.4s, v16.4s, %11.s[3] \n"
                    "fmla v0.4s, v17.4s, %12.s[0] \n"
                    "fmla v0.4s, v18.4s, %12.s[1] \n"

                    "ld2 {v10.4s, v11.4s}, [%4], #32 \n"
                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    "ext v8.16b, v2.16b, v4.16b, #8 \n"
                    "ext v9.16b, v1.16b, v3.16b, #12 \n"
                    "ld2 {v12.4s, v13.4s}, [%4] \n"
                    "fmla v0.4s, v1.4s, %12.s[2] \n"
                    "fmla v0.4s, v2.4s, %12.s[3] \n"
                    "fmla v0.4s, v5.4s, %13.s[0] \n"
                    "fmla v0.4s, v6.4s, %13.s[1] \n"
                    "fmla v0.4s, v7.4s, %13.s[2] \n"
                    "fmla v0.4s, v8.4s, %13.s[3] \n"
                    "fmla v0.4s, v9.4s, %14.s[0] \n"

                    "ld2 {v1.4s, v2.4s}, [%5], #32 \n"
                    "ext v14.16b, v10.16b, v12.16b, #4 \n"
                    "ext v15.16b, v11.16b, v13.16b, #4 \n"
                    "ext v16.16b, v10.16b, v12.16b, #8 \n"
                    "ext v17.16b, v11.16b, v13.16b, #8 \n"
                    "ext v18.16b, v10.16b, v12.16b, #12 \n"
                    "ld2 {v3.4s, v4.4s}, [%5] \n"
                    "fmla v0.4s, v10.4s, %14.s[1] \n"
                    "fmla v0.4s, v11.4s, %14.s[2] \n"
                    "fmla v0.4s, v14.4s, %14.s[3] \n"
                    "fmla v0.4s, v15.4s, %15.s[0] \n"
                    "fmla v0.4s, v16.4s, %15.s[1] \n"
                    "fmla v0.4s, v17.4s, %15.s[2] \n"
                    "fmla v0.4s, v18.4s, %15.s[3] \n"

                    "ld2 {v10.4s, v11.4s}, [%6], #32 \n"
                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    "ext v8.16b, v2.16b, v4.16b, #8 \n"
                    "ext v9.16b, v1.16b, v3.16b, #12 \n"
                    "ld2 {v12.4s, v13.4s}, [%6] \n"
                    "fmla v0.4s, v1.4s, %16.s[0] \n"
                    "fmla v0.4s, v2.4s, %16.s[1] \n"
                    "fmla v0.4s, v5.4s, %16.s[2] \n"
                    "fmla v0.4s, v6.4s, %16.s[3] \n"
                    "fmla v0.4s, v7.4s, %17.s[0] \n"
                    "fmla v0.4s, v8.4s, %17.s[1] \n"
                    "fmla v0.4s, v9.4s, %17.s[2] \n"

                    "ld2 {v1.4s, v2.4s}, [%7], #32 \n"
                    "ext v14.16b, v10.16b, v12.16b, #4 \n"
                    "ext v15.16b, v11.16b, v13.16b, #4 \n"
                    "ext v16.16b, v10.16b, v12.16b, #8 \n"
                    "ext v17.16b, v11.16b, v13.16b, #8 \n"
                    "ext v18.16b, v10.16b, v12.16b, #12 \n"
                    "ld2 {v3.4s, v4.4s}, [%7] \n"
                    "fmla v0.4s, v10.4s, %17.s[3] \n"
                    "fmla v0.4s, v11.4s, %18.s[0] \n"
                    "fmla v0.4s, v14.4s, %18.s[1] \n"
                    "fmla v0.4s, v15.4s, %18.s[2] \n"
                    "fmla v0.4s, v16.4s, %18.s[3] \n"
                    "fmla v0.4s, v17.4s, %19.s[0] \n"
                    "fmla v0.4s, v18.4s, %19.s[1] \n"

                    "ext v5.16b, v1.16b, v3.16b, #4 \n"
                    "ext v6.16b, v2.16b, v4.16b, #4 \n"
                    "ext v7.16b, v1.16b, v3.16b, #8 \n"
                    "ext v8.16b, v2.16b, v4.16b, #8 \n"
                    "ext v9.16b, v1.16b, v3.16b, #12 \n"
                    "fmla v0.4s, v1.4s, %19.s[2] \n"
                    "fmla v0.4s, v2.4s, %19.s[3] \n"
                    "fmla v0.4s, v5.4s, %20.s[0] \n"
                    "fmla v0.4s, v6.4s, %20.s[1] \n"
                    "fmla v0.4s, v7.4s, %20.s[2] \n"
                    "fmla v0.4s, v8.4s, %20.s[3] \n"
                    "fmla v0.4s, v9.4s, %21.4s \n"

                    "add x1, x1, #1 \n"
                    "st1 {v0.4s}, [%0], #16 \n"

                    "1: \n"
                    "cmp %8, x1 \n"
                    "beq 3f \n"

                    // mod2_left == 0
                    "2: \n"
                    "ld1 {v0.4s, v1.4s}, [%0] \n"

                    // v2.4s: 0 2 4 6 v3.4s: 1 3 5 7
                    "ld2 {v2.4s, v3.4s}, [%1], #32 \n"
                    // v4.4s: 8 10 12 14 v3.4s: 9 11 13 15
                    "ld2 {v4.4s, v5.4s}, [%1], #32 \n"
                    // v6.4s: 16 18 20 22 v7.4s: 17 19 21 23
                    "ld2 {v6.4s, v7.4s}, [%1] \n"
                    // v8.4s: 2 4 6 8
                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    // v9.4s: 3 5 7 9
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    // v10.4s: 4 6 8 10
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    // v11.4s: 5 7 9 11
                    "ext v11.16b, v3.16b, v5.16b, #8 \n"
                    // v12.4s: 6 8 10 12
                    "ext v12.16b, v2.16b, v4.16b, #12 \n"
                    "fmla v0.4s, v2.4s, %9.s[0] \n"
                    "fmla v0.4s, v3.4s, %9.s[1] \n"
                    "fmla v0.4s, v8.4s, %9.s[2] \n"
                    "fmla v0.4s, v9.4s, %9.s[3] \n"
                    "fmla v0.4s, v10.4s, %10.s[0] \n"
                    "fmla v0.4s, v11.4s, %10.s[1] \n"
                    "fmla v0.4s, v12.4s, %10.s[2] \n"
                    // v2.4s: 10 12 14 16
                    "ext v2.16b, v4.16b, v6.16b, #4 \n"
                    // v3.4s: 11 13 15 17
                    "ext v3.16b, v5.16b, v7.16b, #4 \n"
                    // v8.4s: 12 14 16 18
                    "ext v8.16b, v4.16b, v6.16b, #8 \n"
                    // v9.4s: 13 15 17 19
                    "ext v9.16b, v5.16b, v7.16b, #8 \n"
                    // v10.4s: 14 16 18 19
                    "ext v10.16b, v4.16b, v6.16b, #12 \n"
                    "fmla v1.4s, v4.4s, %9.s[0] \n"
                    "fmla v1.4s, v5.4s, %9.s[1] \n"
                    "fmla v1.4s, v2.4s, %9.s[2] \n"
                    "fmla v1.4s, v3.4s, %9.s[3] \n"
                    "fmla v1.4s, v8.4s, %10.s[0] \n"
                    "fmla v1.4s, v9.4s, %10.s[1] \n"
                    "fmla v1.4s, v10.4s, %10.s[2] \n"

                    "ld2 {v13.4s, v14.4s}, [%2], #32 \n"
                    "ld2 {v15.4s, v16.4s}, [%2], #32 \n"
                    "ld2 {v17.4s, v18.4s}, [%2] \n"
                    "ext v8.16b, v13.16b, v15.16b, #4 \n"
                    "ext v9.16b, v14.16b, v16.16b, #4 \n"
                    "ext v10.16b, v13.16b, v15.16b, #8 \n"
                    "ext v11.16b, v14.16b, v16.16b, #8 \n"
                    "ext v12.16b, v13.16b, v15.16b, #12 \n"
                    "fmla v0.4s, v13.4s, %10.s[3] \n"
                    "fmla v0.4s, v14.4s, %11.s[0] \n"
                    "fmla v0.4s, v8.4s, %11.s[1] \n"
                    "fmla v0.4s, v9.4s, %11.s[2] \n"
                    "fmla v0.4s, v10.4s, %11.s[3] \n"
                    "fmla v0.4s, v11.4s, %12.s[0] \n"
                    "fmla v0.4s, v12.4s, %12.s[1] \n"
                    "ext v13.16b, v15.16b, v17.16b, #4 \n"
                    "ext v14.16b, v16.16b, v18.16b, #4 \n"
                    "ext v8.16b, v15.16b, v17.16b, #8 \n"
                    "ext v9.16b, v16.16b, v18.16b, #8 \n"
                    "ext v10.16b, v15.16b, v17.16b, #12 \n"
                    "fmla v1.4s, v15.4s, %10.s[3] \n"
                    "fmla v1.4s, v16.4s, %11.s[0] \n"
                    "fmla v1.4s, v13.4s, %11.s[1] \n"
                    "fmla v1.4s, v14.4s, %11.s[2] \n"
                    "fmla v1.4s, v8.4s, %11.s[3] \n"
                    "fmla v1.4s, v9.4s, %12.s[0] \n"
                    "fmla v1.4s, v10.4s, %12.s[1] \n"

                    "ld2 {v2.4s, v3.4s}, [%3], #32 \n"
                    "ld2 {v4.4s, v5.4s}, [%3], #32 \n"
                    "ld2 {v6.4s, v7.4s}, [%3] \n"
                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    "ext v11.16b, v3.16b, v5.16b, #8 \n"
                    "ext v12.16b, v2.16b, v4.16b, #12 \n"
                    "fmla v0.4s, v2.4s, %12.s[2] \n"
                    "fmla v0.4s, v3.4s, %12.s[3] \n"
                    "fmla v0.4s, v8.4s, %13.s[0] \n"
                    "fmla v0.4s, v9.4s, %13.s[1] \n"
                    "fmla v0.4s, v10.4s, %13.s[2] \n"
                    "fmla v0.4s, v11.4s, %13.s[3] \n"
                    "fmla v0.4s, v12.4s, %14.s[0] \n"
                    "ext v2.16b, v4.16b, v6.16b, #4 \n"
                    "ext v3.16b, v5.16b, v7.16b, #4 \n"
                    "ext v8.16b, v4.16b, v6.16b, #8 \n"
                    "ext v9.16b, v5.16b, v7.16b, #8 \n"
                    "ext v10.16b, v4.16b, v6.16b, #12 \n"
                    "fmla v1.4s, v4.4s, %12.s[2] \n"
                    "fmla v1.4s, v5.4s, %12.s[3] \n"
                    "fmla v1.4s, v2.4s, %13.s[0] \n"
                    "fmla v1.4s, v3.4s, %13.s[1] \n"
                    "fmla v1.4s, v8.4s, %13.s[2] \n"
                    "fmla v1.4s, v9.4s, %13.s[3] \n"
                    "fmla v1.4s, v10.4s, %14.s[0] \n"

                    "ld2 {v13.4s, v14.4s}, [%4], #32 \n"
                    "ld2 {v15.4s, v16.4s}, [%4], #32 \n"
                    "ld2 {v17.4s, v18.4s}, [%4] \n"
                    "ext v8.16b, v13.16b, v15.16b, #4 \n"
                    "ext v9.16b, v14.16b, v16.16b, #4 \n"
                    "ext v10.16b, v13.16b, v15.16b, #8 \n"
                    "ext v11.16b, v14.16b, v16.16b, #8 \n"
                    "ext v12.16b, v13.16b, v15.16b, #12 \n"
                    "fmla v0.4s, v13.4s, %14.s[1] \n"
                    "fmla v0.4s, v14.4s, %14.s[2] \n"
                    "fmla v0.4s, v8.4s, %14.s[3] \n"
                    "fmla v0.4s, v9.4s, %15.s[0] \n"
                    "fmla v0.4s, v10.4s, %15.s[1] \n"
                    "fmla v0.4s, v11.4s, %15.s[2] \n"
                    "fmla v0.4s, v12.4s, %15.s[3] \n"
                    "ext v13.16b, v15.16b, v17.16b, #4 \n"
                    "ext v14.16b, v16.16b, v18.16b, #4 \n"
                    "ext v8.16b, v15.16b, v17.16b, #8 \n"
                    "ext v9.16b, v16.16b, v18.16b, #8 \n"
                    "ext v10.16b, v15.16b, v17.16b, #12 \n"
                    "fmla v1.4s, v15.4s, %14.s[1] \n"
                    "fmla v1.4s, v16.4s, %14.s[2] \n"
                    "fmla v1.4s, v13.4s, %14.s[3] \n"
                    "fmla v1.4s, v14.4s, %15.s[0] \n"
                    "fmla v1.4s, v8.4s, %15.s[1] \n"
                    "fmla v1.4s, v9.4s, %15.s[2] \n"
                    "fmla v1.4s, v10.4s, %15.s[3] \n"

                    "ld2 {v2.4s, v3.4s}, [%5], #32 \n"
                    "ld2 {v4.4s, v5.4s}, [%5], #32 \n"
                    "ld2 {v6.4s, v7.4s}, [%5] \n"
                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    "ext v11.16b, v3.16b, v5.16b, #8 \n"
                    "ext v12.16b, v2.16b, v4.16b, #12 \n"
                    "fmla v0.4s, v2.4s, %16.s[0] \n"
                    "fmla v0.4s, v3.4s, %16.s[1] \n"
                    "fmla v0.4s, v8.4s, %16.s[2] \n"
                    "fmla v0.4s, v9.4s, %16.s[3] \n"
                    "fmla v0.4s, v10.4s, %17.s[0] \n"
                    "fmla v0.4s, v11.4s, %17.s[1] \n"
                    "fmla v0.4s, v12.4s, %17.s[2] \n"
                    "ext v2.16b, v4.16b, v6.16b, #4 \n"
                    "ext v3.16b, v5.16b, v7.16b, #4 \n"
                    "ext v8.16b, v4.16b, v6.16b, #8 \n"
                    "ext v9.16b, v5.16b, v7.16b, #8 \n"
                    "ext v10.16b, v4.16b, v6.16b, #12 \n"
                    "fmla v1.4s, v4.4s, %16.s[0] \n"
                    "fmla v1.4s, v5.4s, %16.s[1] \n"
                    "fmla v1.4s, v2.4s, %16.s[2] \n"
                    "fmla v1.4s, v3.4s, %16.s[3] \n"
                    "fmla v1.4s, v8.4s, %17.s[0] \n"
                    "fmla v1.4s, v9.4s, %17.s[1] \n"
                    "fmla v1.4s, v10.4s, %17.s[2] \n"

                    "ld2 {v13.4s, v14.4s}, [%6], #32 \n"
                    "ld2 {v15.4s, v16.4s}, [%6], #32 \n"
                    "ld2 {v17.4s, v18.4s}, [%6] \n"
                    "ext v8.16b, v13.16b, v15.16b, #4 \n"
                    "ext v9.16b, v14.16b, v16.16b, #4 \n"
                    "ext v10.16b, v13.16b, v15.16b, #8 \n"
                    "ext v11.16b, v14.16b, v16.16b, #8 \n"
                    "ext v12.16b, v13.16b, v15.16b, #12 \n"
                    "fmla v0.4s, v13.4s, %17.s[3] \n"
                    "fmla v0.4s, v14.4s, %18.s[0] \n"
                    "fmla v0.4s, v8.4s, %18.s[1] \n"
                    "fmla v0.4s, v9.4s, %18.s[2] \n"
                    "fmla v0.4s, v10.4s, %18.s[3] \n"
                    "fmla v0.4s, v11.4s, %19.s[0] \n"
                    "fmla v0.4s, v12.4s, %19.s[1] \n"
                    "ext v13.16b, v15.16b, v17.16b, #4 \n"
                    "ext v14.16b, v16.16b, v18.16b, #4 \n"
                    "ext v8.16b, v15.16b, v17.16b, #8 \n"
                    "ext v9.16b, v16.16b, v18.16b, #8 \n"
                    "ext v10.16b, v15.16b, v17.16b, #12 \n"
                    "fmla v1.4s, v15.4s, %17.s[3] \n"
                    "fmla v1.4s, v16.4s, %18.s[0] \n"
                    "fmla v1.4s, v13.4s, %18.s[1] \n"
                    "fmla v1.4s, v14.4s, %18.s[2] \n"
                    "fmla v1.4s, v8.4s, %18.s[3] \n"
                    "fmla v1.4s, v9.4s, %19.s[0] \n"
                    "fmla v1.4s, v10.4s, %19.s[1] \n"

                    "ld2 {v2.4s, v3.4s}, [%7], #32 \n"
                    "ld2 {v4.4s, v5.4s}, [%7], #32 \n"
                    "ld2 {v6.4s, v7.4s}, [%7] \n"
                    "ext v8.16b, v2.16b, v4.16b, #4 \n"
                    "ext v9.16b, v3.16b, v5.16b, #4 \n"
                    "ext v10.16b, v2.16b, v4.16b, #8 \n"
                    "ext v11.16b, v3.16b, v5.16b, #8 \n"
                    "ext v12.16b, v2.16b, v4.16b, #12 \n"
                    "fmla v0.4s, v2.4s, %19.s[2] \n"
                    "fmla v0.4s, v3.4s, %19.s[3] \n"
                    "fmla v0.4s, v8.4s, %20.s[0] \n"
                    "fmla v0.4s, v9.4s, %20.s[1] \n"
                    "fmla v0.4s, v10.4s, %20.s[2] \n"
                    "fmla v0.4s, v11.4s, %20.s[3] \n"
                    "fmla v0.4s, v12.4s, %21.4s \n"
                    "ext v2.16b, v4.16b, v6.16b, #4 \n"
                    "ext v3.16b, v5.16b, v7.16b, #4 \n"
                    "ext v8.16b, v4.16b, v6.16b, #8 \n"
                    "ext v9.16b, v5.16b, v7.16b, #8 \n"
                    "ext v10.16b, v4.16b, v6.16b, #12 \n"
                    "fmla v1.4s, v4.4s, %19.s[2] \n"
                    "fmla v1.4s, v5.4s, %19.s[3] \n"
                    "fmla v1.4s, v2.4s, %20.s[0] \n"
                    "fmla v1.4s, v3.4s, %20.s[1] \n"
                    "fmla v1.4s, v8.4s, %20.s[2] \n"
                    "fmla v1.4s, v9.4s, %20.s[3] \n"
                    "fmla v1.4s, v10.4s, %21.4s \n"

                    "add x1, x1, #2 \n"
                    "st1 {v0.4s, v1.4s}, [%0], #32 \n"
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
                    : "cc", "memory", "x1", "v0", "v1", "v2", "v3", "v4", "v5",
                      "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                      "v15", "v16", "v17", "v18");

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
}  // namespace fp32
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
