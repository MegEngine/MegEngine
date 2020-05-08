/**
 * \file dnn/src/arm_common/utils.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;

namespace {

template <typename dtype>
void transpose_naive(const dtype *src, dtype *dst,
        int lda, int ldb, int n, int m)
{
    rep(i, n) rep(j, m) {
        dst[i*ldb + j] = src[j*lda + i];
    }
}

void transpose_4x4_neon(const float *src, float *dst, int lda, int ldb)
{
    float32x4x2_t a0, a1;
    a0.val[0] = vld1q_f32(src + 0*lda);
    a0.val[1] = vld1q_f32(src + 1*lda);
    a1.val[0] = vld1q_f32(src + 2*lda);
    a1.val[1] = vld1q_f32(src + 3*lda);
    float32x4x2_t b0 = vzipq_f32(a0.val[0], a1.val[0]);
    float32x4x2_t b1 = vzipq_f32(a0.val[1], a1.val[1]);
    float32x4x2_t c0 = vzipq_f32(b0.val[0], b1.val[0]);
    float32x4x2_t c1 = vzipq_f32(b0.val[1], b1.val[1]);
    vst1q_f32(dst + 0*ldb, c0.val[0]);
    vst1q_f32(dst + 1*ldb, c0.val[1]);
    vst1q_f32(dst + 2*ldb, c1.val[0]);
    vst1q_f32(dst + 3*ldb, c1.val[1]);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void transpose_8x8_neon(const dt_float16 *src, dt_float16 *dst, int lda, int ldb)
{
    const __fp16* src_ptr = reinterpret_cast<const __fp16*>(src);
    __fp16* dst_ptr = reinterpret_cast<__fp16*>(dst);
    float16x8x4_t a0, a1;
    a0.val[0] = vld1q_f16(src_ptr + 0*lda); // A0A1A2A3A4A5A6A7
    a0.val[1] = vld1q_f16(src_ptr + 1*lda); // B0B1B2B3B4B5B6B7
    a0.val[2] = vld1q_f16(src_ptr + 2*lda); // C0C1C2C3C4C5C6C7
    a0.val[3] = vld1q_f16(src_ptr + 3*lda); // D0D1D2D3D4D5D6D7
    a1.val[0] = vld1q_f16(src_ptr + 4*lda); // E0E1E2E3E4E5E6E7
    a1.val[1] = vld1q_f16(src_ptr + 5*lda); // F0F1F2F3F4F5F6F7
    a1.val[2] = vld1q_f16(src_ptr + 6*lda); // G0G1G2G3G4G5G6G7
    a1.val[3] = vld1q_f16(src_ptr + 7*lda); // H0H1H2H3H4H5H6H7

    float16x8x2_t b0 = vzipq_f16(a0.val[0], a1.val[0]); // A0E0A1E1A2E2A3E3 A4E4A5E5A6E6A7E7
    float16x8x2_t b1 = vzipq_f16(a0.val[2], a1.val[2]); // C0G0C1G1C2G2C3G3 C4G4C5G5C6G6C7G7
    float16x8x2_t c0 = vzipq_f16(a0.val[1], a1.val[1]); // B0F0B1F1B2F2B3F3 B4F4B5F5B6F6B7F7
    float16x8x2_t c1 = vzipq_f16(a0.val[3], a1.val[3]); // D0H0D1H1D2H2D3H3 D4H4D5H5D6H6D7H7

    float16x8x2_t d0 = vzipq_f16(b0.val[0], b1.val[0]); // A0C0E0G0A1C1E1G1 A2C2E2G2A3C3E3G3
    float16x8x2_t d1 = vzipq_f16(c0.val[0], c1.val[0]); // B0D0F0H0B1D1F1H1 B2D2F2H2B3D3F3H3
    float16x8x2_t e0 = vzipq_f16(d0.val[0], d1.val[0]); // A0B0C0D0E0F0G0H0 A1B1C1D1E1F1G1H1
    float16x8x2_t e1 = vzipq_f16(d0.val[1], d1.val[1]); // A2B2C2D2E2F2G2H2 A3B3C3D3E3F3G3H3

    float16x8x2_t f0 = vzipq_f16(b0.val[1], b1.val[1]); // A4C4E4G4A5C5E5G5 A6C6E6G6A7C7E7G7
    float16x8x2_t f1 = vzipq_f16(c0.val[1], c1.val[1]); // B4D4F4H4B5D5F5H5 B6D6E6G6B7D7E7H7
    float16x8x2_t g0 = vzipq_f16(f0.val[0], f1.val[0]); // A4B4C4D4E4F4G4H4 A5B5C5D5E5F5G5H5
    float16x8x2_t g1 = vzipq_f16(f0.val[1], f1.val[1]); // A6B6C6D6E6F6G6H6 A7B7C7D7E7F7G7H7

    vst1q_f16(dst_ptr + 0*ldb, e0.val[0]);
    vst1q_f16(dst_ptr + 1*ldb, e0.val[1]);
    vst1q_f16(dst_ptr + 2*ldb, e1.val[0]);
    vst1q_f16(dst_ptr + 3*ldb, e1.val[1]);
    vst1q_f16(dst_ptr + 4*ldb, g0.val[0]);
    vst1q_f16(dst_ptr + 5*ldb, g0.val[1]);
    vst1q_f16(dst_ptr + 6*ldb, g1.val[0]);
    vst1q_f16(dst_ptr + 7*ldb, g1.val[1]);
}
#endif

} // anonymous namespace

namespace megdnn {

template <>
void transpose(const float* src, float* dst, size_t m, size_t n, ptrdiff_t lds,
               ptrdiff_t ldd) {
    if (lds == -1) {
        lds = n;
    }
    if (ldd == -1) {
        ldd = m;
    }

    for (size_t is = 0; is < n; is += 16) {
        for (size_t js = 0; js < m; js += 16) {
            auto ie = std::min(is + 16, n), je = std::min(js + 16, m), i = is;
            for (; i + 4 <= ie; i += 4) {
                auto j = js;
                for (; j + 4 <= je; j += 4) {
                    transpose_4x4_neon(src + j * lds + i, dst + i * ldd + j,
                                       lds, ldd);
                }
                if (j < je) {
                    transpose_naive(src + j * lds + i, dst + i * ldd + j, lds,
                                    ldd, 4, je - j);
                }
            }
            if (i < ie) {
                transpose_naive(src + js * lds + i, dst + i * ldd + js, lds,
                                ldd, ie - i, je - js);
            }
        }
    }
}

template<typename dtype>
void transpose_knc2nsck_helper(const dtype *src, dtype *dst,
        size_t k, size_t n, size_t c, size_t n_stride) {
    if (n_stride == k * c) {
        // dst is contiguous
        transpose(src, dst, k, n * c);
    } else {
        for (size_t i = 0; i < n; ++ i) {
            transpose(src + i * c, dst + i * n_stride,
                    k, c, n * c);
        }
    }
}

template <>
void transpose_knc2nsck(const float *src, float *dst,
        size_t k, size_t n, size_t c, size_t n_stride) {
    transpose_knc2nsck_helper(src, dst, k, n, c, n_stride);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
void transpose(const dt_float16* src, dt_float16* dst, size_t m, size_t n,
               ptrdiff_t lds, ptrdiff_t ldd) {
    if (lds == -1) {
        lds = n;
    }
    if (ldd == -1) {
        ldd = m;
    }

    for (size_t is = 0; is < n; is += 16) {
        for (size_t js = 0; js < m; js += 16) {
            auto ie = std::min(is + 16, n), je = std::min(js + 16, m), i = is;
            for (; i + 8 <= ie; i += 8) {
                auto j = js;
                for (; j + 8 <= je; j += 8) {
                    transpose_8x8_neon(src + j * lds + i, dst + i * ldd + j,
                                       lds, ldd);
                }
                if (j < je) {
                    transpose_naive(src + j * lds + i, dst + i * ldd + j, lds,
                                    ldd, 8, je - j);
                }
            }
            if (i < ie) {
                transpose_naive(src + js * lds + i, dst + i * ldd + js, lds,
                                ldd, ie - i, je - js);
            }
        }
    }
}

template <>
void transpose_knc2nsck(const dt_float16* src, dt_float16* dst, size_t k,
                        size_t n, size_t c, size_t n_stride) {
    transpose_knc2nsck_helper(src, dst, k, n, c, n_stride);
}
#endif

} // namespace megdnn
// vim: syntax=cpp.doxygen
