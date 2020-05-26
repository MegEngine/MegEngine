/**
 * \file src/x86/conv_bias/int8/avx2_chanwise_kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/x86/conv_bias/int8/avx2_chanwise_kern.h"
#include <immintrin.h>
#include "src/common/unroll_macro.h"
#include "src/x86/conv_bias/int8/common_helper.h"
#include "src/x86/elemwise_op.h"
#ifdef WIN32
#include <smmintrin.h>
#endif

namespace megdnn {
namespace x86 {
#define load_filter(i) __m128i k_##i = _mm_set1_epi8(*(filter + i));
#define load_src0(i) \
    __m256i cvt16_src##i##0 = _mm256_cvtepi8_epi16_from_ptr(r##i);
#define load_src1(i) \
    __m256i cvt16_src##i##1 = _mm256_cvtepi8_epi16_from_ptr(r##i + 1);
#define load_src2(i) \
    __m256i cvt16_src##i##2 = _mm256_cvtepi8_epi16_from_ptr(r##i + 2);
#define load_src3(i) \
    __m256i cvt16_src##i##3 = _mm256_cvtepi8_epi16_from_ptr(r##i + 3);
#define load_src4(i) \
    __m256i cvt16_src##i##4 = _mm256_cvtepi8_epi16_from_ptr(r##i + 4);
#define load_src5(i) \
    __m256i cvt16_src##i##5 = _mm256_cvtepi8_epi16_from_ptr(r##i + 5);
#define load_src6(i) \
    __m256i cvt16_src##i##6 = _mm256_cvtepi8_epi16_from_ptr(r##i + 6);
#define load_src7(i) \
    __m256i cvt16_src##i##7 = _mm256_cvtepi8_epi16_from_ptr(r##i + 7);
#define load_src16(i) \
    __m256i cvt16_src##i##16 = _mm256_cvtepi8_epi16_from_ptr(r##i + 16);
#define load_src18(i) \
    __m256i cvt16_src##i##18 = _mm256_cvtepi8_epi16_from_ptr(r##i + 18);
#define load_src20(i) \
    __m256i cvt16_src##i##20 = _mm256_cvtepi8_epi16_from_ptr(r##i + 20);
#define load_src22(i) \
    __m256i cvt16_src##i##22 = _mm256_cvtepi8_epi16_from_ptr(r##i + 22);
namespace avx2_chanwise_stride1 {

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride1_2x2_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    size_t tail_step = IW - OW;
    int8_t* dst0 = dst;
    int8_t* dst1 = dst + OW;
    int32_t* out_ptr0 = temp;
    int32_t* out_ptr1 = temp + OW;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;

    UNROLL_CALL0(4, load_filter)

#define pack_filter(i, j) __m128i k_##i##j = _mm_unpacklo_epi8(k_##i, k_##j)
    pack_filter(0, 1);
    pack_filter(2, 3);

    __m256i bias_val;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }
#define cvt_filter(i, j) __m256i filter_##i##j = _mm256_cvtepi8_epi16(k_##i##j)
    cvt_filter(0, 1);
    cvt_filter(2, 3);

    size_t width = OW >> 4;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(3, load_src0)
            UNROLL_CALL0(3, load_src1)

            __m256i sum0_odd, sum0_even, sum1_odd, sum1_even;
            __m256i tmp0_odd, tmp0_even, tmp1_odd, tmp1_even, tmp2_odd,
                    tmp2_even, tmp3_odd, tmp3_even;

            tmp0_odd = _mm256_madd_epi16(cvt16_src00, filter_01);
            tmp0_even = _mm256_madd_epi16(cvt16_src01, filter_01);

            tmp1_odd = _mm256_madd_epi16(cvt16_src10, filter_23);
            tmp1_even = _mm256_madd_epi16(cvt16_src11, filter_23);

            tmp3_odd = _mm256_madd_epi16(cvt16_src10, filter_01);
            tmp3_even = _mm256_madd_epi16(cvt16_src11, filter_01);

            tmp2_odd = _mm256_madd_epi16(cvt16_src20, filter_23);
            tmp2_even = _mm256_madd_epi16(cvt16_src21, filter_23);

            sum0_odd = _mm256_add_epi32(tmp0_odd, tmp1_odd);
            sum0_even = _mm256_add_epi32(tmp0_even, tmp1_even);

            __m256i sum_odd = _mm256_unpacklo_epi32(sum0_odd, sum0_even);
            __m256i sum_even = _mm256_unpackhi_epi32(sum0_odd, sum0_even);

            //! switch_mask_low   = {00100000} = 32
            //! switch_mask_high  = {00110001} = 49
            __m256i sum_left = _mm256_permute2f128_si256(sum_odd, sum_even, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd, sum_even, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));

            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            sum1_odd = _mm256_add_epi32(tmp3_odd, tmp2_odd);
            sum1_even = _mm256_add_epi32(tmp3_even, tmp2_even);

            __m256i sum_1_odd = _mm256_unpacklo_epi32(sum1_odd, sum1_even);
            __m256i sum_1_even = _mm256_unpackhi_epi32(sum1_odd, sum1_even);

            __m256i sum_1_left =
                    _mm256_permute2f128_si256(sum_1_odd, sum_1_even, 32);
            __m256i sum_1_right =
                    _mm256_permute2f128_si256(sum_1_odd, sum_1_even, 49);

            sum_1_left = _mm256_add_epi32(sum_1_left, bias_val);
            sum_1_right = _mm256_add_epi32(sum_1_right, bias_val);

            if (is_quantized) {
                op({{sum_1_left, sum_1_right}},
                   reinterpret_cast<dt_qint8*>(dst1));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr1), sum_1_left);
                _mm256_storeu_si256((__m256i*)(out_ptr1 + 8), sum_1_right);
            }
            r0 += 16;
            r1 += 16;
            r2 += 16;
            dst0 += 16;
            dst1 += 16;
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;

        dst0 += OW;
        dst1 += OW;
        out_ptr0 += OW;
        out_ptr1 += OW;
    }

    for (; h < OH; h++) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(2, load_src0)
            UNROLL_CALL0(2, load_src1)

            __m256i sum0_odd, sum0_even;
            __m256i tmp0_odd, tmp0_even, tmp1_odd, tmp1_even;

            tmp0_odd = _mm256_madd_epi16(cvt16_src00, filter_01);
            tmp0_even = _mm256_madd_epi16(cvt16_src01, filter_01);

            tmp1_odd = _mm256_madd_epi16(cvt16_src10, filter_23);
            tmp1_even = _mm256_madd_epi16(cvt16_src11, filter_23);

            sum0_odd = _mm256_add_epi32(tmp0_odd, tmp1_odd);
            sum0_even = _mm256_add_epi32(tmp0_even, tmp1_even);

            __m256i sum_odd = _mm256_unpacklo_epi32(sum0_odd, sum0_even);
            __m256i sum_even = _mm256_unpackhi_epi32(sum0_odd, sum0_even);

            __m256i sum_left = _mm256_permute2f128_si256(sum_odd, sum_even, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd, sum_even, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 16;
            r1 += 16;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step;
        r1 += tail_step;
    }
    MEGDNN_MARK_USED_VAR(IH);
#undef pack_filter
#undef cvt_filter
}

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride1_3x3_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    size_t tail_step = IW - OW;
    int32_t* out_ptr0 = temp;
    int32_t* out_ptr1 = temp + OW;
    int8_t* dst0 = dst;
    int8_t* dst1 = dst + OW;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;
    const int8_t* r3 = src + 3 * IW;

    uint8_t fill_zero = 0;
    UNROLL_CALL0(9, load_filter)

    __m128i k_fill = _mm_set1_epi8(fill_zero);

    __m128i k01 = _mm_unpacklo_epi8(k_0, k_1);
    __m128i k20 = _mm_unpacklo_epi8(k_2, k_fill);

    __m128i k34 = _mm_unpacklo_epi8(k_3, k_4);
    __m128i k50 = _mm_unpacklo_epi8(k_5, k_fill);

    __m128i k67 = _mm_unpacklo_epi8(k_6, k_7);
    __m128i k80 = _mm_unpacklo_epi8(k_8, k_fill);

    __m256i bias_val;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }

    //! cvt i8 --> i16
    __m256i filter_01 = _mm256_cvtepi8_epi16(k01);
    __m256i filter_20 = _mm256_cvtepi8_epi16(k20);
    __m256i filter_34 = _mm256_cvtepi8_epi16(k34);
    __m256i filter_50 = _mm256_cvtepi8_epi16(k50);
    __m256i filter_67 = _mm256_cvtepi8_epi16(k67);
    __m256i filter_80 = _mm256_cvtepi8_epi16(k80);

    size_t width = OW >> 4;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(4, load_src0)
            UNROLL_CALL0(4, load_src1)
            UNROLL_CALL0(4, load_src2)
            UNROLL_CALL0(4, load_src3)

            __m256i sum00_odd, sum00_even, sum11_odd, sum11_even, sum22_odd,
                    sum22_even;
            __m256i sum11_odd_01, sum11_even_01, sum22_odd_01, sum22_even_01,
                    sum33_odd, sum33_even;
            __m256i temp0, temp1;

            temp0 = _mm256_madd_epi16(cvt16_src00, filter_01);
            temp1 = _mm256_madd_epi16(cvt16_src02, filter_20);
            sum00_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src01, filter_01);
            temp1 = _mm256_madd_epi16(cvt16_src03, filter_20);
            sum00_even = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src10, filter_34);
            temp1 = _mm256_madd_epi16(cvt16_src12, filter_50);
            sum11_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src11, filter_34);
            temp1 = _mm256_madd_epi16(cvt16_src13, filter_50);
            sum11_even = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src10, filter_01);
            temp1 = _mm256_madd_epi16(cvt16_src12, filter_20);
            sum11_odd_01 = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src11, filter_01);
            temp1 = _mm256_madd_epi16(cvt16_src13, filter_20);
            sum11_even_01 = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src20, filter_67);
            temp1 = _mm256_madd_epi16(cvt16_src22, filter_80);
            sum22_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src21, filter_67);
            temp1 = _mm256_madd_epi16(cvt16_src23, filter_80);
            sum22_even = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src20, filter_34);
            temp1 = _mm256_madd_epi16(cvt16_src22, filter_50);
            sum22_odd_01 = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src21, filter_34);
            temp1 = _mm256_madd_epi16(cvt16_src23, filter_50);
            sum22_even_01 = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src30, filter_67);
            temp1 = _mm256_madd_epi16(cvt16_src32, filter_80);
            sum33_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src31, filter_67);
            temp1 = _mm256_madd_epi16(cvt16_src33, filter_80);
            sum33_even = _mm256_add_epi32(temp0, temp1);

            sum00_odd = _mm256_add_epi32(sum00_odd, sum11_odd);
            sum00_odd = _mm256_add_epi32(sum00_odd, sum22_odd);

            sum00_even = _mm256_add_epi32(sum00_even, sum11_even);
            sum00_even = _mm256_add_epi32(sum00_even, sum22_even);

            __m256i sum_odd = _mm256_unpacklo_epi32(sum00_odd, sum00_even);
            __m256i sum_even = _mm256_unpackhi_epi32(sum00_odd, sum00_even);

            __m256i sum_left = _mm256_permute2f128_si256(sum_odd, sum_even, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd, sum_even, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            sum11_odd_01 = _mm256_add_epi32(sum11_odd_01, sum22_odd_01);
            sum11_odd_01 = _mm256_add_epi32(sum11_odd_01, sum33_odd);

            sum11_even_01 = _mm256_add_epi32(sum11_even_01, sum22_even_01);
            sum11_even_01 = _mm256_add_epi32(sum11_even_01, sum33_even);

            __m256i sum_oh1_odd =
                    _mm256_unpacklo_epi32(sum11_odd_01, sum11_even_01);
            __m256i sum_oh1_even =
                    _mm256_unpackhi_epi32(sum11_odd_01, sum11_even_01);

            __m256i sum1_left =
                    _mm256_permute2f128_si256(sum_oh1_odd, sum_oh1_even, 32);
            __m256i sum1_right =
                    _mm256_permute2f128_si256(sum_oh1_odd, sum_oh1_even, 49);

            sum1_left = _mm256_add_epi32(sum1_left, bias_val);
            sum1_right = _mm256_add_epi32(sum1_right, bias_val);

            if (is_quantized) {
                op({{sum1_left, sum1_right}},
                   reinterpret_cast<dt_qint8*>(dst1));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr1), sum1_left);
                _mm256_storeu_si256((__m256i*)(out_ptr1 + 8), sum1_right);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            dst0 += 16;
            dst1 += 16;
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;

        dst0 += OW;
        dst1 += OW;
        out_ptr0 += OW;
        out_ptr1 += OW;
    }

    for (; h < OH; h++) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(3, load_src0)
            UNROLL_CALL0(3, load_src1)
            UNROLL_CALL0(3, load_src2)
            UNROLL_CALL0(3, load_src3)

            __m256i sum00_odd, sum00_even, sum11_odd, sum11_even, sum22_odd,
                    sum22_even;
            __m256i temp0, temp1;

            temp0 = _mm256_madd_epi16(cvt16_src00, filter_01);
            temp1 = _mm256_madd_epi16(cvt16_src02, filter_20);
            sum00_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src01, filter_01);
            temp1 = _mm256_madd_epi16(cvt16_src03, filter_20);
            sum00_even = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src10, filter_34);
            temp1 = _mm256_madd_epi16(cvt16_src12, filter_50);
            sum11_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src11, filter_34);
            temp1 = _mm256_madd_epi16(cvt16_src13, filter_50);
            sum11_even = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src20, filter_67);
            temp1 = _mm256_madd_epi16(cvt16_src22, filter_80);
            sum22_odd = _mm256_add_epi32(temp0, temp1);

            temp0 = _mm256_madd_epi16(cvt16_src21, filter_67);
            temp1 = _mm256_madd_epi16(cvt16_src23, filter_80);
            sum22_even = _mm256_add_epi32(temp0, temp1);

            sum00_odd = _mm256_add_epi32(sum00_odd, sum11_odd);
            sum00_odd = _mm256_add_epi32(sum00_odd, sum22_odd);

            sum00_even = _mm256_add_epi32(sum00_even, sum11_even);
            sum00_even = _mm256_add_epi32(sum00_even, sum22_even);

            __m256i sum_odd = _mm256_unpacklo_epi32(sum00_odd, sum00_even);
            __m256i sum_even = _mm256_unpackhi_epi32(sum00_odd, sum00_even);

            __m256i sum_left = _mm256_permute2f128_si256(sum_odd, sum_even, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd, sum_even, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
    }
}

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride1_5x5_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    size_t tail_step = IW - OW;
    int8_t* dst0 = dst;
    int8_t* dst1 = dst + OW;
    int32_t* out_ptr0 = temp;
    int32_t* out_ptr1 = temp + OW;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;
    const int8_t* r3 = src + 3 * IW;
    const int8_t* r4 = src + 4 * IW;
    const int8_t* r5 = src + 5 * IW;

    uint8_t fill_zero = 0;
    UNROLL_CALL0(25, load_filter)

    __m128i k_fill = _mm_set1_epi8(fill_zero);

    __m128i k01 = _mm_unpacklo_epi8(k_0, k_1);
    __m128i k23 = _mm_unpacklo_epi8(k_2, k_3);
    __m128i k40 = _mm_unpacklo_epi8(k_4, k_fill);

    __m128i k56 = _mm_unpacklo_epi8(k_5, k_6);
    __m128i k78 = _mm_unpacklo_epi8(k_7, k_8);
    __m128i k90 = _mm_unpacklo_epi8(k_9, k_fill);

    __m128i k1011 = _mm_unpacklo_epi8(k_10, k_11);
    __m128i k1213 = _mm_unpacklo_epi8(k_12, k_13);
    __m128i k140 = _mm_unpacklo_epi8(k_14, k_fill);

    __m128i k1516 = _mm_unpacklo_epi8(k_15, k_16);
    __m128i k1718 = _mm_unpacklo_epi8(k_17, k_18);
    __m128i k190 = _mm_unpacklo_epi8(k_19, k_fill);

    __m128i k2021 = _mm_unpacklo_epi8(k_20, k_21);
    __m128i k2223 = _mm_unpacklo_epi8(k_22, k_23);
    __m128i k240 = _mm_unpacklo_epi8(k_24, k_fill);

    __m256i bias_val;
    //! load bias
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }

    //! cvt i8 --> i16
    __m256i filter_01 = _mm256_cvtepi8_epi16(k01);
    __m256i filter_23 = _mm256_cvtepi8_epi16(k23);
    __m256i filter_40 = _mm256_cvtepi8_epi16(k40);

    __m256i filter_56 = _mm256_cvtepi8_epi16(k56);
    __m256i filter_78 = _mm256_cvtepi8_epi16(k78);
    __m256i filter_90 = _mm256_cvtepi8_epi16(k90);

    __m256i filter_1011 = _mm256_cvtepi8_epi16(k1011);
    __m256i filter_1213 = _mm256_cvtepi8_epi16(k1213);
    __m256i filter_140 = _mm256_cvtepi8_epi16(k140);

    __m256i filter_1516 = _mm256_cvtepi8_epi16(k1516);
    __m256i filter_1718 = _mm256_cvtepi8_epi16(k1718);
    __m256i filter_190 = _mm256_cvtepi8_epi16(k190);

    __m256i filter_2021 = _mm256_cvtepi8_epi16(k2021);
    __m256i filter_2223 = _mm256_cvtepi8_epi16(k2223);
    __m256i filter_240 = _mm256_cvtepi8_epi16(k240);

    size_t width = OW >> 4;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(6, load_src0)
            UNROLL_CALL0(6, load_src1)
            UNROLL_CALL0(6, load_src2)
            UNROLL_CALL0(6, load_src3)
            UNROLL_CALL0(6, load_src4)
            UNROLL_CALL0(6, load_src5)

            __m256i sum0_odd, sum0_even, sum1_odd, sum1_even, sum2_odd,
                    sum2_even, sum3_odd, sum3_even, sum4_odd, sum4_even;

            __m256i sum10_odd, sum10_even, sum20_odd, sum20_even, sum30_odd,
                    sum30_even, sum40_odd, sum40_even, sum5_odd, sum5_even;

            //! cal src0
            __m256i dot1, dot2, dot3;
            dot1 = _mm256_madd_epi16(cvt16_src00, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src02, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src04, filter_40);
            sum0_odd = _mm256_add_epi32(dot1, dot2);
            sum0_odd = _mm256_add_epi32(sum0_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src01, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src03, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src05, filter_40);
            sum0_even = _mm256_add_epi32(dot1, dot2);
            sum0_even = _mm256_add_epi32(sum0_even, dot3);

            //! cal src1
            dot1 = _mm256_madd_epi16(cvt16_src10, filter_56);
            dot2 = _mm256_madd_epi16(cvt16_src12, filter_78);
            dot3 = _mm256_madd_epi16(cvt16_src14, filter_90);
            sum1_odd = _mm256_add_epi32(dot1, dot2);
            sum1_odd = _mm256_add_epi32(sum1_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src11, filter_56);
            dot2 = _mm256_madd_epi16(cvt16_src13, filter_78);
            dot3 = _mm256_madd_epi16(cvt16_src15, filter_90);
            sum1_even = _mm256_add_epi32(dot1, dot2);
            sum1_even = _mm256_add_epi32(sum1_even, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src10, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src12, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src14, filter_40);
            sum10_odd = _mm256_add_epi32(dot1, dot2);
            sum10_odd = _mm256_add_epi32(sum10_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src11, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src13, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src15, filter_40);
            sum10_even = _mm256_add_epi32(dot1, dot2);
            sum10_even = _mm256_add_epi32(sum10_even, dot3);

            //! cal src2
            dot1 = _mm256_madd_epi16(cvt16_src20, filter_1011);
            dot2 = _mm256_madd_epi16(cvt16_src22, filter_1213);
            dot3 = _mm256_madd_epi16(cvt16_src24, filter_140);
            sum2_odd = _mm256_add_epi32(dot1, dot2);
            sum2_odd = _mm256_add_epi32(sum2_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src21, filter_1011);
            dot2 = _mm256_madd_epi16(cvt16_src23, filter_1213);
            dot3 = _mm256_madd_epi16(cvt16_src25, filter_140);
            sum2_even = _mm256_add_epi32(dot1, dot2);
            sum2_even = _mm256_add_epi32(sum2_even, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src20, filter_56);
            dot2 = _mm256_madd_epi16(cvt16_src22, filter_78);
            dot3 = _mm256_madd_epi16(cvt16_src24, filter_90);
            sum20_odd = _mm256_add_epi32(dot1, dot2);
            sum20_odd = _mm256_add_epi32(sum20_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src21, filter_56);
            dot2 = _mm256_madd_epi16(cvt16_src23, filter_78);
            dot3 = _mm256_madd_epi16(cvt16_src25, filter_90);
            sum20_even = _mm256_add_epi32(dot1, dot2);
            sum20_even = _mm256_add_epi32(sum20_even, dot3);

            //! cal src3
            dot1 = _mm256_madd_epi16(cvt16_src30, filter_1516);
            dot2 = _mm256_madd_epi16(cvt16_src32, filter_1718);
            dot3 = _mm256_madd_epi16(cvt16_src34, filter_190);
            sum3_odd = _mm256_add_epi32(dot1, dot2);
            sum3_odd = _mm256_add_epi32(sum3_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src31, filter_1516);
            dot2 = _mm256_madd_epi16(cvt16_src33, filter_1718);
            dot3 = _mm256_madd_epi16(cvt16_src35, filter_190);
            sum3_even = _mm256_add_epi32(dot1, dot2);
            sum3_even = _mm256_add_epi32(sum3_even, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src30, filter_1011);
            dot2 = _mm256_madd_epi16(cvt16_src32, filter_1213);
            dot3 = _mm256_madd_epi16(cvt16_src34, filter_140);
            sum30_odd = _mm256_add_epi32(dot1, dot2);
            sum30_odd = _mm256_add_epi32(sum30_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src31, filter_1011);
            dot2 = _mm256_madd_epi16(cvt16_src33, filter_1213);
            dot3 = _mm256_madd_epi16(cvt16_src35, filter_140);
            sum30_even = _mm256_add_epi32(dot1, dot2);
            sum30_even = _mm256_add_epi32(sum30_even, dot3);

            //! cal src4
            dot1 = _mm256_madd_epi16(cvt16_src40, filter_2021);
            dot2 = _mm256_madd_epi16(cvt16_src42, filter_2223);
            dot3 = _mm256_madd_epi16(cvt16_src44, filter_240);
            sum4_odd = _mm256_add_epi32(dot1, dot2);
            sum4_odd = _mm256_add_epi32(sum4_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src41, filter_2021);
            dot2 = _mm256_madd_epi16(cvt16_src43, filter_2223);
            dot3 = _mm256_madd_epi16(cvt16_src45, filter_240);
            sum4_even = _mm256_add_epi32(dot1, dot2);
            sum4_even = _mm256_add_epi32(sum4_even, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src40, filter_1516);
            dot2 = _mm256_madd_epi16(cvt16_src42, filter_1718);
            dot3 = _mm256_madd_epi16(cvt16_src44, filter_190);
            sum40_odd = _mm256_add_epi32(dot1, dot2);
            sum40_odd = _mm256_add_epi32(sum40_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src41, filter_1516);
            dot2 = _mm256_madd_epi16(cvt16_src43, filter_1718);
            dot3 = _mm256_madd_epi16(cvt16_src45, filter_190);
            sum40_even = _mm256_add_epi32(dot1, dot2);
            sum40_even = _mm256_add_epi32(sum40_even, dot3);

            //! cal src5
            dot1 = _mm256_madd_epi16(cvt16_src50, filter_2021);
            dot2 = _mm256_madd_epi16(cvt16_src52, filter_2223);
            dot3 = _mm256_madd_epi16(cvt16_src54, filter_240);
            sum5_odd = _mm256_add_epi32(dot1, dot2);
            sum5_odd = _mm256_add_epi32(sum5_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src51, filter_2021);
            dot2 = _mm256_madd_epi16(cvt16_src53, filter_2223);
            dot3 = _mm256_madd_epi16(cvt16_src55, filter_240);
            sum5_even = _mm256_add_epi32(dot1, dot2);
            sum5_even = _mm256_add_epi32(sum5_even, dot3);

            __m256i sum_odd, sum_even;

            sum_odd = _mm256_add_epi32(sum0_odd, sum1_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum2_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum3_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum4_odd);

            sum_even = _mm256_add_epi32(sum0_even, sum1_even);
            sum_even = _mm256_add_epi32(sum_even, sum2_even);
            sum_even = _mm256_add_epi32(sum_even, sum3_even);
            sum_even = _mm256_add_epi32(sum_even, sum4_even);

            __m256i sum_odd_0 = _mm256_unpacklo_epi32(sum_odd, sum_even);
            __m256i sum_even_0 = _mm256_unpackhi_epi32(sum_odd, sum_even);

            __m256i sum_left =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            __m256i sum_odd_oh1, sum_even_oh1;

            sum_odd_oh1 = _mm256_add_epi32(sum10_odd, sum20_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum30_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum40_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum5_odd);

            sum_even_oh1 = _mm256_add_epi32(sum10_even, sum20_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum30_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum40_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum5_even);

            __m256i sum_odd_1 =
                    _mm256_unpacklo_epi32(sum_odd_oh1, sum_even_oh1);
            __m256i sum_even_1 =
                    _mm256_unpackhi_epi32(sum_odd_oh1, sum_even_oh1);

            sum_left = _mm256_permute2f128_si256(sum_odd_1, sum_even_1, 32);
            sum_right = _mm256_permute2f128_si256(sum_odd_1, sum_even_1, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst1));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr1), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr1 + 8), sum_right);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            dst0 += 16;
            dst1 += 16;
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;
        r4 += tail_step + IW;
        r5 += tail_step + IW;

        dst0 += OW;
        dst1 += OW;
        out_ptr0 += OW;
        out_ptr1 += OW;
    }

    for (; h < OH; h++) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(5, load_src0)
            UNROLL_CALL0(5, load_src1)
            UNROLL_CALL0(5, load_src2)
            UNROLL_CALL0(5, load_src3)
            UNROLL_CALL0(5, load_src4)
            UNROLL_CALL0(5, load_src5)

            __m256i sum0_odd, sum0_even, sum1_odd, sum1_even, sum2_odd,
                    sum2_even, sum3_odd, sum3_even, sum4_odd, sum4_even;

            //! cal src0
            __m256i dot1, dot2, dot3;
            dot1 = _mm256_madd_epi16(cvt16_src00, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src02, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src04, filter_40);
            sum0_odd = _mm256_add_epi32(dot1, dot2);
            sum0_odd = _mm256_add_epi32(sum0_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src01, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src03, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src05, filter_40);
            sum0_even = _mm256_add_epi32(dot1, dot2);
            sum0_even = _mm256_add_epi32(sum0_even, dot3);

            //! cal src1
            dot1 = _mm256_madd_epi16(cvt16_src10, filter_56);
            dot2 = _mm256_madd_epi16(cvt16_src12, filter_78);
            dot3 = _mm256_madd_epi16(cvt16_src14, filter_90);
            sum1_odd = _mm256_add_epi32(dot1, dot2);
            sum1_odd = _mm256_add_epi32(sum1_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src11, filter_56);
            dot2 = _mm256_madd_epi16(cvt16_src13, filter_78);
            dot3 = _mm256_madd_epi16(cvt16_src15, filter_90);
            sum1_even = _mm256_add_epi32(dot1, dot2);
            sum1_even = _mm256_add_epi32(sum1_even, dot3);

            //! cal src2
            dot1 = _mm256_madd_epi16(cvt16_src20, filter_1011);
            dot2 = _mm256_madd_epi16(cvt16_src22, filter_1213);
            dot3 = _mm256_madd_epi16(cvt16_src24, filter_140);
            sum2_odd = _mm256_add_epi32(dot1, dot2);
            sum2_odd = _mm256_add_epi32(sum2_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src21, filter_1011);
            dot2 = _mm256_madd_epi16(cvt16_src23, filter_1213);
            dot3 = _mm256_madd_epi16(cvt16_src25, filter_140);
            sum2_even = _mm256_add_epi32(dot1, dot2);
            sum2_even = _mm256_add_epi32(sum2_even, dot3);

            //! cal src3
            dot1 = _mm256_madd_epi16(cvt16_src30, filter_1516);
            dot2 = _mm256_madd_epi16(cvt16_src32, filter_1718);
            dot3 = _mm256_madd_epi16(cvt16_src34, filter_190);
            sum3_odd = _mm256_add_epi32(dot1, dot2);
            sum3_odd = _mm256_add_epi32(sum3_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src31, filter_1516);
            dot2 = _mm256_madd_epi16(cvt16_src33, filter_1718);
            dot3 = _mm256_madd_epi16(cvt16_src35, filter_190);
            sum3_even = _mm256_add_epi32(dot1, dot2);
            sum3_even = _mm256_add_epi32(sum3_even, dot3);

            //! cal src4
            dot1 = _mm256_madd_epi16(cvt16_src40, filter_2021);
            dot2 = _mm256_madd_epi16(cvt16_src42, filter_2223);
            dot3 = _mm256_madd_epi16(cvt16_src44, filter_240);
            sum4_odd = _mm256_add_epi32(dot1, dot2);
            sum4_odd = _mm256_add_epi32(sum4_odd, dot3);

            dot1 = _mm256_madd_epi16(cvt16_src41, filter_2021);
            dot2 = _mm256_madd_epi16(cvt16_src43, filter_2223);
            dot3 = _mm256_madd_epi16(cvt16_src45, filter_240);
            sum4_even = _mm256_add_epi32(dot1, dot2);
            sum4_even = _mm256_add_epi32(sum4_even, dot3);

            __m256i sum_odd, sum_even;

            sum_odd = _mm256_add_epi32(sum0_odd, sum1_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum2_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum3_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum4_odd);

            sum_even = _mm256_add_epi32(sum0_even, sum1_even);
            sum_even = _mm256_add_epi32(sum_even, sum2_even);
            sum_even = _mm256_add_epi32(sum_even, sum3_even);
            sum_even = _mm256_add_epi32(sum_even, sum4_even);

            __m256i sum_odd_0 = _mm256_unpacklo_epi32(sum_odd, sum_even);
            __m256i sum_even_0 = _mm256_unpackhi_epi32(sum_odd, sum_even);

            __m256i sum_left =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
    }
}

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride1_7x7_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    size_t tail_step = IW - OW;
    int8_t* dst0 = dst;
    int8_t* dst1 = dst + OW;
    int32_t* out_ptr0 = temp;
    int32_t* out_ptr1 = temp + OW;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;
    const int8_t* r3 = src + 3 * IW;
    const int8_t* r4 = src + 4 * IW;
    const int8_t* r5 = src + 5 * IW;
    const int8_t* r6 = src + 6 * IW;
    const int8_t* r7 = src + 7 * IW;

    uint8_t fill_zero = 0;
    UNROLL_CALL0(49, load_filter)

    __m128i k_fill = _mm_set1_epi8(fill_zero);

    __m128i k01 = _mm_unpacklo_epi8(k_0, k_1);
    __m128i k23 = _mm_unpacklo_epi8(k_2, k_3);
    __m128i k45 = _mm_unpacklo_epi8(k_4, k_5);
    __m128i k60 = _mm_unpacklo_epi8(k_6, k_fill);

    __m128i k78 = _mm_unpacklo_epi8(k_7, k_8);
    __m128i k910 = _mm_unpacklo_epi8(k_9, k_10);
    __m128i k1112 = _mm_unpacklo_epi8(k_11, k_12);
    __m128i k130 = _mm_unpacklo_epi8(k_13, k_fill);

    __m128i k1415 = _mm_unpacklo_epi8(k_14, k_15);
    __m128i k1617 = _mm_unpacklo_epi8(k_16, k_17);
    __m128i k1819 = _mm_unpacklo_epi8(k_18, k_19);
    __m128i k200 = _mm_unpacklo_epi8(k_20, k_fill);

    __m128i k2122 = _mm_unpacklo_epi8(k_21, k_22);
    __m128i k2324 = _mm_unpacklo_epi8(k_23, k_24);
    __m128i k2526 = _mm_unpacklo_epi8(k_25, k_26);
    __m128i k270 = _mm_unpacklo_epi8(k_27, k_fill);

    __m128i k2829 = _mm_unpacklo_epi8(k_28, k_29);
    __m128i k3031 = _mm_unpacklo_epi8(k_30, k_31);
    __m128i k3233 = _mm_unpacklo_epi8(k_32, k_33);
    __m128i k340 = _mm_unpacklo_epi8(k_34, k_fill);

    __m128i k3536 = _mm_unpacklo_epi8(k_35, k_36);
    __m128i k3738 = _mm_unpacklo_epi8(k_37, k_38);
    __m128i k3940 = _mm_unpacklo_epi8(k_39, k_40);
    __m128i k410 = _mm_unpacklo_epi8(k_41, k_fill);

    __m128i k4243 = _mm_unpacklo_epi8(k_42, k_43);
    __m128i k4445 = _mm_unpacklo_epi8(k_44, k_45);
    __m128i k4647 = _mm_unpacklo_epi8(k_46, k_47);
    __m128i k480 = _mm_unpacklo_epi8(k_48, k_fill);

    __m256i bias_val;
    //! load bias
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }

    //! cvt i8 --> i16
    __m256i filter_01 = _mm256_cvtepi8_epi16(k01);
    __m256i filter_23 = _mm256_cvtepi8_epi16(k23);
    __m256i filter_45 = _mm256_cvtepi8_epi16(k45);
    __m256i filter_60 = _mm256_cvtepi8_epi16(k60);

    __m256i filter_78 = _mm256_cvtepi8_epi16(k78);
    __m256i filter_910 = _mm256_cvtepi8_epi16(k910);
    __m256i filter_1112 = _mm256_cvtepi8_epi16(k1112);
    __m256i filter_130 = _mm256_cvtepi8_epi16(k130);

    __m256i filter_1415 = _mm256_cvtepi8_epi16(k1415);
    __m256i filter_1617 = _mm256_cvtepi8_epi16(k1617);
    __m256i filter_1819 = _mm256_cvtepi8_epi16(k1819);
    __m256i filter_200 = _mm256_cvtepi8_epi16(k200);

    __m256i filter_2122 = _mm256_cvtepi8_epi16(k2122);
    __m256i filter_2324 = _mm256_cvtepi8_epi16(k2324);
    __m256i filter_2526 = _mm256_cvtepi8_epi16(k2526);
    __m256i filter_270 = _mm256_cvtepi8_epi16(k270);

    __m256i filter_2829 = _mm256_cvtepi8_epi16(k2829);
    __m256i filter_3031 = _mm256_cvtepi8_epi16(k3031);
    __m256i filter_3233 = _mm256_cvtepi8_epi16(k3233);
    __m256i filter_340 = _mm256_cvtepi8_epi16(k340);

    __m256i filter_3536 = _mm256_cvtepi8_epi16(k3536);
    __m256i filter_3738 = _mm256_cvtepi8_epi16(k3738);
    __m256i filter_3940 = _mm256_cvtepi8_epi16(k3940);
    __m256i filter_410 = _mm256_cvtepi8_epi16(k410);

    __m256i filter_4243 = _mm256_cvtepi8_epi16(k4243);
    __m256i filter_4445 = _mm256_cvtepi8_epi16(k4445);
    __m256i filter_4647 = _mm256_cvtepi8_epi16(k4647);
    __m256i filter_480 = _mm256_cvtepi8_epi16(k480);

    size_t width = OW >> 4;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(8, load_src0)
            UNROLL_CALL0(8, load_src1)
            UNROLL_CALL0(8, load_src2)
            UNROLL_CALL0(8, load_src3)
            UNROLL_CALL0(8, load_src4)
            UNROLL_CALL0(8, load_src5)
            UNROLL_CALL0(8, load_src6)
            UNROLL_CALL0(8, load_src7)

            __m256i sum0_odd, sum0_even, sum1_odd, sum1_even, sum2_odd,
                    sum2_even, sum3_odd, sum3_even, sum4_odd, sum4_even,
                    sum5_odd, sum5_even, sum6_odd, sum6_even;

            __m256i sum10_odd, sum10_even, sum20_odd, sum20_even, sum30_odd,
                    sum30_even, sum40_odd, sum40_even, sum50_odd, sum50_even,
                    sum60_odd, sum60_even, sum7_odd, sum7_even;

            //! cal src0
            __m256i dot1, dot2, dot3, dot4;
            dot1 = _mm256_madd_epi16(cvt16_src00, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src02, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src04, filter_45);
            dot4 = _mm256_madd_epi16(cvt16_src06, filter_60);
            sum0_odd = _mm256_add_epi32(dot1, dot2);
            sum0_odd = _mm256_add_epi32(sum0_odd, dot3);
            sum0_odd = _mm256_add_epi32(sum0_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src01, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src03, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src05, filter_45);
            dot4 = _mm256_madd_epi16(cvt16_src07, filter_60);
            sum0_even = _mm256_add_epi32(dot1, dot2);
            sum0_even = _mm256_add_epi32(sum0_even, dot3);
            sum0_even = _mm256_add_epi32(sum0_even, dot4);

            //! cal src1
            dot1 = _mm256_madd_epi16(cvt16_src10, filter_78);
            dot2 = _mm256_madd_epi16(cvt16_src12, filter_910);
            dot3 = _mm256_madd_epi16(cvt16_src14, filter_1112);
            dot4 = _mm256_madd_epi16(cvt16_src16, filter_130);
            sum1_odd = _mm256_add_epi32(dot1, dot2);
            sum1_odd = _mm256_add_epi32(sum1_odd, dot3);
            sum1_odd = _mm256_add_epi32(sum1_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src11, filter_78);
            dot2 = _mm256_madd_epi16(cvt16_src13, filter_910);
            dot3 = _mm256_madd_epi16(cvt16_src15, filter_1112);
            dot4 = _mm256_madd_epi16(cvt16_src17, filter_130);
            sum1_even = _mm256_add_epi32(dot1, dot2);
            sum1_even = _mm256_add_epi32(sum1_even, dot3);
            sum1_even = _mm256_add_epi32(sum1_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src10, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src12, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src14, filter_45);
            dot4 = _mm256_madd_epi16(cvt16_src16, filter_60);
            sum10_odd = _mm256_add_epi32(dot1, dot2);
            sum10_odd = _mm256_add_epi32(sum10_odd, dot3);
            sum10_odd = _mm256_add_epi32(sum10_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src11, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src13, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src15, filter_45);
            dot4 = _mm256_madd_epi16(cvt16_src17, filter_60);
            sum10_even = _mm256_add_epi32(dot1, dot2);
            sum10_even = _mm256_add_epi32(sum10_even, dot3);
            sum10_even = _mm256_add_epi32(sum10_even, dot4);

            //! cal src2
            dot1 = _mm256_madd_epi16(cvt16_src20, filter_1415);
            dot2 = _mm256_madd_epi16(cvt16_src22, filter_1617);
            dot3 = _mm256_madd_epi16(cvt16_src24, filter_1819);
            dot4 = _mm256_madd_epi16(cvt16_src26, filter_200);
            sum2_odd = _mm256_add_epi32(dot1, dot2);
            sum2_odd = _mm256_add_epi32(sum2_odd, dot3);
            sum2_odd = _mm256_add_epi32(sum2_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src21, filter_1415);
            dot2 = _mm256_madd_epi16(cvt16_src23, filter_1617);
            dot3 = _mm256_madd_epi16(cvt16_src25, filter_1819);
            dot4 = _mm256_madd_epi16(cvt16_src27, filter_200);
            sum2_even = _mm256_add_epi32(dot1, dot2);
            sum2_even = _mm256_add_epi32(sum2_even, dot3);
            sum2_even = _mm256_add_epi32(sum2_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src20, filter_78);
            dot2 = _mm256_madd_epi16(cvt16_src22, filter_910);
            dot3 = _mm256_madd_epi16(cvt16_src24, filter_1112);
            dot4 = _mm256_madd_epi16(cvt16_src26, filter_130);
            sum20_odd = _mm256_add_epi32(dot1, dot2);
            sum20_odd = _mm256_add_epi32(sum20_odd, dot3);
            sum20_odd = _mm256_add_epi32(sum20_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src21, filter_78);
            dot2 = _mm256_madd_epi16(cvt16_src23, filter_910);
            dot3 = _mm256_madd_epi16(cvt16_src25, filter_1112);
            dot4 = _mm256_madd_epi16(cvt16_src27, filter_130);
            sum20_even = _mm256_add_epi32(dot1, dot2);
            sum20_even = _mm256_add_epi32(sum20_even, dot3);
            sum20_even = _mm256_add_epi32(sum20_even, dot4);

            //! cal src3
            dot1 = _mm256_madd_epi16(cvt16_src30, filter_2122);
            dot2 = _mm256_madd_epi16(cvt16_src32, filter_2324);
            dot3 = _mm256_madd_epi16(cvt16_src34, filter_2526);
            dot4 = _mm256_madd_epi16(cvt16_src36, filter_270);
            sum3_odd = _mm256_add_epi32(dot1, dot2);
            sum3_odd = _mm256_add_epi32(sum3_odd, dot3);
            sum3_odd = _mm256_add_epi32(sum3_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src31, filter_2122);
            dot2 = _mm256_madd_epi16(cvt16_src33, filter_2324);
            dot3 = _mm256_madd_epi16(cvt16_src35, filter_2526);
            dot4 = _mm256_madd_epi16(cvt16_src37, filter_270);
            sum3_even = _mm256_add_epi32(dot1, dot2);
            sum3_even = _mm256_add_epi32(sum3_even, dot3);
            sum3_even = _mm256_add_epi32(sum3_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src30, filter_1415);
            dot2 = _mm256_madd_epi16(cvt16_src32, filter_1617);
            dot3 = _mm256_madd_epi16(cvt16_src34, filter_1819);
            dot4 = _mm256_madd_epi16(cvt16_src36, filter_200);
            sum30_odd = _mm256_add_epi32(dot1, dot2);
            sum30_odd = _mm256_add_epi32(sum30_odd, dot3);
            sum30_odd = _mm256_add_epi32(sum30_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src31, filter_1415);
            dot2 = _mm256_madd_epi16(cvt16_src33, filter_1617);
            dot3 = _mm256_madd_epi16(cvt16_src35, filter_1819);
            dot4 = _mm256_madd_epi16(cvt16_src37, filter_200);
            sum30_even = _mm256_add_epi32(dot1, dot2);
            sum30_even = _mm256_add_epi32(sum30_even, dot3);
            sum30_even = _mm256_add_epi32(sum30_even, dot4);

            //! cal src4
            dot1 = _mm256_madd_epi16(cvt16_src40, filter_2829);
            dot2 = _mm256_madd_epi16(cvt16_src42, filter_3031);
            dot3 = _mm256_madd_epi16(cvt16_src44, filter_3233);
            dot4 = _mm256_madd_epi16(cvt16_src46, filter_340);
            sum4_odd = _mm256_add_epi32(dot1, dot2);
            sum4_odd = _mm256_add_epi32(sum4_odd, dot3);
            sum4_odd = _mm256_add_epi32(sum4_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src41, filter_2829);
            dot2 = _mm256_madd_epi16(cvt16_src43, filter_3031);
            dot3 = _mm256_madd_epi16(cvt16_src45, filter_3233);
            dot4 = _mm256_madd_epi16(cvt16_src47, filter_340);
            sum4_even = _mm256_add_epi32(dot1, dot2);
            sum4_even = _mm256_add_epi32(sum4_even, dot3);
            sum4_even = _mm256_add_epi32(sum4_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src40, filter_2122);
            dot2 = _mm256_madd_epi16(cvt16_src42, filter_2324);
            dot3 = _mm256_madd_epi16(cvt16_src44, filter_2526);
            dot4 = _mm256_madd_epi16(cvt16_src46, filter_270);
            sum40_odd = _mm256_add_epi32(dot1, dot2);
            sum40_odd = _mm256_add_epi32(sum40_odd, dot3);
            sum40_odd = _mm256_add_epi32(sum40_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src41, filter_2122);
            dot2 = _mm256_madd_epi16(cvt16_src43, filter_2324);
            dot3 = _mm256_madd_epi16(cvt16_src45, filter_2526);
            dot4 = _mm256_madd_epi16(cvt16_src47, filter_270);
            sum40_even = _mm256_add_epi32(dot1, dot2);
            sum40_even = _mm256_add_epi32(sum40_even, dot3);
            sum40_even = _mm256_add_epi32(sum40_even, dot4);

            //! cal src5
            dot1 = _mm256_madd_epi16(cvt16_src50, filter_3536);
            dot2 = _mm256_madd_epi16(cvt16_src52, filter_3738);
            dot3 = _mm256_madd_epi16(cvt16_src54, filter_3940);
            dot4 = _mm256_madd_epi16(cvt16_src56, filter_410);
            sum5_odd = _mm256_add_epi32(dot1, dot2);
            sum5_odd = _mm256_add_epi32(sum5_odd, dot3);
            sum5_odd = _mm256_add_epi32(sum5_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src51, filter_3536);
            dot2 = _mm256_madd_epi16(cvt16_src53, filter_3738);
            dot3 = _mm256_madd_epi16(cvt16_src55, filter_3940);
            dot4 = _mm256_madd_epi16(cvt16_src57, filter_410);
            sum5_even = _mm256_add_epi32(dot1, dot2);
            sum5_even = _mm256_add_epi32(sum5_even, dot3);
            sum5_even = _mm256_add_epi32(sum5_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src50, filter_2829);
            dot2 = _mm256_madd_epi16(cvt16_src52, filter_3031);
            dot3 = _mm256_madd_epi16(cvt16_src54, filter_3233);
            dot4 = _mm256_madd_epi16(cvt16_src56, filter_340);
            sum50_odd = _mm256_add_epi32(dot1, dot2);
            sum50_odd = _mm256_add_epi32(sum50_odd, dot3);
            sum50_odd = _mm256_add_epi32(sum50_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src51, filter_2829);
            dot2 = _mm256_madd_epi16(cvt16_src53, filter_3031);
            dot3 = _mm256_madd_epi16(cvt16_src55, filter_3233);
            dot4 = _mm256_madd_epi16(cvt16_src57, filter_340);
            sum50_even = _mm256_add_epi32(dot1, dot2);
            sum50_even = _mm256_add_epi32(sum50_even, dot3);
            sum50_even = _mm256_add_epi32(sum50_even, dot4);

            //! cal src6
            dot1 = _mm256_madd_epi16(cvt16_src60, filter_4243);
            dot2 = _mm256_madd_epi16(cvt16_src62, filter_4445);
            dot3 = _mm256_madd_epi16(cvt16_src64, filter_4647);
            dot4 = _mm256_madd_epi16(cvt16_src66, filter_480);
            sum6_odd = _mm256_add_epi32(dot1, dot2);
            sum6_odd = _mm256_add_epi32(sum6_odd, dot3);
            sum6_odd = _mm256_add_epi32(sum6_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src61, filter_4243);
            dot2 = _mm256_madd_epi16(cvt16_src63, filter_4445);
            dot3 = _mm256_madd_epi16(cvt16_src65, filter_4647);
            dot4 = _mm256_madd_epi16(cvt16_src67, filter_480);
            sum6_even = _mm256_add_epi32(dot1, dot2);
            sum6_even = _mm256_add_epi32(sum6_even, dot3);
            sum6_even = _mm256_add_epi32(sum6_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src60, filter_3536);
            dot2 = _mm256_madd_epi16(cvt16_src62, filter_3738);
            dot3 = _mm256_madd_epi16(cvt16_src64, filter_3940);
            dot4 = _mm256_madd_epi16(cvt16_src66, filter_410);
            sum60_odd = _mm256_add_epi32(dot1, dot2);
            sum60_odd = _mm256_add_epi32(sum60_odd, dot3);
            sum60_odd = _mm256_add_epi32(sum60_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src61, filter_3536);
            dot2 = _mm256_madd_epi16(cvt16_src63, filter_3738);
            dot3 = _mm256_madd_epi16(cvt16_src65, filter_3940);
            dot4 = _mm256_madd_epi16(cvt16_src67, filter_410);
            sum60_even = _mm256_add_epi32(dot1, dot2);
            sum60_even = _mm256_add_epi32(sum60_even, dot3);
            sum60_even = _mm256_add_epi32(sum60_even, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src70, filter_4243);
            dot2 = _mm256_madd_epi16(cvt16_src72, filter_4445);
            dot3 = _mm256_madd_epi16(cvt16_src74, filter_4647);
            dot4 = _mm256_madd_epi16(cvt16_src76, filter_480);
            sum7_odd = _mm256_add_epi32(dot1, dot2);
            sum7_odd = _mm256_add_epi32(sum7_odd, dot3);
            sum7_odd = _mm256_add_epi32(sum7_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src71, filter_4243);
            dot2 = _mm256_madd_epi16(cvt16_src73, filter_4445);
            dot3 = _mm256_madd_epi16(cvt16_src75, filter_4647);
            dot4 = _mm256_madd_epi16(cvt16_src77, filter_480);
            sum7_even = _mm256_add_epi32(dot1, dot2);
            sum7_even = _mm256_add_epi32(sum7_even, dot3);
            sum7_even = _mm256_add_epi32(sum7_even, dot4);

            __m256i sum_odd, sum_even;

            //! add src0 ~ src6
            sum_odd = _mm256_add_epi32(sum0_odd, sum1_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum2_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum3_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum4_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum5_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum6_odd);

            sum_even = _mm256_add_epi32(sum0_even, sum1_even);
            sum_even = _mm256_add_epi32(sum_even, sum2_even);
            sum_even = _mm256_add_epi32(sum_even, sum3_even);
            sum_even = _mm256_add_epi32(sum_even, sum4_even);
            sum_even = _mm256_add_epi32(sum_even, sum5_even);
            sum_even = _mm256_add_epi32(sum_even, sum6_even);

            __m256i sum_odd_0 = _mm256_unpacklo_epi32(sum_odd, sum_even);
            __m256i sum_even_0 = _mm256_unpackhi_epi32(sum_odd, sum_even);

            __m256i sum_left =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            __m256i sum_odd_oh1, sum_even_oh1;

            //! add src1 ~ src7
            sum_odd_oh1 = _mm256_add_epi32(sum10_odd, sum20_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum30_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum40_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum50_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum60_odd);
            sum_odd_oh1 = _mm256_add_epi32(sum_odd_oh1, sum7_odd);

            sum_even_oh1 = _mm256_add_epi32(sum10_even, sum20_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum30_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum40_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum50_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum60_even);
            sum_even_oh1 = _mm256_add_epi32(sum_even_oh1, sum7_even);

            __m256i sum_odd_1 =
                    _mm256_unpacklo_epi32(sum_odd_oh1, sum_even_oh1);
            __m256i sum_even_1 =
                    _mm256_unpackhi_epi32(sum_odd_oh1, sum_even_oh1);

            sum_left = _mm256_permute2f128_si256(sum_odd_1, sum_even_1, 32);
            sum_right = _mm256_permute2f128_si256(sum_odd_1, sum_even_1, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst1));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr1), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr1 + 8), sum_right);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            r6 += 16;
            r7 += 16;
            dst0 += 16;
            dst1 += 16;
            out_ptr0 += 16;
            out_ptr1 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;
        r4 += tail_step + IW;
        r5 += tail_step + IW;
        r6 += tail_step + IW;
        r7 += tail_step + IW;

        dst0 += OW;
        dst1 += OW;
        out_ptr0 += OW;
        out_ptr1 += OW;
    }

    for (; h < OH; h++) {
        size_t w = 0;
        for (; w < width; w++) {
            UNROLL_CALL0(7, load_src0)
            UNROLL_CALL0(7, load_src1)
            UNROLL_CALL0(7, load_src2)
            UNROLL_CALL0(7, load_src3)
            UNROLL_CALL0(7, load_src4)
            UNROLL_CALL0(7, load_src5)
            UNROLL_CALL0(7, load_src6)
            UNROLL_CALL0(7, load_src7)
            __m256i sum0_odd, sum0_even, sum1_odd, sum1_even, sum2_odd,
                    sum2_even, sum3_odd, sum3_even, sum4_odd, sum4_even,
                    sum5_odd, sum5_even, sum6_odd, sum6_even;

            //! cal src0
            __m256i dot1, dot2, dot3, dot4;
            dot1 = _mm256_madd_epi16(cvt16_src00, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src02, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src04, filter_45);
            dot4 = _mm256_madd_epi16(cvt16_src06, filter_60);
            sum0_odd = _mm256_add_epi32(dot1, dot2);
            sum0_odd = _mm256_add_epi32(sum0_odd, dot3);
            sum0_odd = _mm256_add_epi32(sum0_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src01, filter_01);
            dot2 = _mm256_madd_epi16(cvt16_src03, filter_23);
            dot3 = _mm256_madd_epi16(cvt16_src05, filter_45);
            dot4 = _mm256_madd_epi16(cvt16_src07, filter_60);
            sum0_even = _mm256_add_epi32(dot1, dot2);
            sum0_even = _mm256_add_epi32(sum0_even, dot3);
            sum0_even = _mm256_add_epi32(sum0_even, dot4);

            //! cal src1
            dot1 = _mm256_madd_epi16(cvt16_src10, filter_78);
            dot2 = _mm256_madd_epi16(cvt16_src12, filter_910);
            dot3 = _mm256_madd_epi16(cvt16_src14, filter_1112);
            dot4 = _mm256_madd_epi16(cvt16_src16, filter_130);
            sum1_odd = _mm256_add_epi32(dot1, dot2);
            sum1_odd = _mm256_add_epi32(sum1_odd, dot3);
            sum1_odd = _mm256_add_epi32(sum1_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src11, filter_78);
            dot2 = _mm256_madd_epi16(cvt16_src13, filter_910);
            dot3 = _mm256_madd_epi16(cvt16_src15, filter_1112);
            dot4 = _mm256_madd_epi16(cvt16_src17, filter_130);
            sum1_even = _mm256_add_epi32(dot1, dot2);
            sum1_even = _mm256_add_epi32(sum1_even, dot3);
            sum1_even = _mm256_add_epi32(sum1_even, dot4);

            //! cal src2
            dot1 = _mm256_madd_epi16(cvt16_src20, filter_1415);
            dot2 = _mm256_madd_epi16(cvt16_src22, filter_1617);
            dot3 = _mm256_madd_epi16(cvt16_src24, filter_1819);
            dot4 = _mm256_madd_epi16(cvt16_src26, filter_200);
            sum2_odd = _mm256_add_epi32(dot1, dot2);
            sum2_odd = _mm256_add_epi32(sum2_odd, dot3);
            sum2_odd = _mm256_add_epi32(sum2_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src21, filter_1415);
            dot2 = _mm256_madd_epi16(cvt16_src23, filter_1617);
            dot3 = _mm256_madd_epi16(cvt16_src25, filter_1819);
            dot4 = _mm256_madd_epi16(cvt16_src27, filter_200);
            sum2_even = _mm256_add_epi32(dot1, dot2);
            sum2_even = _mm256_add_epi32(sum2_even, dot3);
            sum2_even = _mm256_add_epi32(sum2_even, dot4);

            //! cal src3
            dot1 = _mm256_madd_epi16(cvt16_src30, filter_2122);
            dot2 = _mm256_madd_epi16(cvt16_src32, filter_2324);
            dot3 = _mm256_madd_epi16(cvt16_src34, filter_2526);
            dot4 = _mm256_madd_epi16(cvt16_src36, filter_270);
            sum3_odd = _mm256_add_epi32(dot1, dot2);
            sum3_odd = _mm256_add_epi32(sum3_odd, dot3);
            sum3_odd = _mm256_add_epi32(sum3_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src31, filter_2122);
            dot2 = _mm256_madd_epi16(cvt16_src33, filter_2324);
            dot3 = _mm256_madd_epi16(cvt16_src35, filter_2526);
            dot4 = _mm256_madd_epi16(cvt16_src37, filter_270);
            sum3_even = _mm256_add_epi32(dot1, dot2);
            sum3_even = _mm256_add_epi32(sum3_even, dot3);
            sum3_even = _mm256_add_epi32(sum3_even, dot4);

            //! cal src4
            dot1 = _mm256_madd_epi16(cvt16_src40, filter_2829);
            dot2 = _mm256_madd_epi16(cvt16_src42, filter_3031);
            dot3 = _mm256_madd_epi16(cvt16_src44, filter_3233);
            dot4 = _mm256_madd_epi16(cvt16_src46, filter_340);
            sum4_odd = _mm256_add_epi32(dot1, dot2);
            sum4_odd = _mm256_add_epi32(sum4_odd, dot3);
            sum4_odd = _mm256_add_epi32(sum4_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src41, filter_2829);
            dot2 = _mm256_madd_epi16(cvt16_src43, filter_3031);
            dot3 = _mm256_madd_epi16(cvt16_src45, filter_3233);
            dot4 = _mm256_madd_epi16(cvt16_src47, filter_340);
            sum4_even = _mm256_add_epi32(dot1, dot2);
            sum4_even = _mm256_add_epi32(sum4_even, dot3);
            sum4_even = _mm256_add_epi32(sum4_even, dot4);

            //! cal src5
            dot1 = _mm256_madd_epi16(cvt16_src50, filter_3536);
            dot2 = _mm256_madd_epi16(cvt16_src52, filter_3738);
            dot3 = _mm256_madd_epi16(cvt16_src54, filter_3940);
            dot4 = _mm256_madd_epi16(cvt16_src56, filter_410);
            sum5_odd = _mm256_add_epi32(dot1, dot2);
            sum5_odd = _mm256_add_epi32(sum5_odd, dot3);
            sum5_odd = _mm256_add_epi32(sum5_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src51, filter_3536);
            dot2 = _mm256_madd_epi16(cvt16_src53, filter_3738);
            dot3 = _mm256_madd_epi16(cvt16_src55, filter_3940);
            dot4 = _mm256_madd_epi16(cvt16_src57, filter_410);
            sum5_even = _mm256_add_epi32(dot1, dot2);
            sum5_even = _mm256_add_epi32(sum5_even, dot3);
            sum5_even = _mm256_add_epi32(sum5_even, dot4);

            //! cal src6
            dot1 = _mm256_madd_epi16(cvt16_src60, filter_4243);
            dot2 = _mm256_madd_epi16(cvt16_src62, filter_4445);
            dot3 = _mm256_madd_epi16(cvt16_src64, filter_4647);
            dot4 = _mm256_madd_epi16(cvt16_src66, filter_480);
            sum6_odd = _mm256_add_epi32(dot1, dot2);
            sum6_odd = _mm256_add_epi32(sum6_odd, dot3);
            sum6_odd = _mm256_add_epi32(sum6_odd, dot4);

            dot1 = _mm256_madd_epi16(cvt16_src61, filter_4243);
            dot2 = _mm256_madd_epi16(cvt16_src63, filter_4445);
            dot3 = _mm256_madd_epi16(cvt16_src65, filter_4647);
            dot4 = _mm256_madd_epi16(cvt16_src67, filter_480);
            sum6_even = _mm256_add_epi32(dot1, dot2);
            sum6_even = _mm256_add_epi32(sum6_even, dot3);
            sum6_even = _mm256_add_epi32(sum6_even, dot4);

            __m256i sum_odd, sum_even;

            //! add src0 ~ src6
            sum_odd = _mm256_add_epi32(sum0_odd, sum1_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum2_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum3_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum4_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum5_odd);
            sum_odd = _mm256_add_epi32(sum_odd, sum6_odd);

            sum_even = _mm256_add_epi32(sum0_even, sum1_even);
            sum_even = _mm256_add_epi32(sum_even, sum2_even);
            sum_even = _mm256_add_epi32(sum_even, sum3_even);
            sum_even = _mm256_add_epi32(sum_even, sum4_even);
            sum_even = _mm256_add_epi32(sum_even, sum5_even);
            sum_even = _mm256_add_epi32(sum_even, sum6_even);

            __m256i sum_odd_0 = _mm256_unpacklo_epi32(sum_odd, sum_even);
            __m256i sum_even_0 = _mm256_unpackhi_epi32(sum_odd, sum_even);

            __m256i sum_left =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 32);
            __m256i sum_right =
                    _mm256_permute2f128_si256(sum_odd_0, sum_even_0, 49);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));
            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            r6 += 16;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
        r5 += tail_step;
        r6 += tail_step;
    }
}
#define INSTANTIATION(stride, i, bias, is_quantized, Op)                      \
    template void avx2_chanwise_direct_##stride##_##i##x##i##_int8<           \
            bias, is_quantized, Op>(const int8_t*, const int8_t*,             \
                                    const int32_t*, int32_t*, int8_t*,        \
                                    const size_t, const size_t, const size_t, \
                                    const size_t, const Op&);

#define FOR_OP(stride, i, is_quantized, bias)                                  \
    INSTANTIATION(stride, i, bias, is_quantized,                               \
                  TypeCvtOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 MEGDNN_COMMA \
                                    dt_qint8>)                                 \
    INSTANTIATION(stride, i, bias, is_quantized,                               \
                  ReluOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 MEGDNN_COMMA    \
                                 dt_qint8>)                                    \
    INSTANTIATION(stride, i, bias, is_quantized,                               \
                  HSwishOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 MEGDNN_COMMA  \
                                   dt_qint8>)

#define FOR_BIAS(stride, i, is_quantized)              \
    FOR_OP(stride, i, is_quantized, BiasMode::NO_BIAS) \
    FOR_OP(stride, i, is_quantized, BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_QUANTIZED(stride, i) \
    FOR_BIAS(stride, i, true)    \
    FOR_BIAS(stride, i, false)

#define FOR_FILTER(stride)   \
    FOR_QUANTIZED(stride, 2) \
    FOR_QUANTIZED(stride, 3) \
    FOR_QUANTIZED(stride, 5) \
    FOR_QUANTIZED(stride, 7)

#define FOR_STRIDE FOR_FILTER(stride1)

FOR_STRIDE

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_QUANTIZED
#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION
}  // namespace avx2_chanwise_stride1

namespace avx2_chanwise_stride2 {

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride2_2x2_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    size_t tail_step = IW - OW * 2;
    int8_t* dst0 = dst;
    int32_t* out_ptr0 = temp;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;

    UNROLL_CALL0(4, load_filter)

#define pack_filter(i, j) __m128i k_##i##j = _mm_unpacklo_epi8(k_##i, k_##j)
    pack_filter(0, 1);
    pack_filter(2, 3);

    __m256i bias_val;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }
#define cvt_filter(i, j) __m256i filter_##i##j = _mm256_cvtepi8_epi16(k_##i##j)
    cvt_filter(0, 1);
    cvt_filter(2, 3);

    size_t width = OW >> 4;
    for (size_t h = 0; h < OH; h++) {
        for (size_t w = 0; w < width; w++) {
            UNROLL_CALL0(2, load_src0)
            UNROLL_CALL0(2, load_src16)

            __m256i t0_left, t0_right, t1_left, t1_right, sum_left, sum_right;

            t0_left = _mm256_madd_epi16(cvt16_src00, filter_01);
            t0_right = _mm256_madd_epi16(cvt16_src016, filter_01);

            t1_left = _mm256_madd_epi16(cvt16_src10, filter_23);
            t1_right = _mm256_madd_epi16(cvt16_src116, filter_23);

            sum_left = _mm256_add_epi32(t0_left, t1_left);
            sum_right = _mm256_add_epi32(t0_right, t1_right);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));

            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 32;
            r1 += 32;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
    }

    MEGDNN_MARK_USED_VAR(IH);
#undef pack_filter
#undef cvt_filter
}

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride2_3x3_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    size_t tail_step = IW - OW * 2;
    int32_t* out_ptr0 = temp;
    int8_t* dst0 = dst;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;

    uint8_t fill_zero = 0;
    UNROLL_CALL0(9, load_filter)

    __m128i k_fill = _mm_set1_epi8(fill_zero);

    __m128i k01 = _mm_unpacklo_epi8(k_0, k_1);
    __m128i k20 = _mm_unpacklo_epi8(k_2, k_fill);

    __m128i k34 = _mm_unpacklo_epi8(k_3, k_4);
    __m128i k50 = _mm_unpacklo_epi8(k_5, k_fill);

    __m128i k67 = _mm_unpacklo_epi8(k_6, k_7);
    __m128i k80 = _mm_unpacklo_epi8(k_8, k_fill);

    __m256i bias_val;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }

    //! cvt i8 --> i16
    __m256i filter_01 = _mm256_cvtepi8_epi16(k01);
    __m256i filter_20 = _mm256_cvtepi8_epi16(k20);
    __m256i filter_34 = _mm256_cvtepi8_epi16(k34);
    __m256i filter_50 = _mm256_cvtepi8_epi16(k50);
    __m256i filter_67 = _mm256_cvtepi8_epi16(k67);
    __m256i filter_80 = _mm256_cvtepi8_epi16(k80);

    size_t width = OW >> 4;
    for (size_t h = 0; h < OH; h++) {
        for (size_t w = 0; w < width; w++) {
            UNROLL_CALL0(3, load_src0)
            UNROLL_CALL0(3, load_src2)
            UNROLL_CALL0(3, load_src16)
            UNROLL_CALL0(3, load_src18)

            __m256i temp, t0_left, t0_right, t1_left, t1_right, t2_left,
                    t2_right, sum_left, sum_right;

            t0_left = _mm256_madd_epi16(cvt16_src00, filter_01);
            temp = _mm256_madd_epi16(cvt16_src02, filter_20);
            t0_left = _mm256_add_epi32(t0_left, temp);

            t0_right = _mm256_madd_epi16(cvt16_src016, filter_01);
            temp = _mm256_madd_epi16(cvt16_src018, filter_20);
            t0_right = _mm256_add_epi32(t0_right, temp);

            t1_left = _mm256_madd_epi16(cvt16_src10, filter_34);
            temp = _mm256_madd_epi16(cvt16_src12, filter_50);
            t1_left = _mm256_add_epi32(t1_left, temp);

            t1_right = _mm256_madd_epi16(cvt16_src116, filter_34);
            temp = _mm256_madd_epi16(cvt16_src118, filter_50);
            t1_right = _mm256_add_epi32(t1_right, temp);

            t2_left = _mm256_madd_epi16(cvt16_src20, filter_67);
            temp = _mm256_madd_epi16(cvt16_src22, filter_80);
            t2_left = _mm256_add_epi32(t2_left, temp);

            t2_right = _mm256_madd_epi16(cvt16_src216, filter_67);
            temp = _mm256_madd_epi16(cvt16_src218, filter_80);
            t2_right = _mm256_add_epi32(t2_right, temp);

            sum_left = _mm256_add_epi32(t0_left, t1_left);
            sum_left = _mm256_add_epi32(sum_left, t2_left);
            sum_right = _mm256_add_epi32(t0_right, t1_right);
            sum_right = _mm256_add_epi32(sum_right, t2_right);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));

            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 32;
            r1 += 32;
            r2 += 32;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
    }
}

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride2_5x5_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    size_t tail_step = IW - OW * 2;
    int8_t* dst0 = dst;
    int32_t* out_ptr0 = temp;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;
    const int8_t* r3 = src + 3 * IW;
    const int8_t* r4 = src + 4 * IW;

    uint8_t fill_zero = 0;
    UNROLL_CALL0(25, load_filter)

    __m128i k_fill = _mm_set1_epi8(fill_zero);

    __m128i k01 = _mm_unpacklo_epi8(k_0, k_1);
    __m128i k23 = _mm_unpacklo_epi8(k_2, k_3);
    __m128i k40 = _mm_unpacklo_epi8(k_4, k_fill);

    __m128i k56 = _mm_unpacklo_epi8(k_5, k_6);
    __m128i k78 = _mm_unpacklo_epi8(k_7, k_8);
    __m128i k90 = _mm_unpacklo_epi8(k_9, k_fill);

    __m128i k1011 = _mm_unpacklo_epi8(k_10, k_11);
    __m128i k1213 = _mm_unpacklo_epi8(k_12, k_13);
    __m128i k140 = _mm_unpacklo_epi8(k_14, k_fill);

    __m128i k1516 = _mm_unpacklo_epi8(k_15, k_16);
    __m128i k1718 = _mm_unpacklo_epi8(k_17, k_18);
    __m128i k190 = _mm_unpacklo_epi8(k_19, k_fill);

    __m128i k2021 = _mm_unpacklo_epi8(k_20, k_21);
    __m128i k2223 = _mm_unpacklo_epi8(k_22, k_23);
    __m128i k240 = _mm_unpacklo_epi8(k_24, k_fill);

    __m256i bias_val;
    //! load bias
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }

    //! cvt i8 --> i16
    __m256i filter_01 = _mm256_cvtepi8_epi16(k01);
    __m256i filter_23 = _mm256_cvtepi8_epi16(k23);
    __m256i filter_40 = _mm256_cvtepi8_epi16(k40);

    __m256i filter_56 = _mm256_cvtepi8_epi16(k56);
    __m256i filter_78 = _mm256_cvtepi8_epi16(k78);
    __m256i filter_90 = _mm256_cvtepi8_epi16(k90);

    __m256i filter_1011 = _mm256_cvtepi8_epi16(k1011);
    __m256i filter_1213 = _mm256_cvtepi8_epi16(k1213);
    __m256i filter_140 = _mm256_cvtepi8_epi16(k140);

    __m256i filter_1516 = _mm256_cvtepi8_epi16(k1516);
    __m256i filter_1718 = _mm256_cvtepi8_epi16(k1718);
    __m256i filter_190 = _mm256_cvtepi8_epi16(k190);

    __m256i filter_2021 = _mm256_cvtepi8_epi16(k2021);
    __m256i filter_2223 = _mm256_cvtepi8_epi16(k2223);
    __m256i filter_240 = _mm256_cvtepi8_epi16(k240);

    size_t width = OW >> 4;
    for (size_t h = 0; h < OH; h++) {
        for (size_t w = 0; w < width; w++) {
            UNROLL_CALL0(5, load_src0)
            UNROLL_CALL0(5, load_src2)
            UNROLL_CALL0(5, load_src4)
            UNROLL_CALL0(5, load_src16)
            UNROLL_CALL0(5, load_src18)
            UNROLL_CALL0(5, load_src20)

            __m256i temp, t0_left, t0_right, t1_left, t1_right, t2_left,
                    t2_right, t3_left, t3_right, t4_left, t4_right, sum_left,
                    sum_right;

            t0_left = _mm256_madd_epi16(cvt16_src00, filter_01);
            temp = _mm256_madd_epi16(cvt16_src02, filter_23);
            t0_left = _mm256_add_epi32(t0_left, temp);
            temp = _mm256_madd_epi16(cvt16_src04, filter_40);
            t0_left = _mm256_add_epi32(t0_left, temp);

            t0_right = _mm256_madd_epi16(cvt16_src016, filter_01);
            temp = _mm256_madd_epi16(cvt16_src018, filter_23);
            t0_right = _mm256_add_epi32(t0_right, temp);
            temp = _mm256_madd_epi16(cvt16_src020, filter_40);
            t0_right = _mm256_add_epi32(t0_right, temp);

            t1_left = _mm256_madd_epi16(cvt16_src10, filter_56);
            temp = _mm256_madd_epi16(cvt16_src12, filter_78);
            t1_left = _mm256_add_epi32(t1_left, temp);
            temp = _mm256_madd_epi16(cvt16_src14, filter_90);
            t1_left = _mm256_add_epi32(t1_left, temp);

            t1_right = _mm256_madd_epi16(cvt16_src116, filter_56);
            temp = _mm256_madd_epi16(cvt16_src118, filter_78);
            t1_right = _mm256_add_epi32(t1_right, temp);
            temp = _mm256_madd_epi16(cvt16_src120, filter_90);
            t1_right = _mm256_add_epi32(t1_right, temp);

            t2_left = _mm256_madd_epi16(cvt16_src20, filter_1011);
            temp = _mm256_madd_epi16(cvt16_src22, filter_1213);
            t2_left = _mm256_add_epi32(t2_left, temp);
            temp = _mm256_madd_epi16(cvt16_src24, filter_140);
            t2_left = _mm256_add_epi32(t2_left, temp);

            t2_right = _mm256_madd_epi16(cvt16_src216, filter_1011);
            temp = _mm256_madd_epi16(cvt16_src218, filter_1213);
            t2_right = _mm256_add_epi32(t2_right, temp);
            temp = _mm256_madd_epi16(cvt16_src220, filter_140);
            t2_right = _mm256_add_epi32(t2_right, temp);

            t3_left = _mm256_madd_epi16(cvt16_src30, filter_1516);
            temp = _mm256_madd_epi16(cvt16_src32, filter_1718);
            t3_left = _mm256_add_epi32(t3_left, temp);
            temp = _mm256_madd_epi16(cvt16_src34, filter_190);
            t3_left = _mm256_add_epi32(t3_left, temp);

            t3_right = _mm256_madd_epi16(cvt16_src316, filter_1516);
            temp = _mm256_madd_epi16(cvt16_src318, filter_1718);
            t3_right = _mm256_add_epi32(t3_right, temp);
            temp = _mm256_madd_epi16(cvt16_src320, filter_190);
            t3_right = _mm256_add_epi32(t3_right, temp);

            t4_left = _mm256_madd_epi16(cvt16_src40, filter_2021);
            temp = _mm256_madd_epi16(cvt16_src42, filter_2223);
            t4_left = _mm256_add_epi32(t4_left, temp);
            temp = _mm256_madd_epi16(cvt16_src44, filter_240);
            t4_left = _mm256_add_epi32(t4_left, temp);

            t4_right = _mm256_madd_epi16(cvt16_src416, filter_2021);
            temp = _mm256_madd_epi16(cvt16_src418, filter_2223);
            t4_right = _mm256_add_epi32(t4_right, temp);
            temp = _mm256_madd_epi16(cvt16_src420, filter_240);
            t4_right = _mm256_add_epi32(t4_right, temp);

            sum_left = _mm256_add_epi32(t0_left, t1_left);
            sum_left = _mm256_add_epi32(sum_left, t2_left);
            sum_left = _mm256_add_epi32(sum_left, t3_left);
            sum_left = _mm256_add_epi32(sum_left, t4_left);
            sum_right = _mm256_add_epi32(t0_right, t1_right);
            sum_right = _mm256_add_epi32(sum_right, t2_right);
            sum_right = _mm256_add_epi32(sum_right, t3_right);
            sum_right = _mm256_add_epi32(sum_right, t4_right);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));

            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 32;
            r1 += 32;
            r2 += 32;
            r3 += 32;
            r4 += 32;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;
        r4 += tail_step + IW;
    }
}

template <BiasMode bias_mode, bool is_quantized, typename Op>
void avx2_chanwise_direct_stride2_7x7_int8(const int8_t* src,
                                           const int8_t* filter,
                                           const int32_t* bias, int32_t* temp,
                                           int8_t* dst, const size_t IH,
                                           const size_t IW, const size_t OH,
                                           const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    size_t tail_step = IW - OW * 2;
    int8_t* dst0 = dst;
    int32_t* out_ptr0 = temp;
    const int8_t* r0 = src;
    const int8_t* r1 = src + IW;
    const int8_t* r2 = src + 2 * IW;
    const int8_t* r3 = src + 3 * IW;
    const int8_t* r4 = src + 4 * IW;
    const int8_t* r5 = src + 5 * IW;
    const int8_t* r6 = src + 6 * IW;

    uint8_t fill_zero = 0;
    UNROLL_CALL0(49, load_filter)

    __m128i k_fill = _mm_set1_epi8(fill_zero);

    __m128i k01 = _mm_unpacklo_epi8(k_0, k_1);
    __m128i k23 = _mm_unpacklo_epi8(k_2, k_3);
    __m128i k45 = _mm_unpacklo_epi8(k_4, k_5);
    __m128i k60 = _mm_unpacklo_epi8(k_6, k_fill);

    __m128i k78 = _mm_unpacklo_epi8(k_7, k_8);
    __m128i k910 = _mm_unpacklo_epi8(k_9, k_10);
    __m128i k1112 = _mm_unpacklo_epi8(k_11, k_12);
    __m128i k130 = _mm_unpacklo_epi8(k_13, k_fill);

    __m128i k1415 = _mm_unpacklo_epi8(k_14, k_15);
    __m128i k1617 = _mm_unpacklo_epi8(k_16, k_17);
    __m128i k1819 = _mm_unpacklo_epi8(k_18, k_19);
    __m128i k200 = _mm_unpacklo_epi8(k_20, k_fill);

    __m128i k2122 = _mm_unpacklo_epi8(k_21, k_22);
    __m128i k2324 = _mm_unpacklo_epi8(k_23, k_24);
    __m128i k2526 = _mm_unpacklo_epi8(k_25, k_26);
    __m128i k270 = _mm_unpacklo_epi8(k_27, k_fill);

    __m128i k2829 = _mm_unpacklo_epi8(k_28, k_29);
    __m128i k3031 = _mm_unpacklo_epi8(k_30, k_31);
    __m128i k3233 = _mm_unpacklo_epi8(k_32, k_33);
    __m128i k340 = _mm_unpacklo_epi8(k_34, k_fill);

    __m128i k3536 = _mm_unpacklo_epi8(k_35, k_36);
    __m128i k3738 = _mm_unpacklo_epi8(k_37, k_38);
    __m128i k3940 = _mm_unpacklo_epi8(k_39, k_40);
    __m128i k410 = _mm_unpacklo_epi8(k_41, k_fill);

    __m128i k4243 = _mm_unpacklo_epi8(k_42, k_43);
    __m128i k4445 = _mm_unpacklo_epi8(k_44, k_45);
    __m128i k4647 = _mm_unpacklo_epi8(k_46, k_47);
    __m128i k480 = _mm_unpacklo_epi8(k_48, k_fill);

    __m256i bias_val;
    //! load bias
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_val = _mm256_set1_epi32(*(bias));
    } else {
        bias_val = _mm256_set1_epi32(0);
    }

    //! cvt i8 --> i16
    __m256i filter_01 = _mm256_cvtepi8_epi16(k01);
    __m256i filter_23 = _mm256_cvtepi8_epi16(k23);
    __m256i filter_45 = _mm256_cvtepi8_epi16(k45);
    __m256i filter_60 = _mm256_cvtepi8_epi16(k60);

    __m256i filter_78 = _mm256_cvtepi8_epi16(k78);
    __m256i filter_910 = _mm256_cvtepi8_epi16(k910);
    __m256i filter_1112 = _mm256_cvtepi8_epi16(k1112);
    __m256i filter_130 = _mm256_cvtepi8_epi16(k130);

    __m256i filter_1415 = _mm256_cvtepi8_epi16(k1415);
    __m256i filter_1617 = _mm256_cvtepi8_epi16(k1617);
    __m256i filter_1819 = _mm256_cvtepi8_epi16(k1819);
    __m256i filter_200 = _mm256_cvtepi8_epi16(k200);

    __m256i filter_2122 = _mm256_cvtepi8_epi16(k2122);
    __m256i filter_2324 = _mm256_cvtepi8_epi16(k2324);
    __m256i filter_2526 = _mm256_cvtepi8_epi16(k2526);
    __m256i filter_270 = _mm256_cvtepi8_epi16(k270);

    __m256i filter_2829 = _mm256_cvtepi8_epi16(k2829);
    __m256i filter_3031 = _mm256_cvtepi8_epi16(k3031);
    __m256i filter_3233 = _mm256_cvtepi8_epi16(k3233);
    __m256i filter_340 = _mm256_cvtepi8_epi16(k340);

    __m256i filter_3536 = _mm256_cvtepi8_epi16(k3536);
    __m256i filter_3738 = _mm256_cvtepi8_epi16(k3738);
    __m256i filter_3940 = _mm256_cvtepi8_epi16(k3940);
    __m256i filter_410 = _mm256_cvtepi8_epi16(k410);

    __m256i filter_4243 = _mm256_cvtepi8_epi16(k4243);
    __m256i filter_4445 = _mm256_cvtepi8_epi16(k4445);
    __m256i filter_4647 = _mm256_cvtepi8_epi16(k4647);
    __m256i filter_480 = _mm256_cvtepi8_epi16(k480);

    size_t width = OW >> 4;
    for (size_t h = 0; h < OH; h++) {
        for (size_t w = 0; w < width; w++) {
            UNROLL_CALL0(7, load_src0)
            UNROLL_CALL0(7, load_src2)
            UNROLL_CALL0(7, load_src4)
            UNROLL_CALL0(7, load_src6)
            UNROLL_CALL0(7, load_src16)
            UNROLL_CALL0(7, load_src18)
            UNROLL_CALL0(7, load_src20)
            UNROLL_CALL0(7, load_src22)

            __m256i temp, t0_left, t0_right, t1_left, t1_right, t2_left,
                    t2_right, t3_left, t3_right, t4_left, t4_right, sum_left,
                    t5_left, t5_right, t6_left, t6_right, sum_right;

            t0_left = _mm256_madd_epi16(cvt16_src00, filter_01);
            temp = _mm256_madd_epi16(cvt16_src02, filter_23);
            t0_left = _mm256_add_epi32(t0_left, temp);
            temp = _mm256_madd_epi16(cvt16_src04, filter_45);
            t0_left = _mm256_add_epi32(t0_left, temp);
            temp = _mm256_madd_epi16(cvt16_src06, filter_60);
            t0_left = _mm256_add_epi32(t0_left, temp);

            t0_right = _mm256_madd_epi16(cvt16_src016, filter_01);
            temp = _mm256_madd_epi16(cvt16_src018, filter_23);
            t0_right = _mm256_add_epi32(t0_right, temp);
            temp = _mm256_madd_epi16(cvt16_src020, filter_45);
            t0_right = _mm256_add_epi32(t0_right, temp);
            temp = _mm256_madd_epi16(cvt16_src022, filter_60);
            t0_right = _mm256_add_epi32(t0_right, temp);

            t1_left = _mm256_madd_epi16(cvt16_src10, filter_78);
            temp = _mm256_madd_epi16(cvt16_src12, filter_910);
            t1_left = _mm256_add_epi32(t1_left, temp);
            temp = _mm256_madd_epi16(cvt16_src14, filter_1112);
            t1_left = _mm256_add_epi32(t1_left, temp);
            temp = _mm256_madd_epi16(cvt16_src16, filter_130);
            t1_left = _mm256_add_epi32(t1_left, temp);

            t1_right = _mm256_madd_epi16(cvt16_src116, filter_78);
            temp = _mm256_madd_epi16(cvt16_src118, filter_910);
            t1_right = _mm256_add_epi32(t1_right, temp);
            temp = _mm256_madd_epi16(cvt16_src120, filter_1112);
            t1_right = _mm256_add_epi32(t1_right, temp);
            temp = _mm256_madd_epi16(cvt16_src122, filter_130);
            t1_right = _mm256_add_epi32(t1_right, temp);

            t2_left = _mm256_madd_epi16(cvt16_src20, filter_1415);
            temp = _mm256_madd_epi16(cvt16_src22, filter_1617);
            t2_left = _mm256_add_epi32(t2_left, temp);
            temp = _mm256_madd_epi16(cvt16_src24, filter_1819);
            t2_left = _mm256_add_epi32(t2_left, temp);
            temp = _mm256_madd_epi16(cvt16_src26, filter_200);
            t2_left = _mm256_add_epi32(t2_left, temp);

            t2_right = _mm256_madd_epi16(cvt16_src216, filter_1415);
            temp = _mm256_madd_epi16(cvt16_src218, filter_1617);
            t2_right = _mm256_add_epi32(t2_right, temp);
            temp = _mm256_madd_epi16(cvt16_src220, filter_1819);
            t2_right = _mm256_add_epi32(t2_right, temp);
            temp = _mm256_madd_epi16(cvt16_src222, filter_200);
            t2_right = _mm256_add_epi32(t2_right, temp);

            t3_left = _mm256_madd_epi16(cvt16_src30, filter_2122);
            temp = _mm256_madd_epi16(cvt16_src32, filter_2324);
            t3_left = _mm256_add_epi32(t3_left, temp);
            temp = _mm256_madd_epi16(cvt16_src34, filter_2526);
            t3_left = _mm256_add_epi32(t3_left, temp);
            temp = _mm256_madd_epi16(cvt16_src36, filter_270);
            t3_left = _mm256_add_epi32(t3_left, temp);

            t3_right = _mm256_madd_epi16(cvt16_src316, filter_2122);
            temp = _mm256_madd_epi16(cvt16_src318, filter_2324);
            t3_right = _mm256_add_epi32(t3_right, temp);
            temp = _mm256_madd_epi16(cvt16_src320, filter_2526);
            t3_right = _mm256_add_epi32(t3_right, temp);
            temp = _mm256_madd_epi16(cvt16_src322, filter_270);
            t3_right = _mm256_add_epi32(t3_right, temp);

            t4_left = _mm256_madd_epi16(cvt16_src40, filter_2829);
            temp = _mm256_madd_epi16(cvt16_src42, filter_3031);
            t4_left = _mm256_add_epi32(t4_left, temp);
            temp = _mm256_madd_epi16(cvt16_src44, filter_3233);
            t4_left = _mm256_add_epi32(t4_left, temp);
            temp = _mm256_madd_epi16(cvt16_src46, filter_340);
            t4_left = _mm256_add_epi32(t4_left, temp);

            t4_right = _mm256_madd_epi16(cvt16_src416, filter_2829);
            temp = _mm256_madd_epi16(cvt16_src418, filter_3031);
            t4_right = _mm256_add_epi32(t4_right, temp);
            temp = _mm256_madd_epi16(cvt16_src420, filter_3233);
            t4_right = _mm256_add_epi32(t4_right, temp);
            temp = _mm256_madd_epi16(cvt16_src422, filter_340);
            t4_right = _mm256_add_epi32(t4_right, temp);

            t5_left = _mm256_madd_epi16(cvt16_src50, filter_3536);
            temp = _mm256_madd_epi16(cvt16_src52, filter_3738);
            t5_left = _mm256_add_epi32(t5_left, temp);
            temp = _mm256_madd_epi16(cvt16_src54, filter_3940);
            t5_left = _mm256_add_epi32(t5_left, temp);
            temp = _mm256_madd_epi16(cvt16_src56, filter_410);
            t5_left = _mm256_add_epi32(t5_left, temp);

            t5_right = _mm256_madd_epi16(cvt16_src516, filter_3536);
            temp = _mm256_madd_epi16(cvt16_src518, filter_3738);
            t5_right = _mm256_add_epi32(t5_right, temp);
            temp = _mm256_madd_epi16(cvt16_src520, filter_3940);
            t5_right = _mm256_add_epi32(t5_right, temp);
            temp = _mm256_madd_epi16(cvt16_src522, filter_410);
            t5_right = _mm256_add_epi32(t5_right, temp);

            t6_left = _mm256_madd_epi16(cvt16_src60, filter_4243);
            temp = _mm256_madd_epi16(cvt16_src62, filter_4445);
            t6_left = _mm256_add_epi32(t6_left, temp);
            temp = _mm256_madd_epi16(cvt16_src64, filter_4647);
            t6_left = _mm256_add_epi32(t6_left, temp);
            temp = _mm256_madd_epi16(cvt16_src66, filter_480);
            t6_left = _mm256_add_epi32(t6_left, temp);

            t6_right = _mm256_madd_epi16(cvt16_src616, filter_4243);
            temp = _mm256_madd_epi16(cvt16_src618, filter_4445);
            t6_right = _mm256_add_epi32(t6_right, temp);
            temp = _mm256_madd_epi16(cvt16_src620, filter_4647);
            t6_right = _mm256_add_epi32(t6_right, temp);
            temp = _mm256_madd_epi16(cvt16_src622, filter_480);
            t6_right = _mm256_add_epi32(t6_right, temp);

            sum_left = _mm256_add_epi32(t0_left, t1_left);
            sum_left = _mm256_add_epi32(sum_left, t2_left);
            sum_left = _mm256_add_epi32(sum_left, t3_left);
            sum_left = _mm256_add_epi32(sum_left, t4_left);
            sum_left = _mm256_add_epi32(sum_left, t5_left);
            sum_left = _mm256_add_epi32(sum_left, t6_left);
            sum_right = _mm256_add_epi32(t0_right, t1_right);
            sum_right = _mm256_add_epi32(sum_right, t2_right);
            sum_right = _mm256_add_epi32(sum_right, t3_right);
            sum_right = _mm256_add_epi32(sum_right, t4_right);
            sum_right = _mm256_add_epi32(sum_right, t5_right);
            sum_right = _mm256_add_epi32(sum_right, t6_right);

            sum_left = _mm256_add_epi32(sum_left, bias_val);
            sum_right = _mm256_add_epi32(sum_right, bias_val);

            if (is_quantized) {
                op({{sum_left, sum_right}}, reinterpret_cast<dt_qint8*>(dst0));

            } else {
                _mm256_storeu_si256((__m256i*)(out_ptr0), sum_left);
                _mm256_storeu_si256((__m256i*)(out_ptr0 + 8), sum_right);
            }

            r0 += 32;
            r1 += 32;
            r2 += 32;
            r3 += 32;
            r4 += 32;
            r5 += 32;
            r6 += 32;
            dst0 += 16;
            out_ptr0 += 16;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;
        r4 += tail_step + IW;
        r5 += tail_step + IW;
        r6 += tail_step + IW;
    }
}
#define INSTANTIATION(stride, i, bias, is_quantized, Op)                      \
    template void avx2_chanwise_direct_##stride##_##i##x##i##_int8<           \
            bias, is_quantized, Op>(const int8_t*, const int8_t*,             \
                                    const int32_t*, int32_t*, int8_t*,        \
                                    const size_t, const size_t, const size_t, \
                                    const size_t, const Op&);

#define FOR_OP(stride, i, is_quantized, bias)                                  \
    INSTANTIATION(stride, i, bias, is_quantized,                               \
                  TypeCvtOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 MEGDNN_COMMA \
                                    dt_qint8>)                                 \
    INSTANTIATION(stride, i, bias, is_quantized,                               \
                  ReluOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 MEGDNN_COMMA    \
                                 dt_qint8>)                                    \
    INSTANTIATION(stride, i, bias, is_quantized,                               \
                  HSwishOp<SIMDType::AVX2 MEGDNN_COMMA dt_qint32 MEGDNN_COMMA  \
                                   dt_qint8>)

#define FOR_BIAS(stride, i, is_quantized)              \
    FOR_OP(stride, i, is_quantized, BiasMode::NO_BIAS) \
    FOR_OP(stride, i, is_quantized, BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_QUANTIZED(stride, i) \
    FOR_BIAS(stride, i, true)    \
    FOR_BIAS(stride, i, false)

#define FOR_FILTER(stride)   \
    FOR_QUANTIZED(stride, 2) \
    FOR_QUANTIZED(stride, 3) \
    FOR_QUANTIZED(stride, 5) \
    FOR_QUANTIZED(stride, 7)

#define FOR_STRIDE FOR_FILTER(stride2)

FOR_STRIDE

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_QUANTIZED
#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION
}  // namespace avx2_chanwise_stride2
#undef load_filter
#undef load_src0
#undef load_src1
#undef load_src2
#undef load_src3
#undef load_src4
#undef load_src5
#undef load_src6
#undef load_src16
#undef load_src18
#undef load_src20
#undef load_src22
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
