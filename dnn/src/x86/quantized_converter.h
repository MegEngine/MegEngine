/**
 * \file dnn/src/x86/quantized_converter.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <immintrin.h>
#ifdef WIN32
#include <avx2intrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <smmintrin.h>
#endif
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"
#include "src/x86/simd_macro/immintrin.h"

namespace megdnn {
namespace x86 {

struct QConverterBase {
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    inline static __m128 vfzero() { return _mm_set1_ps(0.f); }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    inline static __m128 vfmin_int8() { return _mm_set1_ps(-128.f); }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    inline static __m128 vfmax_int8() { return _mm_set1_ps(127.f); }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    inline static __m128 vfhalf() { return _mm_set1_ps(0.5f); }
    MEGDNN_ATTRIBUTE_TARGET("sse4.2")
    inline static __m128 vfneg_half() { return _mm_set1_ps(-0.5f); }
};
struct QConverterBaseAvx {
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    inline static __m256 vfzero() { return _mm256_set1_ps(0.f); }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    inline static __m256 vfmin_int8() { return _mm256_set1_ps(-128.f); }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    inline static __m256 vfmax_int8() { return _mm256_set1_ps(127.f); }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    inline static __m256 vfhalf() { return _mm256_set1_ps(0.5f); }
    MEGDNN_ATTRIBUTE_TARGET("avx2")
    inline static __m256 vfneg_half() { return _mm256_set1_ps(-0.5f); }
};
struct QConverter {
    template <typename dst_type, typename... src_type>
    static inline dst_type convert(const src_type&... src);
};

template <>
inline dt_qint8 QConverter::convert(const float& src) {
    return dt_qint8(saturate<int8_t, float>(std::round(src), -128, 127));
}

template <>
inline dt_quint8 QConverter::convert(const float& src, const uint8_t& zp) {
    return dt_quint8(saturate<uint8_t, float>(std::round(src) + zp, 0, 255));
}

template <>
inline dt_qint32 QConverter::convert(const float& src) {
    return dt_qint32(saturate<int32_t, float>(
            std::round(src),
            static_cast<float>(std::numeric_limits<int32_t>::min()),
            static_cast<float>(std::numeric_limits<int32_t>::max())));
}

template <>
MEGDNN_ATTRIBUTE_TARGET("sse4.2")
inline int64_t QConverter::convert(const __m128x2& vsrc) {
    __m128 vinc0 = _mm_blendv_ps(
            QConverterBase::vfneg_half(), QConverterBase::vfhalf(),
            _mm_cmpge_ps(vsrc.val[0], QConverterBase::vfzero()));
    __m128 vinc1 = _mm_blendv_ps(
            QConverterBase::vfneg_half(), QConverterBase::vfhalf(),
            _mm_cmpge_ps(vsrc.val[1], QConverterBase::vfzero()));

    __m128 vres0 = _mm_add_ps(vsrc.val[0], vinc0);
    __m128 vres1 = _mm_add_ps(vsrc.val[1], vinc1);

    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm_min_ps(_mm_max_ps(vres0, QConverterBase::vfmin_int8()),
                       QConverterBase::vfmax_int8());
    vres1 = _mm_min_ps(_mm_max_ps(vres1, QConverterBase::vfmin_int8()),
                       QConverterBase::vfmax_int8());

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi32_1 = _mm_cvtps_epi32(vres1);
    __m128i vepi16 = _mm_packs_epi32(vepi32_0, vepi32_1);
    __m128i vzero = _mm_setzero_si128();
    __m128i vepi8 = _mm_packs_epi16(vepi16, vzero);
#ifdef __x86_64__
    return _mm_extract_epi64(vepi8, 0);
#else
    int64_t result = 0;
    _mm_storel_epi64((__m128i*)&result, vepi8);
    return result;
#endif
}

template <>
MEGDNN_ATTRIBUTE_TARGET("sse4.2")
inline int64_t QConverter::convert(const __m128x2& vsrc, const __m128i& zpo) {
    __m128 vinc0 = _mm_blendv_ps(
            QConverterBase::vfneg_half(), QConverterBase::vfhalf(),
            _mm_cmpge_ps(vsrc.val[0], QConverterBase::vfzero()));
    __m128 vinc1 = _mm_blendv_ps(
            QConverterBase::vfneg_half(), QConverterBase::vfhalf(),
            _mm_cmpge_ps(vsrc.val[1], QConverterBase::vfzero()));
    __m128 vres0 = _mm_add_ps(vsrc.val[0], vinc0);
    __m128 vres1 = _mm_add_ps(vsrc.val[1], vinc1);
    __m128 voffset =
            _mm_add_ps(_mm_cvtepi32_ps(zpo), QConverterBase::vfmin_int8());
    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm_add_ps(vres0, voffset);
    vres1 = _mm_add_ps(vres1, voffset);
    vres0 = _mm_min_ps(_mm_max_ps(vres0, QConverterBase::vfmin_int8()),
                       QConverterBase::vfmax_int8());
    vres1 = _mm_min_ps(_mm_max_ps(vres1, QConverterBase::vfmin_int8()),
                       QConverterBase::vfmax_int8());
    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi32_1 = _mm_cvtps_epi32(vres1);
    __m128i vepi16 = _mm_packs_epi32(vepi32_0, vepi32_1);
    __m128i vzero = _mm_setzero_si128();
    __m128i vepi8 = _mm_packs_epi16(vepi16, vzero);
    vepi8 = _mm_sub_epi8(vepi8, _mm_set1_epi8(-128));
#ifdef __x86_64__
    return _mm_extract_epi64(vepi8, 0);
#else
    int64_t result = 0;
    _mm_storel_epi64((__m128i*)&result, vepi8);
    return result;
#endif
}

template <>
MEGDNN_ATTRIBUTE_TARGET("sse4.2")
inline __m128i QConverter::convert(const __m128& vsrc) {
    __m128 vinc = _mm_blendv_ps(QConverterBase::vfneg_half(),
                                QConverterBase::vfhalf(),
                                _mm_cmpge_ps(vsrc, QConverterBase::vfzero()));
    return _mm_cvttps_epi32(_mm_add_ps(vsrc, vinc));
}
////////////////////////////////////////avx//////////////////////////////

template <>
MEGDNN_ATTRIBUTE_TARGET("avx2")
inline __m128i QConverter::convert(const __m256x2& vsrc) {
    __m256 vinc0 = _mm256_blendv_ps(
            QConverterBaseAvx::vfneg_half(), QConverterBaseAvx::vfhalf(),
            _mm256_cmp_ps(vsrc.val[0], QConverterBaseAvx::vfzero(),
                          _CMP_GE_OQ));
    __m256 vinc1 = _mm256_blendv_ps(
            QConverterBaseAvx::vfneg_half(), QConverterBaseAvx::vfhalf(),
            _mm256_cmp_ps(vsrc.val[1], QConverterBaseAvx::vfzero(),
                          _CMP_GE_OQ));

    __m256 vres0 = _mm256_add_ps(vsrc.val[0], vinc0);
    __m256 vres1 = _mm256_add_ps(vsrc.val[1], vinc1);

    vres0 = _mm256_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm256_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm256_min_ps(_mm256_max_ps(vres0, QConverterBaseAvx::vfmin_int8()),
                          QConverterBaseAvx::vfmax_int8());
    vres1 = _mm256_min_ps(_mm256_max_ps(vres1, QConverterBaseAvx::vfmin_int8()),
                          QConverterBaseAvx::vfmax_int8());
    __m256i vepi32_0 = _mm256_cvtps_epi32(vres0);
    __m256i vepi32_1 = _mm256_cvtps_epi32(vres1);
    __m128i vepi16_lo = _mm_packs_epi32(_mm256_extractf128_si256(vepi32_0, 0),
                                        _mm256_extractf128_si256(vepi32_0, 1));
    __m128i vepi16_ho = _mm_packs_epi32(_mm256_extractf128_si256(vepi32_1, 0),
                                        _mm256_extractf128_si256(vepi32_1, 1));
    return _mm_packs_epi16(vepi16_lo, vepi16_ho);
}

template <>
MEGDNN_ATTRIBUTE_TARGET("avx2")
inline __m128i QConverter::convert(const __m256x2& vsrc, const __m256i& zpo) {
    __m256 vinc0 = _mm256_blendv_ps(
            QConverterBaseAvx::vfneg_half(), QConverterBaseAvx::vfhalf(),
            _mm256_cmp_ps(vsrc.val[0], QConverterBaseAvx::vfzero(),
                          _CMP_GE_OQ));
    __m256 vinc1 = _mm256_blendv_ps(
            QConverterBaseAvx::vfneg_half(), QConverterBaseAvx::vfhalf(),
            _mm256_cmp_ps(vsrc.val[1], QConverterBaseAvx::vfzero(),
                          _CMP_GE_OQ));
    __m256 vres0 = _mm256_add_ps(vsrc.val[0], vinc0);
    __m256 vres1 = _mm256_add_ps(vsrc.val[1], vinc1);
    __m256 voffset = _mm256_add_ps(_mm256_cvtepi32_ps(zpo),
                                   QConverterBaseAvx::vfmin_int8());
    vres0 = _mm256_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm256_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm256_add_ps(vres0, voffset);
    vres1 = _mm256_add_ps(vres1, voffset);
    vres0 = _mm256_min_ps(_mm256_max_ps(vres0, QConverterBaseAvx::vfmin_int8()),
                          QConverterBaseAvx::vfmax_int8());
    vres1 = _mm256_min_ps(_mm256_max_ps(vres1, QConverterBaseAvx::vfmin_int8()),
                          QConverterBaseAvx::vfmax_int8());
    __m256i vepi32_0 = _mm256_cvtps_epi32(vres0);
    __m256i vepi32_1 = _mm256_cvtps_epi32(vres1);
    __m128i vepi16_lo = _mm_packs_epi32(_mm256_extractf128_si256(vepi32_0, 0),
                                        _mm256_extractf128_si256(vepi32_0, 1));
    __m128i vepi16_ho = _mm_packs_epi32(_mm256_extractf128_si256(vepi32_1, 0),
                                        _mm256_extractf128_si256(vepi32_1, 1));
    __m128i vepi8 = _mm_packs_epi16(vepi16_lo, vepi16_ho);
    return _mm_sub_epi8(vepi8, _mm_set1_epi8(-128));
}

template <>
MEGDNN_ATTRIBUTE_TARGET("avx2")
inline __m256i QConverter::convert(const __m256& vsrc) {
    __m256 vinc = _mm256_blendv_ps(
            QConverterBaseAvx::vfneg_half(), QConverterBaseAvx::vfhalf(),
            _mm256_cmp_ps(vsrc, QConverterBaseAvx::vfzero(), _CMP_GE_OQ));
    return _mm256_cvttps_epi32(_mm256_add_ps(vsrc, vinc));
}

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
