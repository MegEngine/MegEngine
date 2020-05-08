/**
 * \file dnn/src/arm_common/pooling/pooling_hleper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/dtype.h"
#include "src/arm_common/pooling/do_max_pooling_3x3_s2x2_float.h"
#include "src/arm_common/pooling/do_max_pooling_3x3_s2x2_float16.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"

#include "src/arm_common/simd_macro/marm_neon.h"

namespace {

/* ======================= MeanPooler ======================== */
using namespace megdnn;
/**
 * \brief  Mean mode for pooling
 * \tparam area the pooling area size, FH * FW
 * \tparam dtype the input type
 * \tparam ctype the inner raw type
 * \tparam comp_type compute type
 */
template <int area, typename dtype, typename ctype, typename comp_type>
struct MeanPoolerCommon {
    //! the neon register size is 16 bytes(128 bits)
    static constexpr int SIMD_WIDTH = 16 / sizeof(ctype);
    static constexpr comp_type coef = static_cast<comp_type>(1.0f) / area;
    comp_type res;
    MeanPoolerCommon() : res(0) {}
    void feed(const ctype* val) { res += *val; }
};
template <int area, typename dtype, typename ctype, typename comp_type>
constexpr comp_type MeanPoolerCommon<area, dtype, ctype, comp_type>::coef;

template <int area, typename dtype, typename _ctype, typename comp_type>
struct MeanInPooler : MeanPoolerCommon<area, dtype, _ctype, comp_type> {
    using ctype = _ctype;
    //! `MIDOUT_CASE_NUM` is a unique int id
    static constexpr int MIDOUT_CASE_NUM = 1;
    MeanInPooler(DType) : MeanPoolerCommon<area, dtype, _ctype, comp_type>() {}
    void post(ctype* dst) {
        this->res *= this->coef;
        *dst = this->res;
    }
};

template <int area, typename dtype, typename _ctype>
struct MeanInRoundPooler : MeanPoolerCommon<area, dtype, _ctype, float> {
    using ctype = _ctype;
    void post(ctype* dst) {
        this->res *= this->coef;
        *dst = std::round(this->res);
    }
};

template <int area>
struct MeanInPooler<area, dt_qint8, int8_t, float>
        : MeanInRoundPooler<area, dt_qint8, int8_t> {
    static constexpr int MIDOUT_CASE_NUM = 2;
    MeanInPooler(DType) : MeanInRoundPooler<area, dt_qint8, int8_t>() {}
};

template <int area>
struct MeanInPooler<area, dt_quint8, uint8_t, float>
        : MeanInRoundPooler<area, dt_quint8, uint8_t> {
    static constexpr int MIDOUT_CASE_NUM = 3;
    uint8_t zero_point;
    uint8_t feed_cnt;
    MeanInPooler(DType dtype)
            : MeanInRoundPooler<area, dt_quint8, uint8_t>(),
              zero_point(dtype.param<dtype::Quantized8Asymm>().zero_point),
              feed_cnt(0) {}
    void feed(const uint8_t* val) {
        this->res += *val;
        feed_cnt += 1;
    }
    void post(uint8_t* dst) {
        this->res =
                this->res + static_cast<float>(area - feed_cnt) * zero_point;
        this->res *= this->coef;
        *dst = std::round(this->res);
    }
};

template <int area, typename dtype, typename ctype, typename comp_type>
struct NeonMeanPooler;

template <int area>
struct NeonMeanPooler<area, dt_float32, float, float> {
    using ctype = float;
    static constexpr int MIDOUT_CASE_NUM = 1;
    static constexpr int SIMD_WIDTH = 4;

    static const float32x4_t coef;
    float32x4_t res;
    NeonMeanPooler(DType) : res(vdupq_n_f32(0.0f)) {}
    void feed(const float* val) { res = vaddq_f32(res, vld1q_f32(val)); }
    void post(float* dst) {
        res = vmulq_f32(res, coef);
        vst1q_f32(dst, res);
    }
};
template <int area>
const float32x4_t NeonMeanPooler<area, dt_float32, float, float>::coef =
        vdupq_n_f32(1.0f / area);

template <int area>
struct NeonMeanPooler<area, dt_qint8, int8_t, float> {
    using ctype = int8_t;
    static constexpr int MIDOUT_CASE_NUM = 2;
    static constexpr int SIMD_WIDTH = 16;

    static const float32x4_t coef;
#if MEGDNN_ARMV7
    static const float32x4_t fzero;
    static const float32x4_t half;
    static const float32x4_t neg_half;
#endif
    float32x4_t sum0;
    float32x4_t sum1;
    float32x4_t sum2;
    float32x4_t sum3;
    NeonMeanPooler(DType)
            : sum0(vdupq_n_f32(0.0f)),
              sum1(vdupq_n_f32(0.0f)),
              sum2(vdupq_n_f32(0.0f)),
              sum3(vdupq_n_f32(0.0f)) {}
    void feed(const int8_t* val) {
        int8x16_t item = vld1q_s8(val);
        float32x4_t tmp;
#define cb(i)                                                                \
    tmp = (float32x4_t){static_cast<float>(vgetq_lane_s8(item, 4 * i + 0)),  \
                        static_cast<float>(vgetq_lane_s8(item, 4 * i + 1)),  \
                        static_cast<float>(vgetq_lane_s8(item, 4 * i + 2)),  \
                        static_cast<float>(vgetq_lane_s8(item, 4 * i + 3))}; \
    sum##i = vaddq_f32(sum##i, tmp);
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
    }
    void post(int8_t* dst) {
#define cb(i) sum##i = vmulq_f32(sum##i, coef);
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#if MEGDNN_AARCH64
#define cb(i) auto res##i = vcvtaq_s32_f32(sum##i);
#elif MEGDNN_ARMV7
#define cb(i)                                                          \
    auto inc##i = vbslq_f32(vcgeq_f32(sum##i, fzero), half, neg_half); \
    sum##i = vaddq_f32(sum##i, inc##i);                                \
    auto res##i = vcvtq_s32_f32(sum##i);
#else
#error "unsupport android arch"
#endif
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        int8x8_t merge_res1 =
                vqmovn_s16(vcombine_s16(vqmovn_s32(res0), vqmovn_s32(res1)));
        int8x8_t merge_res2 =
                vqmovn_s16(vcombine_s16(vqmovn_s32(res2), vqmovn_s32(res3)));

        vst1q_s8(dst, vcombine_s8(merge_res1, merge_res2));
    }
};
template <int area>
const float32x4_t NeonMeanPooler<area, dt_qint8, int8_t, float>::coef =
        vdupq_n_f32(1.0f / area);
#if MEGDNN_ARMV7
template <int area>
const float32x4_t NeonMeanPooler<area, dt_qint8, int8_t, float>::fzero =
        vdupq_n_f32(0.f);
template <int area>
const float32x4_t NeonMeanPooler<area, dt_qint8, int8_t, float>::half =
        vdupq_n_f32(0.5f);
template <int area>
const float32x4_t NeonMeanPooler<area, dt_qint8, int8_t, float>::neg_half =
        vdupq_n_f32(-0.5f);
#endif

template <int area>
struct NeonMeanPooler<area, dt_quint8, uint8_t, float> {
    using ctype = uint8_t;
    static constexpr int MIDOUT_CASE_NUM = 3;
    static constexpr int SIMD_WIDTH = 16;

    static const float32x4_t coef;
#if MEGDNN_ARMV7
    static const float32x4_t fzero;
    static const float32x4_t half;
    static const float32x4_t neg_half;
#endif
    float32x4_t sum0;
    float32x4_t sum1;
    float32x4_t sum2;
    float32x4_t sum3;
    NeonMeanPooler(DType)
            : sum0(vdupq_n_f32(0.0f)),
              sum1(vdupq_n_f32(0.0f)),
              sum2(vdupq_n_f32(0.0f)),
              sum3(vdupq_n_f32(0.0f)) {}
    void feed(const uint8_t* val) {
        uint8x16_t item = vld1q_u8(val);
        float32x4_t tmp;
#define cb(i)                                                                \
    tmp = (float32x4_t){static_cast<float>(vgetq_lane_u8(item, 4 * i + 0)),  \
                        static_cast<float>(vgetq_lane_u8(item, 4 * i + 1)),  \
                        static_cast<float>(vgetq_lane_u8(item, 4 * i + 2)),  \
                        static_cast<float>(vgetq_lane_u8(item, 4 * i + 3))}; \
    sum##i = vaddq_f32(sum##i, tmp);
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
    }
    void post(uint8_t* dst) {
#define cb(i) sum##i = vmulq_f32(sum##i, coef);
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#if MEGDNN_AARCH64
#define cb(i) auto res##i = vcvtaq_s32_f32(sum##i);
#elif MEGDNN_ARMV7
#define cb(i)                                                          \
    auto inc##i = vbslq_f32(vcgeq_f32(sum##i, fzero), half, neg_half); \
    sum##i = vaddq_f32(sum##i, inc##i);                                \
    auto res##i = vcvtq_s32_f32(sum##i);
#else
#error "unsupport android arch"
#endif
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        uint8x8_t merge_res1 = vqmovn_u16(vreinterpretq_u16_s16(
                vcombine_s16(vqmovn_s32(res0), vqmovn_s32(res1))));
        uint8x8_t merge_res2 = vqmovn_u16(vreinterpretq_u16_s16(
                vcombine_s16(vqmovn_s32(res2), vqmovn_s32(res3))));

        vst1q_u8(dst, vcombine_u8(merge_res1, merge_res2));
    }
};
template <int area>
const float32x4_t NeonMeanPooler<area, dt_quint8, uint8_t, float>::coef =
        vdupq_n_f32(1.0f / area);
#if MEGDNN_ARMV7
template <int area>
const float32x4_t NeonMeanPooler<area, dt_quint8, uint8_t, float>::fzero =
        vdupq_n_f32(0.f);
template <int area>
const float32x4_t NeonMeanPooler<area, dt_quint8, uint8_t, float>::half =
        vdupq_n_f32(0.5f);
template <int area>
const float32x4_t NeonMeanPooler<area, dt_quint8, uint8_t, float>::neg_half =
        vdupq_n_f32(-0.5f);
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <int area>
struct NeonMeanPooler<area, dt_float16, __fp16, __fp16> {
    using ctype = __fp16;
    static constexpr int MIDOUT_CASE_NUM = 4;
    static constexpr int SIMD_WIDTH = 8;

    static const float16x8_t coef;
    float16x8_t res;
    NeonMeanPooler(DType) : res(vdupq_n_f16(0.0f)) {}
    void feed(const __fp16* val) { res = vaddq_f16(res, vld1q_f16(val)); }
    void post(__fp16* dst) {
        res = vmulq_fix_f16(res, coef);
        vst1q_f16(dst, res);
    }
};
template <int area>
const float16x8_t NeonMeanPooler<area, dt_float16, __fp16, __fp16>::coef =
        vdupq_n_f16(1.0f / area);
#endif

/* ======================= MaxPooler ======================== */

template <int area, typename dtype, typename _ctype, typename comp_type>
struct MaxPooler {
    using ctype = _ctype;
    static constexpr int MIDOUT_CASE_NUM = 11;
    static constexpr int SIMD_WIDTH = 16 / sizeof(ctype);

    static const ctype outsider;
    ctype res;
    MaxPooler(DType) : res(DTypeTrait<dtype>::min()) {}
    void feed(const ctype* val) { res = std::max(res, *val); }
    void post(ctype* dst) { *dst = res; }
};
template <int area, typename dtype, typename ctype, typename comp_type>
const ctype MaxPooler<area, dtype, ctype, comp_type>::outsider =
        DTypeTrait<dtype>::min();

template <int area, typename dtype, typename ctype, typename comp_type>
struct NeonMaxPooler;

template <int area>
struct NeonMaxPooler<area, dt_float32, float, float> {
    using ctype = float;
    static constexpr int MIDOUT_CASE_NUM = 11;
    static constexpr int SIMD_WIDTH = 4;

    float32x4_t res;
    NeonMaxPooler(DType) : res(vdupq_n_f32(DTypeTrait<dt_float32>::min())) {}
    void feed(const float* val) { res = vmaxq_f32(res, vld1q_f32(val)); }
    void post(float* dst) { vst1q_f32(dst, res); }
};

template <int area>
struct NeonMaxPooler<area, dt_qint8, int8_t, float> {
    using ctype = int8_t;
    static constexpr int MIDOUT_CASE_NUM = 12;
    static constexpr int SIMD_WIDTH = 16;

    int8x16_t res;
    NeonMaxPooler(DType)
            : res(vdupq_n_s8(std::numeric_limits<int8_t>::lowest())) {}
    void feed(const int8_t* val) { res = vmaxq_s8(res, vld1q_s8(val)); }
    void post(int8_t* dst) { vst1q_s8(dst, res); }
};

template <int area>
struct NeonMaxPooler<area, dt_quint8, uint8_t, float> {
    using ctype = uint8_t;
    static constexpr int MIDOUT_CASE_NUM = 13;
    static constexpr int SIMD_WIDTH = 16;

    uint8x16_t res;
    NeonMaxPooler(DType)
            : res(vdupq_n_u8(std::numeric_limits<uint8_t>::lowest())) {}
    void feed(const uint8_t* val) { res = vmaxq_u8(res, vld1q_u8(val)); }
    void post(uint8_t* dst) { vst1q_u8(dst, res); }
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <int area>
struct NeonMaxPooler<area, dt_float16, __fp16, __fp16> {
    using ctype = __fp16;
    static constexpr int MIDOUT_CASE_NUM = 14;
    static constexpr int SIMD_WIDTH = 8;

    float16x8_t res;
    NeonMaxPooler(DType) : res(vdupq_n_f16(DTypeTrait<dt_float16>::min())) {}
    void feed(const __fp16* val) { res = vmaxq_f16(res, vld1q_f16(val)); }
    void post(__fp16* dst) { vst1q_f16(dst, res); }
};
#endif

template <typename Pooler, int window>
void do_pxl_naive(int oh, int ow, const typename Pooler::ctype* src,
                  typename Pooler::ctype* dst, DType src_dtype, const int IH,
                  const int IW, const int OH, const int OW, const int PH,
                  const int PW, const int SH, const int SW) {
    MEGDNN_MARK_USED_VAR(OH);
    Pooler pooler(src_dtype);
    rep(wh, window) rep(ww, window) {
        int ih = -PH + oh * SH + wh;
        int iw = -PW + ow * SW + ww;
        if (ih >= 0 && iw >= 0 && ih < IH && iw < IW) {
            pooler.feed(src + ih * IW + iw);
        }
    }
    pooler.post(dst + oh * OW + ow);
}

namespace detail {

template <typename Pooler, Pooling::Mode mode>
struct do_pxl_2x2_pack_proxy {
    static void gao(int oh, int ow, const typename Pooler::ctype* src,
                    typename Pooler::ctype* dst, DType, const int IH,
                    const int IW, const int OH, const int OW, const int PH,
                    const int PW);
};

template <>
struct do_pxl_2x2_pack_proxy<MeanInPooler<4, dt_float32, float, float>,
                             Pooling::Mode::AVERAGE> {
    static void gao(int oh, int ow, const dt_float32* src, dt_float32* dst,
                    DType, const int IH, const int IW, const int OH,
                    const int OW, const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        static const auto avg_coef = vdupq_n_f32(0.25f);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_f32(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_f32(src + (ih + 0) * IW + (iw + 4)),
             i10 = vld1q_f32(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_f32(src + (ih + 1) * IW + (iw + 4));
        auto sum0 = vaddq_f32(i00, i10), sum1 = vaddq_f32(i01, i11);
        auto vlow = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
        auto vhigh = vpadd_f32(vget_low_f32(sum1), vget_high_f32(sum1));
        auto comb = vcombine_f32(vlow, vhigh);
        auto result = vmulq_f32(comb, avg_coef);
        vst1q_f32(dst + oh * OW + ow, result);
    }
};

template <>
struct do_pxl_2x2_pack_proxy<MeanInPooler<4, dt_qint8, int8_t, float>,
                             Pooling::Mode::AVERAGE> {
    static void gao(int oh, int ow, const int8_t* src, int8_t* dst, DType,
                    const int IH, const int IW, const int OH, const int OW,
                    const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto zero = vdupq_n_s16(0);
        auto one = vdupq_n_s16(1);
        auto i00 = vld1q_s8(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_s8(src + (ih + 0) * IW + (iw + 16)),
             i10 = vld1q_s8(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_s8(src + (ih + 1) * IW + (iw + 16));
        int16x8_t sum0 = vaddl_s8(vget_low_s8(i00), vget_low_s8(i10)),
                  sum1 = vaddl_s8(vget_high_s8(i00), vget_high_s8(i10)),
                  sum2 = vaddl_s8(vget_low_s8(i01), vget_low_s8(i11)),
                  sum3 = vaddl_s8(vget_high_s8(i01), vget_high_s8(i11));

        auto vlow0 = vpadd_s16(vget_low_s16(sum0), vget_high_s16(sum0));
        auto vhigh0 = vpadd_s16(vget_low_s16(sum1), vget_high_s16(sum1));
        auto vlow1 = vpadd_s16(vget_low_s16(sum2), vget_high_s16(sum2));
        auto vhigh1 = vpadd_s16(vget_low_s16(sum3), vget_high_s16(sum3));
        auto comb0 = vcombine_s16(vlow0, vhigh0);
        auto comb1 = vcombine_s16(vlow1, vhigh1);

        auto fixup0 = vcltq_s16(comb0, zero);
        comb0 = vsubq_s16(comb0, vbslq_s16(fixup0, one, zero));
        //! as vqrshrn_n_s16 is round to positive infinity
        auto result0 = vqrshrn_n_s16(comb0, 2);
        auto fixup1 = vcltq_s16(comb1, zero);
        comb1 = vsubq_s16(comb1, vbslq_s16(fixup1, one, zero));
        auto result1 = vqrshrn_n_s16(comb1, 2);
        vst1q_s8(dst + oh * OW + ow, vcombine_s8(result0, result1));
    }
};

template <>
struct do_pxl_2x2_pack_proxy<MeanInPooler<4, dt_quint8, uint8_t, float>,
                             Pooling::Mode::AVERAGE> {
    static void gao(int oh, int ow, const uint8_t* src, uint8_t* dst, DType,
                    const int IH, const int IW, const int OH, const int OW,
                    const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_u8(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_u8(src + (ih + 0) * IW + (iw + 16)),
             i10 = vld1q_u8(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_u8(src + (ih + 1) * IW + (iw + 16));
        uint16x8_t sum0 = vaddl_u8(vget_low_u8(i00), vget_low_u8(i10)),
                   sum1 = vaddl_u8(vget_high_u8(i00), vget_high_u8(i10)),
                   sum2 = vaddl_u8(vget_low_u8(i01), vget_low_u8(i11)),
                   sum3 = vaddl_u8(vget_high_u8(i01), vget_high_u8(i11));

        auto vlow0 = vpadd_u16(vget_low_u16(sum0), vget_high_u16(sum0));
        auto vhigh0 = vpadd_u16(vget_low_u16(sum1), vget_high_u16(sum1));
        auto vlow1 = vpadd_u16(vget_low_u16(sum2), vget_high_u16(sum2));
        auto vhigh1 = vpadd_u16(vget_low_u16(sum3), vget_high_u16(sum3));
        auto comb0 = vcombine_u16(vlow0, vhigh0);
        auto comb1 = vcombine_u16(vlow1, vhigh1);

        auto result0 = vqrshrn_n_u16(comb0, 2);
        auto result1 = vqrshrn_n_u16(comb1, 2);
        vst1q_u8(dst + oh * OW + ow, vcombine_u8(result0, result1));
    }
};

template <>
struct do_pxl_2x2_pack_proxy<MaxPooler<4, dt_float32, float, float>,
                             Pooling::Mode::MAX> {
    static void gao(int oh, int ow, const dt_float32* src, dt_float32* dst,
                    DType, const int IH, const int IW, const int OH,
                    const int OW, const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_f32(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_f32(src + (ih + 0) * IW + (iw + 4)),
             i10 = vld1q_f32(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_f32(src + (ih + 1) * IW + (iw + 4));
        auto sum0 = vmaxq_f32(i00, i10), sum1 = vmaxq_f32(i01, i11);
        auto vlow = vpmax_f32(vget_low_f32(sum0), vget_high_f32(sum0));
        auto vhigh = vpmax_f32(vget_low_f32(sum1), vget_high_f32(sum1));
        auto comb = vcombine_f32(vlow, vhigh);
        vst1q_f32(dst + oh * OW + ow, comb);
    }
};

template <>
struct do_pxl_2x2_pack_proxy<MaxPooler<4, dt_qint8, int8_t, float>,
                             Pooling::Mode::MAX> {
    static void gao(int oh, int ow, const int8_t* src, int8_t* dst, DType,
                    const int IH, const int IW, const int OH, const int OW,
                    const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_s8(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_s8(src + (ih + 0) * IW + (iw + 16)),
             i10 = vld1q_s8(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_s8(src + (ih + 1) * IW + (iw + 16));
        auto sum0 = vmaxq_s8(i00, i10), sum1 = vmaxq_s8(i01, i11);
        auto vlow = vpmax_s8(vget_low_s8(sum0), vget_high_s8(sum0));
        auto vhigh = vpmax_s8(vget_low_s8(sum1), vget_high_s8(sum1));
        auto comb = vcombine_s8(vlow, vhigh);
        vst1q_s8(dst + oh * OW + ow, comb);
    }
};

template <>
struct do_pxl_2x2_pack_proxy<MaxPooler<4, dt_quint8, uint8_t, float>,
                             Pooling::Mode::MAX> {
    static void gao(int oh, int ow, const uint8_t* src, uint8_t* dst, DType,
                    const int IH, const int IW, const int OH, const int OW,
                    const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_u8(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_u8(src + (ih + 0) * IW + (iw + 16)),
             i10 = vld1q_u8(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_u8(src + (ih + 1) * IW + (iw + 16));
        auto sum0 = vmaxq_u8(i00, i10), sum1 = vmaxq_u8(i01, i11);
        auto vlow = vpmax_u8(vget_low_u8(sum0), vget_high_u8(sum0));
        auto vhigh = vpmax_u8(vget_low_u8(sum1), vget_high_u8(sum1));
        auto comb = vcombine_u8(vlow, vhigh);
        vst1q_u8(dst + oh * OW + ow, comb);
    }
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
struct do_pxl_2x2_pack_proxy<MeanInPooler<4, dt_float16, __fp16, __fp16>,
                             Pooling::Mode::AVERAGE> {
    static void gao(int oh, int ow, const __fp16* src, __fp16* dst, DType,
                    const int IH, const int IW, const int OH, const int OW,
                    const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        static const auto avg_coef = vdupq_n_f16(0.25f);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_f16(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_f16(src + (ih + 0) * IW + (iw + 8)),
             i10 = vld1q_f16(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_f16(src + (ih + 1) * IW + (iw + 8));
        auto sum0 = vaddq_f16(i00, i10), sum1 = vaddq_f16(i01, i11);
        auto vlow = vpadd_f16(vget_low_f16(sum0), vget_high_f16(sum0));
        auto vhigh = vpadd_f16(vget_low_f16(sum1), vget_high_f16(sum1));
        auto comb = vcombine_f16(vlow, vhigh);
        auto result = vmulq_f16(comb, avg_coef);
        vst1q_f16(dst + oh * OW + ow, result);
    }
};

template <>
struct do_pxl_2x2_pack_proxy<MaxPooler<4, dt_float16, __fp16, __fp16>,
                             Pooling::Mode::MAX> {
    static void gao(int oh, int ow, const __fp16* src, __fp16* dst, DType,
                    const int IH, const int IW, const int OH, const int OW,
                    const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = vld1q_f16(src + (ih + 0) * IW + (iw + 0)),
             i01 = vld1q_f16(src + (ih + 0) * IW + (iw + 8)),
             i10 = vld1q_f16(src + (ih + 1) * IW + (iw + 0)),
             i11 = vld1q_f16(src + (ih + 1) * IW + (iw + 8));
        auto sum0 = vmaxq_f16(i00, i10), sum1 = vmaxq_f16(i01, i11);
        auto vlow = vpmax_f16(vget_low_f16(sum0), vget_high_f16(sum0));
        auto vhigh = vpmax_f16(vget_low_f16(sum1), vget_high_f16(sum1));
        auto comb = vcombine_f16(vlow, vhigh);
        vst1q_f16(dst + oh * OW + ow, comb);
    }
};
#endif

}  // namespace detail

template <typename Pooler, Pooling::Mode mode>
void do_pxl_2x2_pack(int oh, int ow, const typename Pooler::ctype* src,
                     typename Pooler::ctype* dst, DType src_dtype, const int IH,
                     const int IW, const int OH, const int OW, const int PH,
                     const int PW) {
    detail::do_pxl_2x2_pack_proxy<Pooler, mode>::gao(
            oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW);
}

template <typename NeonPooler, int window>
void do_pxl_compact_packed(int oh, int ow,
                           const typename NeonPooler::ctype* src,
                           typename NeonPooler::ctype* dst, DType src_dtype,
                           const int IH, const int IW, const int OH,
                           const int OW, const int PH, const int PW) {
    MEGDNN_MARK_USED_VAR(IH);
    MEGDNN_MARK_USED_VAR(OH);
    NeonPooler pooler(src_dtype);
    rep(wh, window) rep(ww, window) {
        int ih = -PH + oh + wh;
        int iw = -PW + ow + ww;
        pooler.feed(src + ih * IW + iw);
    }
    pooler.post(dst + oh * OW + ow);
}

template <typename Pooler, typename NeonPooler, int window>
void do_pooling_compact(const typename Pooler::ctype* src,
                        typename Pooler::ctype* dst, DType src_dtype,
                        const int IH, const int IW, const int OH, const int OW,
                        const int PH, const int PW) {
    static_assert(std::is_same<typename Pooler::ctype,
                               typename NeonPooler::ctype>::value,
                  "ctype of Pooler and NeonPooler is not the same");
    const int stride = 1;
    int oh = 0;
    for (; oh < OH && oh - PH < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH && oh - PH + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && ow - PW < 0; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
        for (; ow + NeonPooler::SIMD_WIDTH <= OW &&
               ow + NeonPooler::SIMD_WIDTH - 1 - PW + window <= IW;
             ow += NeonPooler::SIMD_WIDTH) {
            do_pxl_compact_packed<NeonPooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
}

template <typename Pooler, Pooling::Mode mode>
void do_pooling_2x2(const typename Pooler::ctype* src,
                    typename Pooler::ctype* dst, DType src_dtype, const int IH,
                    const int IW, const int OH, const int OW, const int PH,
                    const int PW) {
    const int window = 2;
    const int stride = 2;
    int oh = 0;
    for (; oh < OH && -PH + stride * oh < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH && -PH + stride * oh + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && -PW + stride * ow < 0; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
        for (; ow + Pooler::SIMD_WIDTH <= OW &&
               -PW + stride * (ow + Pooler::SIMD_WIDTH - 1) + window <= IW;
             ow += Pooler::SIMD_WIDTH) {
            do_pxl_2x2_pack<Pooler, mode>(oh, ow, src, dst, src_dtype, IH, IW,
                                          OH, OW, PH, PW);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
}
inline float32x4_t vload(const float* src) {
    return vld1q_f32(src);
}
inline float32x4x2_t vload2(const float* src) {
    return vld2q_f32(src);
}
inline float32x4_t vdupq(float a) {
    return vdupq_n_f32(a);
}
inline float32x4_t vaddq(float32x4_t a, float32x4_t b) {
    return vaddq_f32(a, b);
}
inline float32x4_t vmulq(float32x4_t a, float32x4_t b) {
    return vmulq_f32(a, b);
}
inline float32x4_t vmax(float32x4_t a, float32x4_t b) {
    return vmaxq_f32(a, b);
}
inline void vset(float* src, float32x4_t dst) {
    vst1q_f32(src, dst);
}
inline float32x4x2_t vunzip(float32x4_t a, float32x4_t b) {
    return vuzpq_f32(a, b);
}

inline int8x16_t vload(const int8_t* src) {
    return vld1q_s8(src);
}
inline int8x16_t vmax(int8x16_t a, int8x16_t b) {
    return vmaxq_s8(a, b);
}
inline void vset(int8_t* src, int8x16_t dst) {
    vst1q_s8(src, dst);
}
inline int8x16x2_t vunzip(int8x16_t a, int8x16_t b) {
    return vuzpq_s8(a, b);
}

inline uint8x16_t vload(const uint8_t* src) {
    return vld1q_u8(src);
}
inline uint8x16_t vmax(uint8x16_t a, uint8x16_t b) {
    return vmaxq_u8(a, b);
}
inline void vset(uint8_t* src, uint8x16_t dst) {
    vst1q_u8(src, dst);
}
inline uint8x16x2_t vunzip(uint8x16_t a, uint8x16_t b) {
    return vuzpq_u8(a, b);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
inline float16x8_t vload(const __fp16* src) {
    return vld1q_f16(src);
}
inline float16x8x2_t vload2(const __fp16* src) {
    return vld2q_f16(src);
}
inline float16x8_t vaddq(float16x8_t a, float16x8_t b) {
    return vaddq_f16(a, b);
}
inline float16x8_t vmulq(float16x8_t a, float16x8_t b) {
    return vmulq_fix_f16(a, b);
}
inline float16x8_t vdupq(__fp16 a) {
    return vdupq_n_f16(a);
}
inline float16x8_t vmax(float16x8_t a, float16x8_t b) {
    return vmaxq_f16(a, b);
}
inline void vset(__fp16* src, float16x8_t dst) {
    vst1q_f16(src, dst);
}
inline float16x8x2_t vunzip(float16x8_t a, float16x8_t b) {
    return vuzpq_f16(a, b);
}
#endif

// because the __fp16 can't get the lowest value, so add dtype
template <typename dtype, typename ctype>
void do_max_pooling_w5x5_s2x2_NEON(const ctype* src, ctype* dst, const int IH,
                                   const int IW, const int OH, const int OW,
                                   const int PH, const int PW,
                                   const WorkspaceBundle& ws,
                                   const int MEGDNN_SIMD_WIDTH) {
    ctype* cache[5] = {
            static_cast<ctype*>(ws.get(0)), static_cast<ctype*>(ws.get(1)),
            static_cast<ctype*>(ws.get(2)), static_cast<ctype*>(ws.get(3)),
            static_cast<ctype*>(ws.get(4))};
    ctype* odd = static_cast<ctype*>(ws.get(5));
    ctype* even = static_cast<ctype*>(ws.get(6));
    int ih_next = 0;
    int OW_from = (PW + 1) / 2, OW_to = (IW + PW - 5) / 2 + 1;
    auto process_cache = [&](int ih) {
        const ctype* __restrict sptr = src + ih * IW;
        auto tmp = cache[4];
        for (auto i = 4; i >= 1; --i)
            cache[i] = cache[i - 1];
        cache[0] = tmp;
        auto run_single = [&](int ow) {
            int iw = ow * 2 - PW;
            ctype res = std::numeric_limits<dtype>::lowest();
            for (auto i = 0; i < 5; ++i)
                if (iw + i >= 0 && iw + i < IW)
                    res = std::max(res, sptr[iw + i]);
            cache[0][ow] = res;
        };
        int iw = 0;
        int odd_offset = 0, even_offset = 0;
        for (; iw + 2 * MEGDNN_SIMD_WIDTH <= IW; iw += 2 * MEGDNN_SIMD_WIDTH) {
            auto s0 = vload(sptr + iw + 0);
            auto s1 = vload(sptr + iw + MEGDNN_SIMD_WIDTH);
            auto d = vunzip(s0, s1);
            vset(even + even_offset, d.val[0]);
            vset(odd + odd_offset, d.val[1]);
            even_offset += MEGDNN_SIMD_WIDTH;
            odd_offset += MEGDNN_SIMD_WIDTH;
        }
        for (; iw < IW; ++iw) {
            if (iw & 1)
                odd[odd_offset++] = sptr[iw];
            else
                even[even_offset++] = sptr[iw];
        }
        int ow = 0;
        for (; ow < OW_from; ++ow)
            run_single(ow);
        if (PW & 1) {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = vload(odd + ow - (PW >> 1) - 1);
                auto s1 = vload(even + ow - (PW >> 1));
                auto s2 = vload(odd + ow - (PW >> 1));
                auto s3 = vload(even + ow - (PW >> 1) + 1);
                auto s4 = vload(odd + ow - (PW >> 1) + 1);
                auto d = vmax(s0, vmax(vmax(s1, s2), vmax(s3, s4)));
                vset(cache[0] + ow, d);
            }
        } else {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = vload(even + ow - (PW >> 1));
                auto s1 = vload(odd + ow - (PW >> 1));
                auto s2 = vload(even + ow - (PW >> 1) + 1);
                auto s3 = vload(odd + ow - (PW >> 1) + 1);
                auto s4 = vload(even + ow - (PW >> 1) + 2);
                auto d = vmax(s0, vmax(vmax(s1, s2), vmax(s3, s4)));
                vset(cache[0] + ow, d);
            }
        }
        for (; ow < OW; ++ow)
            run_single(ow);
    };

    for (int oh = 0; oh < OH; ++oh) {
        ctype* __restrict dptr = dst + oh * OW;
        int ih_from = std::min(IH, std::max(0, oh * 2 - PH));
        int ih_to = std::min(IH, std::max(0, oh * 2 - PH + 5));
        while (ih_next < ih_to)
            process_cache(ih_next++);
        if (ih_to - ih_from == 5) {
            int ow = 0;
            for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = vload(cache[0] + ow);
                auto s1 = vload(cache[1] + ow);
                auto s2 = vload(cache[2] + ow);
                auto s3 = vload(cache[3] + ow);
                auto s4 = vload(cache[4] + ow);
                auto d = vmax(s0, vmax(vmax(s1, s2), vmax(s3, s4)));
                vset(dptr + ow, d);
            }
            for (; ow < OW; ++ow)
                dptr[ow] = std::max({cache[0][ow], cache[1][ow], cache[2][ow],
                                     cache[3][ow], cache[4][ow]});
        } else {
            std::memcpy(dptr, cache[0], sizeof(ctype) * OW);
            for (int i = 1; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                    auto s = vload(cache[i] + ow);
                    auto d = vload(dptr + ow);
                    d = vmax(d, s);
                    vset(dptr + ow, d);
                }
                for (; ow < OW; ++ow)
                    dptr[ow] = std::max(dptr[ow], cache[i][ow]);
            }
        }
    }
}

template <typename ctype>
void do_average_pooling_3x3_s2x2_NEON(const ctype* src, ctype* dst, size_t IH_,
                                      size_t IW_, size_t OH_, size_t OW_,
                                      size_t PH_, size_t PW_,
                                      const WorkspaceBundle& ws,
                                      const int MEGDNN_SIMD_WIDTH) {
    int IH = IH_, IW = IW_, OH = OH_, OW = OW_, PH = PH_, PW = PW_;
    // cache[i] stores the answer of the i-th line after
    // pooling along the W dimension.
    ctype* cache[3] = {static_cast<ctype*>(ws.get(0)),
                       static_cast<ctype*>(ws.get(1)),
                       static_cast<ctype*>(ws.get(2))};
    ctype* odd = static_cast<ctype*>(ws.get(3));
    ctype* even = static_cast<ctype*>(ws.get(4));
    int ih_next = 0;
    // "good" area means we can use SIMD to accelerate.
    auto get_good_area = [](int I, int /* O */, int P, int& O_from, int& O_to) {
        // x*2 - P >= 0; 2x >= P; x >= P/2
        O_from = (P + 1) / 2;
        // x*2 - P + 3 <= I; x*2 <= I+P-3; x <= (I+P-3)/2
        O_to = (I + P - 3) / 2 + 1;
        // we must have I >= 2 to ensure O_from <= O_to
    };
    int OW_from, OW_to;
    get_good_area(IW, OW, PW, OW_from, OW_to);
    auto process_cache = [&](int ih) {
        const ctype* __restrict sptr = src + ih * IW;
        auto tmp = cache[2];
        cache[2] = cache[1];
        cache[1] = cache[0];
        cache[0] = tmp;
        // cache 0 is used to store the current answer.
        auto run_single = [&](int ow) {
            int iw = ow * 2 - PW;
            ctype res = 0;
            if (iw + 0 >= 0 && iw + 0 < IW) {
                res += sptr[iw + 0];
            }
            if (iw + 1 >= 0 && iw + 1 < IW) {
                res += sptr[iw + 1];
            }
            if (iw + 2 >= 0 && iw + 2 < IW) {
                res += sptr[iw + 2];
            }
            cache[0][ow] = res;
        };
        // build odd/even
        int iw = 0;
        int odd_offset = 0, even_offset = 0;

        for (; iw + 2 * MEGDNN_SIMD_WIDTH <= IW; iw += 2 * MEGDNN_SIMD_WIDTH) {
            auto s0 = vload2(sptr + iw);
            vset(even + even_offset, s0.val[0]);
            vset(odd + odd_offset, s0.val[1]);
            even_offset += MEGDNN_SIMD_WIDTH;
            odd_offset += MEGDNN_SIMD_WIDTH;
        }
        for (; iw < IW; ++iw) {
            if (iw & 1)
                odd[odd_offset++] = sptr[iw];
            else
                even[even_offset++] = sptr[iw];
        }
        int ow = 0;
        for (; ow < OW_from; ++ow)
            run_single(ow);
        if (PW & 1) {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = vload(odd + ow - (PW >> 1) - 1);
                auto s1 = vload(even + ow - (PW >> 1));
                auto s2 = vload(odd + ow - (PW >> 1));
                auto d = vaddq(vaddq(s0, s1), s2);
                vset(cache[0] + ow, d);
            }
        } else {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = vload(even + ow - (PW >> 1));
                auto s1 = vload(odd + ow - (PW >> 1));
                auto s2 = vload(even + ow - (PW >> 1) + 1);
                auto d = vaddq(vaddq(s0, s1), s2);
                vset(cache[0] + ow, d);
            }
        }
        for (; ow < OW; ++ow)
            run_single(ow);
    };
    for (int oh = 0; oh < OH; ++oh) {
        ctype* __restrict dptr = dst + oh * OW;
        int ih_from = std::min(IH, std::max(0, oh * 2 - PH));
        int ih_to = std::min(IH, std::max(0, oh * 2 - PH + 3));
        while (ih_next < ih_to) {
            process_cache(ih_next++);
        }
        ctype factor = (1.0f / 9);
        auto coef = vdupq(factor);
        if (ih_to - ih_from == 3) {
            int ow = 0;
            for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = vload(cache[0] + ow);
                auto s1 = vload(cache[1] + ow);
                auto s2 = vload(cache[2] + ow);
                auto d = vaddq(vaddq(s0, s1), s2);
                d = vmulq(d, coef);
                vset(dptr + ow, d);
            }
#if MEGDNN_FIX_AARCH32_BUG
// FIXME: as llvm may cause cannot select error if enable vectorize
#pragma clang loop vectorize(disable)
#endif
            for (; ow < OW; ++ow) {
                dptr[ow] =
                        (cache[0][ow] + cache[1][ow] + cache[2][ow]) * factor;
            }
        } else {
            std::memcpy(dptr, cache[0], sizeof(ctype) * OW);
            int i = 1;
            for (; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                    auto s = vload(cache[i] + ow);
                    auto d = vload(dptr + ow);
                    d = vaddq(d, s);
                    vset(dptr + ow, d);
                }
                for (; ow < OW; ++ow) {
                    dptr[ow] = (dptr[ow] + cache[i][ow]);
                }
            }
            int ow = 0;
            for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                auto d = vload(dptr + ow);
                d = vmulq(d, coef);
                vset(dptr + ow, d);
            }
#if MEGDNN_FIX_AARCH32_BUG
// FIXME: as llvm may cause cannot select error if enable vectorize
#pragma clang loop vectorize(disable)
#endif
            for (; ow < OW; ++ow) {
                dptr[ow] *= factor;
            }
        }
    }
}
}  // anonymous namespace

// vim: syntax=cpp.doxygen
