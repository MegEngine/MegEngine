/**
 * \file dnn/src/fallback/pooling/gi/pooling_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "do_max_pooling_3x3_s2x2_float.h"
#include "megdnn/dtype.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"

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
    //! the gi imp register size is 16 bytes(128 bits)
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

template <int area, typename dtype, typename ctype, typename comp_type>
struct GiMeanPooler;

template <int area>
struct GiMeanPooler<area, dt_float32, float, float> {
    using ctype = float;
    static constexpr int MIDOUT_CASE_NUM = 1;
    static constexpr int SIMD_WIDTH = 4;

    static const GI_FLOAT32_t coef;
    GI_FLOAT32_t res;
    GiMeanPooler(DType) : res(GiBroadcastFloat32(0.0f)) {}
    void feed(const float* val) { res = GiAddFloat32(res, GiLoadFloat32(val)); }
    void post(float* dst) {
        res = GiMultiplyFloat32(res, coef);
        GiStoreFloat32(dst, res);
    }
};
template <int area>
const GI_FLOAT32_t GiMeanPooler<area, dt_float32, float, float>::coef =
        GiBroadcastFloat32(1.0f / area);

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
struct GiMaxPooler;

template <int area>
struct GiMaxPooler<area, dt_float32, float, float> {
    using ctype = float;
    static constexpr int MIDOUT_CASE_NUM = 11;
    static constexpr int SIMD_WIDTH = 4;

    GI_FLOAT32_t res;
    GiMaxPooler(DType) : res(GiBroadcastFloat32(DTypeTrait<dt_float32>::min())) {}
    void feed(const float* val) { res = GiMaximumFloat32(res, GiLoadFloat32(val)); }
    void post(float* dst) { GiStoreFloat32(dst, res); }
};

template <typename Pooler, int window>
void do_pxl_naive(
        int oh, int ow, const typename Pooler::ctype* src, typename Pooler::ctype* dst,
        DType src_dtype, const int IH, const int IW, const int OH, const int OW,
        const int PH, const int PW, const int SH, const int SW) {
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
    static void gao(
            int oh, int ow, const typename Pooler::ctype* src,
            typename Pooler::ctype* dst, DType, const int IH, const int IW,
            const int OH, const int OW, const int PH, const int PW);
};

template <>
struct do_pxl_2x2_pack_proxy<
        MeanInPooler<4, dt_float32, float, float>, Pooling::Mode::AVERAGE> {
    static void gao(
            int oh, int ow, const dt_float32* src, dt_float32* dst, DType, const int IH,
            const int IW, const int OH, const int OW, const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        static const auto avg_coef = GiBroadcastFloat32(0.25f);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = GiLoadFloat32(src + (ih + 0) * IW + (iw + 0)),
             i01 = GiLoadFloat32(src + (ih + 0) * IW + (iw + 4)),
             i10 = GiLoadFloat32(src + (ih + 1) * IW + (iw + 0)),
             i11 = GiLoadFloat32(src + (ih + 1) * IW + (iw + 4));
        auto sum0 = GiAddFloat32(i00, i10), sum1 = GiAddFloat32(i01, i11);
        auto vlow = GiPaddFloat32(GiGetLowFloat32(sum0), GiGetHighFloat32(sum0));
        auto vhigh = GiPaddFloat32(GiGetLowFloat32(sum1), GiGetHighFloat32(sum1));
        auto comb = GiCombineFloat32(vlow, vhigh);
        auto result = GiMultiplyFloat32(comb, avg_coef);
        GiStoreFloat32(dst + oh * OW + ow, result);
    }
};

template <>
struct do_pxl_2x2_pack_proxy<
        MaxPooler<4, dt_float32, float, float>, Pooling::Mode::MAX> {
    static void gao(
            int oh, int ow, const dt_float32* src, dt_float32* dst, DType, const int IH,
            const int IW, const int OH, const int OW, const int PH, const int PW) {
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(OH);
        int ih = -PH + 2 * oh;
        int iw = -PW + 2 * ow;
        auto i00 = GiLoadFloat32(src + (ih + 0) * IW + (iw + 0)),
             i01 = GiLoadFloat32(src + (ih + 0) * IW + (iw + 4)),
             i10 = GiLoadFloat32(src + (ih + 1) * IW + (iw + 0)),
             i11 = GiLoadFloat32(src + (ih + 1) * IW + (iw + 4));
        auto sum0 = GiMaximumFloat32(i00, i10), sum1 = GiMaximumFloat32(i01, i11);
        auto vlow = GiPmaxFloat32(GiGetLowFloat32(sum0), GiGetHighFloat32(sum0));
        auto vhigh = GiPmaxFloat32(GiGetLowFloat32(sum1), GiGetHighFloat32(sum1));
        auto comb = GiCombineFloat32(vlow, vhigh);
        GiStoreFloat32(dst + oh * OW + ow, comb);
    }
};

}  // namespace detail

template <typename Pooler, Pooling::Mode mode>
void do_pxl_2x2_pack(
        int oh, int ow, const typename Pooler::ctype* src, typename Pooler::ctype* dst,
        DType src_dtype, const int IH, const int IW, const int OH, const int OW,
        const int PH, const int PW) {
    detail::do_pxl_2x2_pack_proxy<Pooler, mode>::gao(
            oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW);
}

template <typename GiPooler, int window>
void do_pxl_compact_packed(
        int oh, int ow, const typename GiPooler::ctype* src,
        typename GiPooler::ctype* dst, DType src_dtype, const int IH, const int IW,
        const int OH, const int OW, const int PH, const int PW) {
    MEGDNN_MARK_USED_VAR(IH);
    MEGDNN_MARK_USED_VAR(OH);
    GiPooler pooler(src_dtype);
    rep(wh, window) rep(ww, window) {
        int ih = -PH + oh + wh;
        int iw = -PW + ow + ww;
        pooler.feed(src + ih * IW + iw);
    }
    pooler.post(dst + oh * OW + ow);
}

template <typename Pooler, typename GiPooler, int window>
void do_pooling_compact(
        const typename Pooler::ctype* src, typename Pooler::ctype* dst, DType src_dtype,
        const int IH, const int IW, const int OH, const int OW, const int PH,
        const int PW) {
    static_assert(
            std::is_same<typename Pooler::ctype, typename GiPooler::ctype>::value,
            "ctype of Pooler and GiPooler is not the same");
    const int stride = 1;
    int oh = 0;
    for (; oh < OH && oh - PH < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
    }
    for (; oh < OH && oh - PH + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && ow - PW < 0; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
        for (; ow + GiPooler::SIMD_WIDTH <= OW &&
               ow + GiPooler::SIMD_WIDTH - 1 - PW + window <= IW;
             ow += GiPooler::SIMD_WIDTH) {
            do_pxl_compact_packed<GiPooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
    }
}

template <typename Pooler, Pooling::Mode mode>
void do_pooling_2x2(
        const typename Pooler::ctype* src, typename Pooler::ctype* dst, DType src_dtype,
        const int IH, const int IW, const int OH, const int OW, const int PH,
        const int PW) {
    const int window = 2;
    const int stride = 2;
    int oh = 0;
    for (; oh < OH && -PH + stride * oh < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
    }
    for (; oh < OH && -PH + stride * oh + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && -PW + stride * ow < 0; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
        for (; ow + Pooler::SIMD_WIDTH <= OW &&
               -PW + stride * (ow + Pooler::SIMD_WIDTH - 1) + window <= IW;
             ow += Pooler::SIMD_WIDTH) {
            do_pxl_2x2_pack<Pooler, mode>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(
                    oh, ow, src, dst, src_dtype, IH, IW, OH, OW, PH, PW, stride,
                    stride);
        }
    }
}

template <typename dtype, typename ctype>
void do_max_pooling_w5x5_s2x2_gi(
        const ctype* src, ctype* dst, const int IH, const int IW, const int OH,
        const int OW, const int PH, const int PW, const WorkspaceBundle& ws,
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
            auto s0 = GiLoadFloat32(sptr + iw + 0);
            auto s1 = GiLoadFloat32(sptr + iw + MEGDNN_SIMD_WIDTH);
            auto d = GiUzpqFloat32(s0, s1);
            GiStoreFloat32(even + even_offset, d.val[0]);
            GiStoreFloat32(odd + odd_offset, d.val[1]);
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
                auto s0 = GiLoadFloat32(odd + ow - (PW >> 1) - 1);
                auto s1 = GiLoadFloat32(even + ow - (PW >> 1));
                auto s2 = GiLoadFloat32(odd + ow - (PW >> 1));
                auto s3 = GiLoadFloat32(even + ow - (PW >> 1) + 1);
                auto s4 = GiLoadFloat32(odd + ow - (PW >> 1) + 1);
                auto d = GiMaximumFloat32(
                        s0,
                        GiMaximumFloat32(
                                GiMaximumFloat32(s1, s2), GiMaximumFloat32(s3, s4)));
                GiStoreFloat32(cache[0] + ow, d);
            }
        } else {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = GiLoadFloat32(even + ow - (PW >> 1));
                auto s1 = GiLoadFloat32(odd + ow - (PW >> 1));
                auto s2 = GiLoadFloat32(even + ow - (PW >> 1) + 1);
                auto s3 = GiLoadFloat32(odd + ow - (PW >> 1) + 1);
                auto s4 = GiLoadFloat32(even + ow - (PW >> 1) + 2);
                auto d = GiMaximumFloat32(
                        s0,
                        GiMaximumFloat32(
                                GiMaximumFloat32(s1, s2), GiMaximumFloat32(s3, s4)));
                GiStoreFloat32(cache[0] + ow, d);
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
                auto s0 = GiLoadFloat32(cache[0] + ow);
                auto s1 = GiLoadFloat32(cache[1] + ow);
                auto s2 = GiLoadFloat32(cache[2] + ow);
                auto s3 = GiLoadFloat32(cache[3] + ow);
                auto s4 = GiLoadFloat32(cache[4] + ow);
                auto d = GiMaximumFloat32(
                        s0,
                        GiMaximumFloat32(
                                GiMaximumFloat32(s1, s2), GiMaximumFloat32(s3, s4)));
                GiStoreFloat32(dptr + ow, d);
            }
            for (; ow < OW; ++ow)
                dptr[ow] = std::max(
                        {cache[0][ow], cache[1][ow], cache[2][ow], cache[3][ow],
                         cache[4][ow]});
        } else {
            std::memcpy(dptr, cache[0], sizeof(ctype) * OW);
            for (int i = 1; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                    auto s = GiLoadFloat32(cache[i] + ow);
                    auto d = GiLoadFloat32(dptr + ow);
                    d = GiMaximumFloat32(d, s);
                    GiStoreFloat32(dptr + ow, d);
                }
                for (; ow < OW; ++ow)
                    dptr[ow] = std::max(dptr[ow], cache[i][ow]);
            }
        }
    }
}

template <typename ctype>
void do_average_pooling_3x3_s2x2_gi(
        const ctype* src, ctype* dst, size_t IH_, size_t IW_, size_t OH_, size_t OW_,
        size_t PH_, size_t PW_, const WorkspaceBundle& ws,
        const int MEGDNN_SIMD_WIDTH) {
    int IH = IH_, IW = IW_, OH = OH_, OW = OW_, PH = PH_, PW = PW_;
    // cache[i] stores the answer of the i-th line after
    // pooling along the W dimension.
    ctype* cache[3] = {
            static_cast<ctype*>(ws.get(0)), static_cast<ctype*>(ws.get(1)),
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
            auto s0 = GiLd2qFloat32(sptr + iw);
            GiStoreFloat32(even + even_offset, s0.val[0]);
            GiStoreFloat32(odd + odd_offset, s0.val[1]);
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
                auto s0 = GiLoadFloat32(odd + ow - (PW >> 1) - 1);
                auto s1 = GiLoadFloat32(even + ow - (PW >> 1));
                auto s2 = GiLoadFloat32(odd + ow - (PW >> 1));
                auto d = GiAddFloat32(GiAddFloat32(s0, s1), s2);
                GiStoreFloat32(cache[0] + ow, d);
            }
        } else {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = GiLoadFloat32(even + ow - (PW >> 1));
                auto s1 = GiLoadFloat32(odd + ow - (PW >> 1));
                auto s2 = GiLoadFloat32(even + ow - (PW >> 1) + 1);
                auto d = GiAddFloat32(GiAddFloat32(s0, s1), s2);
                GiStoreFloat32(cache[0] + ow, d);
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
        auto coef = GiBroadcastFloat32(factor);
        if (ih_to - ih_from == 3) {
            int ow = 0;
            for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                auto s0 = GiLoadFloat32(cache[0] + ow);
                auto s1 = GiLoadFloat32(cache[1] + ow);
                auto s2 = GiLoadFloat32(cache[2] + ow);
                auto d = GiAddFloat32(GiAddFloat32(s0, s1), s2);
                d = GiMultiplyFloat32(d, coef);
                GiStoreFloat32(dptr + ow, d);
            }
#if MEGDNN_FIX_AARCH32_BUG
// FIXME: as llvm may cause cannot select error if enable vectorize
#pragma clang loop vectorize(disable)
#endif
            for (; ow < OW; ++ow) {
                dptr[ow] = (cache[0][ow] + cache[1][ow] + cache[2][ow]) * factor;
            }
        } else {
            std::memcpy(dptr, cache[0], sizeof(ctype) * OW);
            int i = 1;
            for (; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                    auto s = GiLoadFloat32(cache[i] + ow);
                    auto d = GiLoadFloat32(dptr + ow);
                    d = GiAddFloat32(d, s);
                    GiStoreFloat32(dptr + ow, d);
                }
                for (; ow < OW; ++ow) {
                    dptr[ow] = (dptr[ow] + cache[i][ow]);
                }
            }
            int ow = 0;
            for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                auto d = GiLoadFloat32(dptr + ow);
                d = GiMultiplyFloat32(d, coef);
                GiStoreFloat32(dptr + ow, d);
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
