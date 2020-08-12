/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/conv_stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/conv_bias/int8x8x16/conv_stride2.h"
#include "src/common/utils.h"

#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"

#pragma GCC diagnostic ignored "-Wunused-parameter"

using namespace megdnn;
using namespace arm_common;
using namespace conv_bias;

template <bool add_to_dst>
void conv_bias::conv_stride2_2x2_sc_int8_int8_int16(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW) {
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH + PH - 2, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW + PW - 2, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh * OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 2; ++fh)
            for (size_t fw = 0; fw < 2; ++fw) {
                size_t ih = oh * 2 + fh - PH;
                size_t iw = ow * 2 + fw - PW;
                if (ih < IH && iw < IW) {
                    dst[oh * OW + ow] +=
                            (int16_t)src[ih * IW + iw] * filter[fh * 2 + fw];
                }
            }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow)
            run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow)
            run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 1x4 block
    int8_t workspace[16];
    for (size_t i = 0; i < 8; ++i)
        workspace[i] = filter[i & 1];
    for (size_t i = 8; i < 16; ++i)
        workspace[i] = filter[(i & 1) + 2];
    int8x8_t f0 = vld1_s8(workspace + 0), f1 = vld1_s8(workspace + 8);
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        size_t ih = oh * 2 - PH;
        const int8_t* sptr = src + ih * IW + (OW_start * 2 - PW);
        int16_t* dptr = dst + oh * OW + OW_start;
        size_t ow = OW_start;
        for (; ow + 4 <= OW_stop; ow += 4) {
            int8x8_t s0 = vld1_s8(sptr + 0 * IW), s1 = vld1_s8(sptr + 1 * IW);
            int16x8_t r0 = vmull_s8(s0, f0), r1 = vmull_s8(s1, f1);
            int16x8_t tmp0 = vaddq_s16(r0, r1);
            int32x4_t tmp1 = vpaddlq_s16(tmp0);
            int16x4_t d = vmovn_s32(tmp1);
            if (add_to_dst) {
                d = vadd_s16(d, vld1_s16(dptr));
            }
            vst1_s16(dptr, d);
            sptr += 8;
            dptr += 4;
        }
        for (; ow < OW_stop; ++ow) {
            int16_t s0 = sptr[0], s1 = sptr[1], s2 = sptr[IW + 0],
                    s3 = sptr[IW + 1];
            int16_t f0 = filter[0], f1 = filter[1], f2 = filter[2],
                    f3 = filter[3];
            int16_t d = s0 * f0 + s1 * f1 + s2 * f2 + s3 * f3;
            if (add_to_dst) {
                *dptr += d;
            } else {
                *dptr = d;
            }
            sptr += 2;
            dptr += 1;
        }
    }
}

template <bool add_to_dst>
void conv_bias::conv_stride2_3x3_sc_int8_int8_int16(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW) {
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH + PH - 3, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW + PW - 3, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh * OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 3; ++fh)
            for (size_t fw = 0; fw < 3; ++fw) {
                size_t ih = oh * 2 + fh - PH;
                size_t iw = ow * 2 + fw - PW;
                if (ih < IH && iw < IW) {
                    dst[oh * OW + ow] +=
                            (int16_t)src[ih * IW + iw] * filter[fh * 3 + fw];
                }
            }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow)
            run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow)
            run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4x8 block
    size_t oh = OH_start;
    int8_t cache_even[9 * 16];
    int8_t cache_odd[9 * 16];
    const int8_t*(sptrs[3]) = {cache_even + 0, cache_odd + 0, cache_even + 1};
    for (; oh + 4 <= OH_stop; oh += 4) {
        size_t ih = oh * 2 - PH;
        size_t ow = OW_start;
        for (; ow + 8 <= OW_stop; ow += 8) {
            size_t iw = ow * 2 - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2, d3;
            int8x8_t k0, k1, k2, s;
            {
                // do transpose
                for (size_t i = 0; i < 9; ++i) {
                    int8x16_t s_full = vld1q_s8(sptr + i * IW);
                    int8x8_t s_low = vget_low_s8(s_full);
                    int8x8_t s_high = vget_high_s8(s_full);
                    int8x8x2_t s_result = vuzp_s8(s_low, s_high);
                    vst1_s8(cache_even + i * 16, s_result.val[0]);
                    vst1_s8(cache_odd + i * 16, s_result.val[1]);
                    // the 8-th elem
                    cache_even[i * 16 + 8] = sptr[i * IW + 16];
                }
            }
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
                d3 = vld1q_s16(dptr + 3 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
                d3 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 3; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 3 + fw]);
                k1 = vdup_n_s8(fptr[1 * 3 + fw]);
                k2 = vdup_n_s8(fptr[2 * 3 + fw]);

                // line 0
                s = vld1_s8(sptrs[fw] + 0 * 16);
                d0 = vmlal_s8(d0, k0, s);

                // line 1
                s = vld1_s8(sptrs[fw] + 1 * 16);
                d0 = vmlal_s8(d0, k1, s);

                // line 2
                s = vld1_s8(sptrs[fw] + 2 * 16);
                d0 = vmlal_s8(d0, k2, s);
                d1 = vmlal_s8(d1, k0, s);

                // line 3
                s = vld1_s8(sptrs[fw] + 3 * 16);
                d1 = vmlal_s8(d1, k1, s);

                // line 4
                s = vld1_s8(sptrs[fw] + 4 * 16);
                d1 = vmlal_s8(d1, k2, s);
                d2 = vmlal_s8(d2, k0, s);

                // line 5
                s = vld1_s8(sptrs[fw] + 5 * 16);
                d2 = vmlal_s8(d2, k1, s);

                // line 6
                s = vld1_s8(sptrs[fw] + 6 * 16);
                d2 = vmlal_s8(d2, k2, s);
                d3 = vmlal_s8(d3, k0, s);

                // line 7
                s = vld1_s8(sptrs[fw] + 7 * 16);
                d3 = vmlal_s8(d3, k1, s);

                // line 8
                s = vld1_s8(sptrs[fw] + 8 * 16);
                d3 = vmlal_s8(d3, k2, s);
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
            vst1q_s16(dptr + 3 * OW, d3);
        }
        for (; ow < OW_stop; ++ow) {
            run_single(oh + 0, ow);
            run_single(oh + 1, ow);
            run_single(oh + 2, ow);
            run_single(oh + 3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}

template <bool add_to_dst>
void conv_bias::conv_stride2_5x5_sc_int8_int8_int16(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW) {
    size_t OH_start = div_ceil<size_t>(PH, 2),
           OH_stop = div_floor<size_t>(IH + PH - 5, 2) + 1,
           OW_start = div_ceil<size_t>(PW, 2),
           OW_stop = div_floor<size_t>(IW + PW - 5, 2) + 1;
    OH_start = std::min<size_t>(OH, OH_start);
    OH_stop = std::min<size_t>(OH, OH_stop);
    OW_start = std::min<size_t>(OW, OW_start);
    OW_stop = std::min<size_t>(OW, OW_stop);
    auto run_single = [&](size_t oh, size_t ow) {
        if (!add_to_dst) {
            dst[oh * OW + ow] = 0;
        }
        for (size_t fh = 0; fh < 5; ++fh)
            for (size_t fw = 0; fw < 5; ++fw) {
                size_t ih = oh * 2 + fh - PH;
                size_t iw = ow * 2 + fw - PW;
                if (ih < IH && iw < IW) {
                    dst[oh * OW + ow] +=
                            (int16_t)src[ih * IW + iw] * filter[fh * 5 + fw];
                }
            }
    };
    for (size_t oh = 0; oh < OH_start; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    for (size_t oh = OH_start; oh < OH_stop; ++oh) {
        for (size_t ow = 0; ow < OW_start; ++ow)
            run_single(oh, ow);
        for (size_t ow = OW_stop; ow < OW; ++ow)
            run_single(oh, ow);
    }
    for (size_t oh = OH_stop; oh < OH; ++oh) {
        for (size_t ow = 0; ow < OW; ++ow) {
            run_single(oh, ow);
        }
    }
    // 4x8 block
    size_t oh = OH_start;
    int8_t cache_even[11 * 16];
    int8_t cache_odd[11 * 16];
    const int8_t*(sptrs[5]) = {
            cache_even + 0, cache_odd + 0,  cache_even + 1,
            cache_odd + 1,  cache_even + 2,
    };
    for (; oh + 4 <= OH_stop; oh += 4) {
        size_t ih = oh * 2 - PH;
        size_t ow = OW_start;
        for (; ow + 8 <= OW_stop; ow += 8) {
            size_t iw = ow * 2 - PW;
            int16_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int8_t* __restrict fptr = filter;
            int16x8_t d0, d1, d2, d3;
            int8x8_t k0, k1, k2, k3, k4, s;
            {
                // do transpose
                for (size_t i = 0; i < 11; ++i) {
                    int8x16_t s_full = vld1q_s8(sptr + i * IW);
                    int8x8_t s_low = vget_low_s8(s_full);
                    int8x8_t s_high = vget_high_s8(s_full);
                    int8x8x2_t s_result = vuzp_s8(s_low, s_high);
                    vst1_s8(cache_even + i * 16, s_result.val[0]);
                    vst1_s8(cache_odd + i * 16, s_result.val[1]);
                    // last elements
                    cache_even[i * 16 + 8] = sptr[i * IW + 16];
                    cache_odd[i * 16 + 8] = sptr[i * IW + 17];
                    cache_even[i * 16 + 9] = sptr[i * IW + 18];
                }
            }
            if (add_to_dst) {
                d0 = vld1q_s16(dptr + 0 * OW);
                d1 = vld1q_s16(dptr + 1 * OW);
                d2 = vld1q_s16(dptr + 2 * OW);
                d3 = vld1q_s16(dptr + 3 * OW);
            } else {
                d0 = vdupq_n_s16(0);
                d1 = vdupq_n_s16(0);
                d2 = vdupq_n_s16(0);
                d3 = vdupq_n_s16(0);
            }
            for (size_t fw = 0; fw < 5; ++fw) {
                k0 = vdup_n_s8(fptr[0 * 5 + fw]);
                k1 = vdup_n_s8(fptr[1 * 5 + fw]);
                k2 = vdup_n_s8(fptr[2 * 5 + fw]);
                k3 = vdup_n_s8(fptr[3 * 5 + fw]);
                k4 = vdup_n_s8(fptr[4 * 5 + fw]);

                // line 0
                s = vld1_s8(sptrs[fw] + 0 * 16);
                d0 = vmlal_s8(d0, k0, s);

                // line 1
                s = vld1_s8(sptrs[fw] + 1 * 16);
                d0 = vmlal_s8(d0, k1, s);

                // line 2
                s = vld1_s8(sptrs[fw] + 2 * 16);
                d0 = vmlal_s8(d0, k2, s);
                d1 = vmlal_s8(d1, k0, s);

                // line 3
                s = vld1_s8(sptrs[fw] + 3 * 16);
                d0 = vmlal_s8(d0, k3, s);
                d1 = vmlal_s8(d1, k1, s);

                // line 4
                s = vld1_s8(sptrs[fw] + 4 * 16);
                d0 = vmlal_s8(d0, k4, s);
                d1 = vmlal_s8(d1, k2, s);
                d2 = vmlal_s8(d2, k0, s);

                // line 5
                s = vld1_s8(sptrs[fw] + 5 * 16);
                d1 = vmlal_s8(d1, k3, s);
                d2 = vmlal_s8(d2, k1, s);

                // line 6
                s = vld1_s8(sptrs[fw] + 6 * 16);
                d1 = vmlal_s8(d1, k4, s);
                d2 = vmlal_s8(d2, k2, s);
                d3 = vmlal_s8(d3, k0, s);

                // line 7
                s = vld1_s8(sptrs[fw] + 7 * 16);
                d2 = vmlal_s8(d2, k3, s);
                d3 = vmlal_s8(d3, k1, s);

                // line 8
                s = vld1_s8(sptrs[fw] + 8 * 16);
                d2 = vmlal_s8(d2, k4, s);
                d3 = vmlal_s8(d3, k2, s);

                // line 9
                s = vld1_s8(sptrs[fw] + 9 * 16);
                d3 = vmlal_s8(d3, k3, s);

                // line 9
                s = vld1_s8(sptrs[fw] + 10 * 16);
                d3 = vmlal_s8(d3, k4, s);
            }
            vst1q_s16(dptr + 0 * OW, d0);
            vst1q_s16(dptr + 1 * OW, d1);
            vst1q_s16(dptr + 2 * OW, d2);
            vst1q_s16(dptr + 3 * OW, d3);
        }
        for (; ow < OW_stop; ++ow) {
            run_single(oh + 0, ow);
            run_single(oh + 1, ow);
            run_single(oh + 2, ow);
            run_single(oh + 3, ow);
        }
    }
    for (; oh < OH_stop; ++oh) {
        for (size_t ow = OW_start; ow < OW_stop; ++ow) {
            run_single(oh, ow);
        }
    }
}
template void conv_bias::conv_stride2_2x2_sc_int8_int8_int16<true>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_stride2_2x2_sc_int8_int8_int16<false>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_stride2_3x3_sc_int8_int8_int16<true>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_stride2_3x3_sc_int8_int8_int16<false>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_stride2_5x5_sc_int8_int8_int16<true>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);
template void conv_bias::conv_stride2_5x5_sc_int8_int8_int16<false>(
        const int8_t* src, const int8_t* filter, int16_t* dst, size_t IH,
        size_t IW, size_t OH, size_t OW, size_t PH, size_t PW);

namespace {
void conv_2x2_optimize_single_channel(const int8_t* src, const uint32_t IH,
                                      const uint32_t IW, const int8_t* filter,
                                      int16_t* dst, const uint32_t OH,
                                      const uint32_t OW) {
    int8_t workspace[16];
    workspace[0] = filter[0];
    workspace[1] = filter[1];
    workspace[2] = filter[0];
    workspace[3] = filter[1];
    workspace[4] = filter[0];
    workspace[5] = filter[1];
    workspace[6] = filter[0];
    workspace[7] = filter[1];
    workspace[8] = filter[2];
    workspace[9] = filter[3];
    workspace[10] = filter[2];
    workspace[11] = filter[3];
    workspace[12] = filter[2];
    workspace[13] = filter[3];
    workspace[14] = filter[2];
    workspace[15] = filter[3];
    int8x8_t f0 = vld1_s8(workspace), f1 = vld1_s8(workspace + 8);

    int8x8_t v0, v1;
    int16x8_t r0, r1, s0;
    int16x4_t s, s16;
    for (uint32_t i = 0, j; i < IH; i += 2) {
        for (j = 0; j + 8 <= IW; j += 8) {
            v0 = vld1_s8(src), v1 = vld1_s8(src + IW);
            r0 = vmull_s8(v0, f0), r1 = vmull_s8(v1, f1);
            s0 = vaddq_s16(r0, r1);
            s16 = vmovn_s32(vpaddlq_s16(s0));
            s = vadd_s16(vld1_s16(dst), s16);
            vst1_s16(dst, s);
            src += 8;
            dst += 4;
        }
        for (; j < IW; j += 2) {
            (*dst++) += static_cast<int16_t>(src[0]) *
                                static_cast<int16_t>(filter[0]) +
                        static_cast<int16_t>(src[1]) *
                                static_cast<int16_t>(filter[1]) +
                        static_cast<int16_t>(src[IW]) *
                                static_cast<int16_t>(filter[2]) +
                        static_cast<int16_t>(src[IW + 1]) *
                                static_cast<int16_t>(filter[3]);
            src += 2;
        }
        src += IW;
    }
}

}  // anonymous namespace

size_t conv_bias::get_workspace_in_bytes_conv_int8x8x16_stride2_flt2(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    return 0;
}

bool conv_bias::can_conv_int8x8x16_stride2_flt2(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    return fm.format == param::ConvBias::Format::NCHW && !fm.should_flip &&
           param.src_type.enumv() == DTypeEnum::Int8 &&
           param.filter_type.enumv() == DTypeEnum::Int8 &&
           param.dst_type.enumv() == DTypeEnum::Int16 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && fm.stride[0] == 2 && fm.stride[1] == 2 &&
           FH == fm.spatial[1] && (FH == 2 || FH == 3 || FH == 5) &&
           param.isz[0] % 2 == 0 && param.isz[1] % 2 == 0 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 && fm.spatial[0] == 2 &&
           fm.spatial[1] == 2 && fm.padding[0] == 0 && fm.padding[1] == 0;
}

void conv_bias::conv_int8x8x16_stride2_flt2(
        const ConvBiasImpl::NCBKernParam& param) {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    megdnn_ignore(FH);
    megdnn_ignore(FW);
    megdnn_ignore(SH);
    megdnn_ignore(SW);
    megdnn_ignore(PH);
    megdnn_ignore(PW);
    auto src = param.src<int8_t>();
    auto dst_init = param.dst<int16_t>();
    auto filter_init = param.filter<int8_t>();
    const uint32_t shape = IH * IW;
    for (uint32_t n = 0; n < N; ++n) {
        auto fptr = filter_init;
        auto dst = dst_init + n * param.out_bs;
        memset(dst, 0, sizeof(dst[0]) * OC * OH * OW);
        for (uint32_t j = 0; j < OC; ++j) {
            for (uint32_t k = 0; k < IC; ++k) {
                conv_2x2_optimize_single_channel(src + k * shape, IH, IW, fptr,
                                                 dst, OH, OW);
                fptr += 4;
            }
            dst += OH * OW;
        }
        src += param.inp_bs;
    }
}

// vim: syntax=cpp.doxygen
