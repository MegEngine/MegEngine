/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_3x3_s2x2_float16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/pooling/do_max_pooling_3x3_s2x2_float16.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include <vector>
#include <algorithm>
#include <limits>
#include "src/arm_common/simd_macro/marm_neon.h"
#include <cstring>

namespace megdnn {
namespace arm_common {

#define MEGDNN_SIMD_WIDTH 8
void do_max_pooling_3x3_s2x2_float16_NEON(const __fp16* src, __fp16* dst,
                                          size_t IH_, size_t IW_, size_t OH_,
                                          size_t OW_, size_t PH_, size_t PW_,
                                          const WorkspaceBundle& ws) {
    int IH = IH_, IW = IW_, OH = OH_, OW = OW_, PH = PH_, PW = PW_;
    // cache[i] stores the answer of the i-th line after
    // pooling along the W dimension.
    __fp16* cache[3] = {static_cast<__fp16*>(ws.get(0)),
                        static_cast<__fp16*>(ws.get(1)),
                        static_cast<__fp16*>(ws.get(2))};
    __fp16* odd = static_cast<__fp16*>(ws.get(3));
    __fp16* even = static_cast<__fp16*>(ws.get(4));
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
        const __fp16* __restrict sptr = src + ih * IW;
        auto tmp = cache[2];
        cache[2] = cache[1];
        cache[1] = cache[0];
        cache[0] = tmp;
        // cache 0 is used to store the current answer.
        auto run_single = [&](int ow) {
            int iw = ow * 2 - PW;
            __fp16 res = std::numeric_limits<dt_float16>::lowest();
            if (iw + 0 >= 0 && iw + 0 < IW) {
                res = std::max(res, sptr[iw + 0]);
            }
            if (iw + 1 >= 0 && iw + 1 < IW) {
                res = std::max(res, sptr[iw + 1]);
            }
            if (iw + 2 >= 0 && iw + 2 < IW) {
                res = std::max(res, sptr[iw + 2]);
            }
            cache[0][ow] = res;
        };
        // build odd/even
        int iw = 0;
        int odd_offset = 0, even_offset = 0;

        for (; iw + 2 * MEGDNN_SIMD_WIDTH <= IW; iw += 2 * MEGDNN_SIMD_WIDTH) {
            float16x8_t s0, s1;
            s0 = vld1q_f16(sptr + iw + 0);
            s1 = vld1q_f16(sptr + iw + MEGDNN_SIMD_WIDTH);
            float16x8x2_t d = vuzpq_f16(s0, s1);
            vst1q_f16(even + even_offset, d.val[0]);
            vst1q_f16(odd + odd_offset, d.val[1]);
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
                float16x8_t d, s0, s1, s2;
                s0 = vld1q_f16(odd + ow - (PW >> 1) - 1);
                s1 = vld1q_f16(even + ow - (PW >> 1));
                s2 = vld1q_f16(odd + ow - (PW >> 1));
                d = vmaxq_f16(vmaxq_f16(s0, s1), s2);
                vst1q_f16(cache[0] + ow, d);
            }
        } else {
            for (; ow + MEGDNN_SIMD_WIDTH <= OW_to;
                 ow += MEGDNN_SIMD_WIDTH) {
                float16x8_t d, s0, s1, s2;
                s0 = vld1q_f16(even + ow - (PW >> 1));
                s1 = vld1q_f16(odd + ow - (PW >> 1));
                s2 = vld1q_f16(even + ow - (PW >> 1) + 1);
                d = vmaxq_f16(vmaxq_f16(s0, s1), s2);
                vst1q_f16(cache[0] + ow, d);
            }
        }
        for (; ow < OW; ++ow)
            run_single(ow);
    };
    for (int oh = 0; oh < OH; ++oh) {
        __fp16* __restrict dptr = dst + oh * OW;
        int ih_from = std::min(IH, std::max(0, oh * 2 - PH));
        int ih_to = std::min(IH, std::max(0, oh * 2 - PH + 3));
        while (ih_next < ih_to) {
            process_cache(ih_next++);
        }
        if (ih_to - ih_from == 3) {
            int ow = 0;
            for (; ow + MEGDNN_SIMD_WIDTH <= OW; ow += MEGDNN_SIMD_WIDTH) {
                float16x8_t d, s0, s1, s2;
                s0 = vld1q_f16(cache[0] + ow);
                s1 = vld1q_f16(cache[1] + ow);
                s2 = vld1q_f16(cache[2] + ow);
                d = vmaxq_f16(vmaxq_f16(s0, s1), s2);
                vst1q_f16(dptr + ow, d);
            }
            for (; ow < OW; ++ow) {
                dptr[ow] = std::max(std::max(cache[0][ow], cache[1][ow]),
                                    cache[2][ow]);
            }
        } else {
            std::memcpy(dptr, cache[0], sizeof(__fp16) * OW);
            for (int i = 1; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + MEGDNN_SIMD_WIDTH <= OW;
                     ow += MEGDNN_SIMD_WIDTH) {
                    float16x8_t d, s;
                    s = vld1q_f16(cache[i] + ow);
                    d = vld1q_f16(dptr + ow);
                    d = vmaxq_f16(d, s);
                    vst1q_f16(dptr + ow, d);
                }
                for (; ow < OW; ++ow) {
                    dptr[ow] = std::max(dptr[ow], cache[i][ow]);
                }
            }
        }
    }
}

}  // namespace arm_common
}  // namespace megdnn

#endif

// vim: syntax=cpp.doxygen
