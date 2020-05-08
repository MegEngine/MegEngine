/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_3x3_s2x2_int8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/pooling/do_max_pooling_3x3_s2x2_int8.h"

#include <vector>
#include <algorithm>
#include <limits>
#include "src/arm_common/simd_macro/marm_neon.h"
#include <cstring>

namespace megdnn {
namespace arm_common {

void do_max_pooling_3x3_s2x2_int8_NEON(const int8_t* src, int8_t* dst,
                                       size_t IH_, size_t IW_, size_t OH_,
                                       size_t OW_, size_t PH_, size_t PW_,
                                       const WorkspaceBundle& ws) {
    int IH = IH_, IW = IW_, OH = OH_, OW = OW_, PH = PH_, PW = PW_;
    // cache[i] stores the answer of the i-th line after
    // pooling along the W dimension.
    int8_t* cache[3] = {static_cast<int8_t*>(ws.get(0)),
                        static_cast<int8_t*>(ws.get(1)),
                        static_cast<int8_t*>(ws.get(2))};
    int8_t* odd = static_cast<int8_t*>(ws.get(3));
    int8_t* even = static_cast<int8_t*>(ws.get(4));

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
        const int8_t* __restrict sptr = src + ih * IW;
        auto tmp = cache[2];
        cache[2] = cache[1];
        cache[1] = cache[0];
        cache[0] = tmp;
        // cache 0 is used to store the current answer.
        auto run_single = [&](int ow) {
            int iw = ow * 2 - PW;
            int8_t res = std::numeric_limits<int8_t>::lowest();
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

        for (; iw + 32 <= IW; iw += 32) {
            int8x16_t s0, s1;
            s0 = vld1q_s8(sptr + iw + 0);
            s1 = vld1q_s8(sptr + iw + 16);
            int8x16x2_t d = vuzpq_s8(s0, s1);
            vst1q_s8(even + even_offset, d.val[0]);
            vst1q_s8(odd + odd_offset, d.val[1]);
            even_offset += 16;
            odd_offset += 16;
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
            for (; ow + 16 <= OW_to; ow += 16) {
                int8x16_t d, s0, s1, s2;
                s0 = vld1q_s8(odd + ow - (PW >> 1) - 1);
                s1 = vld1q_s8(even + ow - (PW >> 1));
                s2 = vld1q_s8(odd + ow - (PW >> 1));
                d = vmaxq_s8(vmaxq_s8(s0, s1), s2);
                vst1q_s8(cache[0] + ow, d);
            }
        } else {
            for (; ow + 16 <= OW_to; ow += 16) {
                int8x16_t d, s0, s1, s2;
                s0 = vld1q_s8(even + ow - (PW >> 1));
                s1 = vld1q_s8(odd + ow - (PW >> 1));
                s2 = vld1q_s8(even + ow - (PW >> 1) + 1);
                d = vmaxq_s8(vmaxq_s8(s0, s1), s2);
                vst1q_s8(cache[0] + ow, d);
            }
        }
        for (; ow < OW; ++ow)
            run_single(ow);
    };
    for (int oh = 0; oh < OH; ++oh) {
        int8_t* __restrict dptr = dst + oh * OW;
        int ih_from = std::min(IH, std::max(0, oh * 2 - PH));
        int ih_to = std::min(IH, std::max(0, oh * 2 - PH + 3));
        while (ih_next < ih_to) {
            process_cache(ih_next++);
        }
        if (ih_to - ih_from == 3) {
            int ow = 0;
            for (; ow + 16 <= OW; ow += 16) {
                int8x16_t d, s0, s1, s2;
                s0 = vld1q_s8(cache[0] + ow);
                s1 = vld1q_s8(cache[1] + ow);
                s2 = vld1q_s8(cache[2] + ow);
                d = vmaxq_s8(vmaxq_s8(s0, s1), s2);
                vst1q_s8(dptr + ow, d);
            }
            for (; ow < OW; ++ow) {
                dptr[ow] = std::max(std::max(cache[0][ow], cache[1][ow]),
                                    cache[2][ow]);
            }
        } else {
            std::memcpy(dptr, cache[0], sizeof(int8_t) * OW);
            for (int i = 1; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + 16 <= OW; ow += 16) {
                    int8x16_t d, s;
                    s = vld1q_s8(cache[i] + ow);
                    d = vld1q_s8(dptr + ow);
                    d = vmaxq_s8(d, s);
                    vst1q_s8(dptr + ow, d);
                }
                for (; ow < OW; ++ow) {
                    dptr[ow] = std::max(dptr[ow], cache[i][ow]);
                }
            }
        }
    }
}

void do_max_pooling_3x3_s2x2_uint8_NEON(const uint8_t* src, uint8_t* dst,
                                        size_t IH_, size_t IW_, size_t OH_,
                                        size_t OW_, size_t PH_, size_t PW_,
                                        const WorkspaceBundle& ws) {
    int IH = IH_, IW = IW_, OH = OH_, OW = OW_, PH = PH_, PW = PW_;
    // cache[i] stores the answer of the i-th line after
    // pooling along the W dimension.
    uint8_t* cache[3] = {static_cast<uint8_t*>(ws.get(0)),
                         static_cast<uint8_t*>(ws.get(1)),
                         static_cast<uint8_t*>(ws.get(2))};
    uint8_t* odd = static_cast<uint8_t*>(ws.get(3));
    uint8_t* even = static_cast<uint8_t*>(ws.get(4));

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
        const uint8_t* __restrict sptr = src + ih * IW;
        auto tmp = cache[2];
        cache[2] = cache[1];
        cache[1] = cache[0];
        cache[0] = tmp;
        // cache 0 is used to store the current answer.
        auto run_single = [&](int ow) {
            int iw = ow * 2 - PW;
            uint8_t res = std::numeric_limits<uint8_t>::lowest();
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

        for (; iw + 32 <= IW; iw += 32) {
            uint8x16_t s0, s1;
            s0 = vld1q_u8(sptr + iw + 0);
            s1 = vld1q_u8(sptr + iw + 16);
            uint8x16x2_t d = vuzpq_u8(s0, s1);
            vst1q_u8(even + even_offset, d.val[0]);
            vst1q_u8(odd + odd_offset, d.val[1]);
            even_offset += 16;
            odd_offset += 16;
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
            for (; ow + 16 <= OW_to; ow += 16) {
                uint8x16_t d, s0, s1, s2;
                s0 = vld1q_u8(odd + ow - (PW >> 1) - 1);
                s1 = vld1q_u8(even + ow - (PW >> 1));
                s2 = vld1q_u8(odd + ow - (PW >> 1));
                d = vmaxq_u8(vmaxq_u8(s0, s1), s2);
                vst1q_u8(cache[0] + ow, d);
            }
        } else {
            for (; ow + 16 <= OW_to; ow += 16) {
                uint8x16_t d, s0, s1, s2;
                s0 = vld1q_u8(even + ow - (PW >> 1));
                s1 = vld1q_u8(odd + ow - (PW >> 1));
                s2 = vld1q_u8(even + ow - (PW >> 1) + 1);
                d = vmaxq_u8(vmaxq_u8(s0, s1), s2);
                vst1q_u8(cache[0] + ow, d);
            }
        }
        for (; ow < OW; ++ow)
            run_single(ow);
    };
    for (int oh = 0; oh < OH; ++oh) {
        uint8_t* __restrict dptr = dst + oh * OW;
        int ih_from = std::min(IH, std::max(0, oh * 2 - PH));
        int ih_to = std::min(IH, std::max(0, oh * 2 - PH + 3));
        while (ih_next < ih_to) {
            process_cache(ih_next++);
        }
        if (ih_to - ih_from == 3) {
            int ow = 0;
            for (; ow + 16 <= OW; ow += 16) {
                uint8x16_t d, s0, s1, s2;
                s0 = vld1q_u8(cache[0] + ow);
                s1 = vld1q_u8(cache[1] + ow);
                s2 = vld1q_u8(cache[2] + ow);
                d = vmaxq_u8(vmaxq_u8(s0, s1), s2);
                vst1q_u8(dptr + ow, d);
            }
            for (; ow < OW; ++ow) {
                dptr[ow] = std::max(std::max(cache[0][ow], cache[1][ow]),
                                    cache[2][ow]);
            }
        } else {
            std::memcpy(dptr, cache[0], sizeof(uint8_t) * OW);
            for (int i = 1; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + 16 <= OW; ow += 16) {
                    uint8x16_t d, s;
                    s = vld1q_u8(cache[i] + ow);
                    d = vld1q_u8(dptr + ow);
                    d = vmaxq_u8(d, s);
                    vst1q_u8(dptr + ow, d);
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
   // vim: syntax=cpp.doxygen
