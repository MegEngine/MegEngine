#include "src/common/utils.h"

#include <algorithm>
#include <vector>
#include "do_max_pooling_3x3_s2x2_float.h"
#include "src/common/macro_helper.h"

namespace megdnn {
namespace fallback {

#define GI_UZP(s0, s1, d0, d1)              \
    do {                                    \
        auto tmp__ = GiUzpqFloat32(s0, s1); \
        d0 = tmp__.val[0];                  \
        d1 = tmp__.val[1];                  \
    } while (0)

void do_max_pooling_3x3_s2x2_float_gi(
        const float* src, float* dst, size_t IH_, size_t IW_, size_t OH_, size_t OW_,
        size_t PH_, size_t PW_, const WorkspaceBundle& ws) {
    int IH = IH_, IW = IW_, OH = OH_, OW = OW_, PH = PH_, PW = PW_;
    // cache[i] stores the answer of the i-th line after
    // pooling along the W dimension.
    float* cache[3] = {
            static_cast<float*>(ws.get(0)), static_cast<float*>(ws.get(1)),
            static_cast<float*>(ws.get(2))};
    float* odd = static_cast<float*>(ws.get(3));
    float* even = static_cast<float*>(ws.get(4));
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
        const float* __restrict sptr = src + ih * IW;
        auto tmp = cache[2];
        cache[2] = cache[1];
        cache[1] = cache[0];
        cache[0] = tmp;
        // cache 0 is used to store the current answer.
        auto run_single = [&](int ow) {
            int iw = ow * 2 - PW;
            float res = std::numeric_limits<float>::lowest();
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

        for (; iw + 2 * 4 <= IW; iw += 2 * 4) {
            GI_FLOAT32_t s0, s1, d0, d1;
            s0 = GiLoadFloat32(sptr + iw);
            s1 = GiLoadFloat32(sptr + iw + 4);
            GI_UZP(s0, s1, d0, d1);
            GiStoreFloat32(even + even_offset, d0);
            GiStoreFloat32(odd + odd_offset, d1);
            even_offset += 4;
            odd_offset += 4;
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
            for (; ow + 4 <= OW_to; ow += 4) {
                GI_FLOAT32_t d, s0, s1, s2;
                s0 = GiLoadFloat32(odd + ow - (PW >> 1) - 1);
                s1 = GiLoadFloat32(even + ow - (PW >> 1));
                s2 = GiLoadFloat32(odd + ow - (PW >> 1));
                d = GiMaximumFloat32(GiMaximumFloat32(s0, s1), s2);
                GiStoreFloat32(cache[0] + ow, d);
            }
        } else {
            for (; ow + 4 <= OW_to; ow += 4) {
                GI_FLOAT32_t d, s0, s1, s2;
                s0 = GiLoadFloat32(even + ow - (PW >> 1));
                s1 = GiLoadFloat32(odd + ow - (PW >> 1));
                s2 = GiLoadFloat32(even + ow - (PW >> 1) + 1);
                d = GiMaximumFloat32(GiMaximumFloat32(s0, s1), s2);
                GiStoreFloat32(cache[0] + ow, d);
            }
        }
        for (; ow < OW; ++ow)
            run_single(ow);
    };
    for (int oh = 0; oh < OH; ++oh) {
        float* __restrict dptr = dst + oh * OW;
        int ih_from = std::min(IH, std::max(0, oh * 2 - PH));
        int ih_to = std::min(IH, std::max(0, oh * 2 - PH + 3));
        while (ih_next < ih_to) {
            process_cache(ih_next++);
        }
        if (ih_to - ih_from == 3) {
            int ow = 0;
            for (; ow + 4 <= OW; ow += 4) {
                GI_FLOAT32_t d, s0, s1, s2;
                s0 = GiLoadFloat32(cache[0] + ow);
                s1 = GiLoadFloat32(cache[1] + ow);
                s2 = GiLoadFloat32(cache[2] + ow);
                d = GiMaximumFloat32(GiMaximumFloat32(s0, s1), s2);
                GiStoreFloat32(dptr + ow, d);
            }
            for (; ow < OW; ++ow) {
                dptr[ow] = std::max(std::max(cache[0][ow], cache[1][ow]), cache[2][ow]);
            }
        } else {
            std::memcpy(dptr, cache[0], sizeof(float) * OW);
            for (int i = 1; i < ih_to - ih_from; ++i) {
                int ow = 0;
                for (; ow + 4 <= OW; ow += 4) {
                    GI_FLOAT32_t d, s;
                    s = GiLoadFloat32(cache[i] + ow);
                    d = GiLoadFloat32(dptr + ow);
                    d = GiMaximumFloat32(d, s);
                    GiStoreFloat32(dptr + ow, d);
                }
                for (; ow < OW; ++ow) {
                    dptr[ow] = std::max(dptr[ow], cache[i][ow]);
                }
            }
        }
    }
}

}  // namespace fallback
}  // namespace megdnn
