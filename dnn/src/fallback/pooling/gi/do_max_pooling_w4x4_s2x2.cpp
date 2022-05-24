#include "do_max_pooling_w4x4_s2x2.h"
#include "pooling_helper.h"

namespace megdnn {
namespace fallback {

void do_max_pooling_w4x4_s2x2_float_gi(
        const dt_float32* src, dt_float32* dst, DType src_dtype, const int IH,
        const int IW, const int OH, const int OW, const int PH, const int PW) {
    const int window = 4;
    const int stride = 2;
    using Pooler = MaxPooler<16, dt_float32, float, float>;
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
        dt_float32 last_hf_res = -std::numeric_limits<dt_float32>::infinity();
        int ih = -PH + stride * oh, iw = -PW + stride * ow;
        if (-PW + stride * ow + window <= IW) {
            GI_FLOAT32_t i0 = GiLoadFloat32(src + (ih + 0) * IW + iw),
                         i1 = GiLoadFloat32(src + (ih + 1) * IW + iw),
                         i2 = GiLoadFloat32(src + (ih + 2) * IW + iw),
                         i3 = GiLoadFloat32(src + (ih + 3) * IW + iw);
            GI_FLOAT32_t sum0 = GiMaximumFloat32(
                    GiMaximumFloat32(i0, i1), GiMaximumFloat32(i2, i3));
            float32x2_t t =
                    GiPmaxFloat32(GiGetLowFloat32(sum0), GiGetHighFloat32(sum0));
            dst[oh * OW + ow] =
                    std::max(GiGetLaneFloat32(t, 0), GiGetLaneFloat32(t, 1));
            last_hf_res = GiGetLaneFloat32(t, 1);
            ow += 1;
        }
        for (; ow + 1 < OW && -PW + stride * (ow + 1) + window <= IW; ow += 2) {
            iw = -PW + stride * (ow + 1);
            GI_FLOAT32_t i0 = GiLoadFloat32(src + (ih + 0) * IW + iw),
                         i1 = GiLoadFloat32(src + (ih + 1) * IW + iw),
                         i2 = GiLoadFloat32(src + (ih + 2) * IW + iw),
                         i3 = GiLoadFloat32(src + (ih + 3) * IW + iw);
            GI_FLOAT32_t sum0 = GiMaximumFloat32(
                    GiMaximumFloat32(i0, i1), GiMaximumFloat32(i2, i3));
            float32x2_t t =
                    GiPmaxFloat32(GiGetLowFloat32(sum0), GiGetHighFloat32(sum0));
            dst[oh * OW + ow + 0] = std::max(GiGetLaneFloat32(t, 0), last_hf_res);
            dst[oh * OW + ow + 1] =
                    std::max(GiGetLaneFloat32(t, 0), GiGetLaneFloat32(t, 1));
            last_hf_res = GiGetLaneFloat32(t, 1);
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

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
