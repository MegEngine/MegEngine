#include "src/fallback/resize/gi/direct_nchwxx.h"

using namespace megdnn;
using namespace fallback;
using namespace resize;

namespace {

template <typename ctype, InterpolationMode imode>
void resize_direct_nchwxx(
        const ctype* sptr, ctype* dptr, size_t N, size_t IH, size_t IW, size_t OH,
        size_t OW) {
    using simd_helper = SIMDHelper<ctype>;
    constexpr size_t PC = simd_helper::simd_width;
    using simd_type = typename simd_helper::simd_type;

    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < OH; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                int ih0, ih1, iw0, iw1;
                float ah0, ah1, aw0, aw1;

                std::tie(ah0, ih0, ah1, ih1) =
                        get_nearest_linear_coord(imode, scale_h, IH, oh);
                std::tie(aw0, iw0, aw1, iw1) =
                        get_nearest_linear_coord(imode, scale_w, IW, ow);

                simd_type r0 = simd_helper::load(sptr + (ih0 * IW + iw0) * PC);
                simd_type r1 = simd_helper::load(sptr + (ih0 * IW + iw1) * PC);
                simd_type r2 = simd_helper::load(sptr + (ih1 * IW + iw0) * PC);
                simd_type r3 = simd_helper::load(sptr + (ih1 * IW + iw1) * PC);

                // FIXME: weight fp16 may cause precision problem
                ctype a0 = ah0 * aw0;
                ctype a1 = ah0 * aw1;
                ctype a2 = ah1 * aw0;
                ctype a3 = ah1 * aw1;

                simd_type c = simd_helper::dup(0);
                c = simd_helper::fma(c, r0, a0);
                c = simd_helper::fma(c, r1, a1);
                c = simd_helper::fma(c, r2, a2);
                c = simd_helper::fma(c, r3, a3);

                simd_helper::store(dptr + (oh * OW + ow) * PC, c);
            }
        }
        sptr += IH * IW * PC;
        dptr += OH * OW * PC;
    }
}
}  // namespace

void megdnn::fallback::resize_direct_nearest_nchw44_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    resize_direct_nchwxx<float, InterpolationMode::INTER_NEAREST>(
            kern_param.src(), kern_param.dst(), kern_param.n * kern_param.c / 4,
            kern_param.ih, kern_param.iw, kern_param.oh, kern_param.ow);
}

void megdnn::fallback::resize_direct_linear_nchw44_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    resize_direct_nchwxx<float, InterpolationMode::INTER_LINEAR>(
            kern_param.src(), kern_param.dst(), kern_param.n * kern_param.c / 4,
            kern_param.ih, kern_param.iw, kern_param.oh, kern_param.ow);
}
