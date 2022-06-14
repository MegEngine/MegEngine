#include "src/fallback/resize/gi/upsample2_nchw.h"

using namespace megdnn;
using namespace fallback;
using namespace resize;

namespace {

template <typename ctype, size_t fh, size_t fw>
static GI_FORCEINLINE ctype
compute_linear_element(const ctype src[4], const ctype alpha[2]) {
    return src[0] * alpha[0 ^ fh] * alpha[0 ^ fw] +
           src[1] * alpha[0 ^ fh] * alpha[1 ^ fw] +
           src[2] * alpha[1 ^ fh] * alpha[0 ^ fw] +
           src[3] * alpha[1 ^ fh] * alpha[1 ^ fw];
}

template <typename simd_helper, size_t fh, size_t fw>
static GI_FORCEINLINE typename simd_helper::simd_type compute_linear_element_simd(
        const typename simd_helper::simd_type_x4 src,
        const typename simd_helper::simd_fixlen_type alpha[2][2]) {
    typename simd_helper::simd_type c = simd_helper::dup(0);
    c = simd_helper::fma(
            c, GiGetSubVectorFloat32V4(src, 0),
            GiFixLenType2GiFloat32Type(alpha[0 ^ fh][0 ^ fw]));
    c = simd_helper::fma(
            c, GiGetSubVectorFloat32V4(src, 1),
            GiFixLenType2GiFloat32Type(alpha[0 ^ fh][1 ^ fw]));
    c = simd_helper::fma(
            c, GiGetSubVectorFloat32V4(src, 2),
            GiFixLenType2GiFloat32Type(alpha[1 ^ fh][0 ^ fw]));
    c = simd_helper::fma(
            c, GiGetSubVectorFloat32V4(src, 3),
            GiFixLenType2GiFloat32Type(alpha[1 ^ fh][1 ^ fw]));
    return c;
}

template <typename ctype, bool has_right, bool has_bottom>
static GI_FORCEINLINE void compute_linear_2x2_element(
        const ctype* src, ctype* dst, size_t IW, size_t OW, const ctype alpha[2]) {
    const ctype* src_ptr[4] = {src, src, src, src};

    if (has_right) {
        src_ptr[1] += 1;
        src_ptr[3] += 1;
    }
    if (has_bottom) {
        src_ptr[2] += IW;
        src_ptr[3] += IW;
    }

    ctype rsrc[4];
    rsrc[0] = *src_ptr[0];
    rsrc[1] = *src_ptr[1];
    rsrc[2] = *src_ptr[2];
    rsrc[3] = *src_ptr[3];

    dst[0] = compute_linear_element<ctype, 0, 0>(rsrc, alpha);
    if (has_right) {
        dst[1] = compute_linear_element<ctype, 0, 1>(rsrc, alpha);
    }
    if (has_bottom) {
        dst[OW] = compute_linear_element<ctype, 1, 0>(rsrc, alpha);
    }
    if (has_right && has_bottom) {
        dst[OW + 1] = compute_linear_element<ctype, 1, 1>(rsrc, alpha);
    }
}

template <typename simd_helper>
static GI_FORCEINLINE void compute_linear_2x2_element_simd(
        const typename simd_helper::ctype* src, typename simd_helper::ctype* dst,
        size_t IW, size_t OW,
        const typename simd_helper::simd_fixlen_type alpha[2][2]) {
    using simd_type_x4 = typename simd_helper::simd_type_x4;
    using simd_type = typename simd_helper::simd_type;

    simd_type_x4 rsrc;
    simd_type tmp;
    tmp = simd_helper::load(src);
    GiSetSubVectorFloat32V4(rsrc, 0, tmp);
    tmp = simd_helper::load(src + 1);
    GiSetSubVectorFloat32V4(rsrc, 1, tmp);
    tmp = simd_helper::load(src + IW);
    GiSetSubVectorFloat32V4(rsrc, 2, tmp);
    tmp = simd_helper::load(src + IW + 1);
    GiSetSubVectorFloat32V4(rsrc, 3, tmp);

    simd_type_x4 rdst;
    tmp = compute_linear_element_simd<simd_helper, 0, 0>(rsrc, alpha);
    GiSetSubVectorFloat32V4(rdst, 0, tmp);
    tmp = compute_linear_element_simd<simd_helper, 0, 1>(rsrc, alpha);
    GiSetSubVectorFloat32V4(rdst, 1, tmp);
    tmp = compute_linear_element_simd<simd_helper, 1, 0>(rsrc, alpha);
    GiSetSubVectorFloat32V4(rdst, 2, tmp);
    tmp = compute_linear_element_simd<simd_helper, 1, 1>(rsrc, alpha);
    GiSetSubVectorFloat32V4(rdst, 3, tmp);

    simd_helper::store2_interleave(
            dst, GiGetSubVectorFloat32V4(rdst, 0), GiGetSubVectorFloat32V4(rdst, 1));
    simd_helper::store2_interleave(
            dst + OW, GiGetSubVectorFloat32V4(rdst, 2),
            GiGetSubVectorFloat32V4(rdst, 3));
}

template <typename ctype>
void linear_upsample2_nchw(
        const ctype* src_ptr, ctype* dst_ptr, size_t N, size_t IH, size_t IW) {
    using simd_helper = SIMDHelper<ctype>;
    size_t OW = IW * 2;
    constexpr size_t PC = simd_helper::simd_width;

    ctype alpha[2] = {0.75, 0.25};

    typename simd_helper::simd_fixlen_type simd_alpha[2][2];
    simd_alpha[0][0] = GiFloat32Type2FixLenType(simd_helper::dup(0.75 * 0.75));
    simd_alpha[0][1] = GiFloat32Type2FixLenType(simd_helper::dup(0.75 * 0.25));
    simd_alpha[1][0] = GiFloat32Type2FixLenType(simd_helper::dup(0.25 * 0.75));
    simd_alpha[1][1] = GiFloat32Type2FixLenType(simd_helper::dup(0.25 * 0.25));

    for (size_t i = 0; i < N; ++i) {
        compute_linear_2x2_element<ctype, false, false>(
                src_ptr, dst_ptr, IW, OW, alpha);
        {
            for (size_t iw = 0; iw + 1 < IW; ++iw) {
                compute_linear_2x2_element<ctype, true, false>(
                        src_ptr + iw, dst_ptr + (iw * 2 + 1), IW, OW, alpha);
            }
        }
        compute_linear_2x2_element<ctype, false, false>(
                src_ptr + (IW - 1), dst_ptr + (OW - 1), IW, OW, alpha);
        dst_ptr += OW;

        for (size_t ih = 0; ih + 1 < IH; ++ih) {
            compute_linear_2x2_element<ctype, false, true>(
                    src_ptr, dst_ptr, IW, OW, alpha);
            size_t iw = 0;
            for (; iw + PC < IW; iw += PC) {
                compute_linear_2x2_element_simd<simd_helper>(
                        src_ptr + iw, dst_ptr + (iw * 2 + 1), IW, OW, simd_alpha);
            }
            for (; iw + 1 < IW; ++iw) {
                compute_linear_2x2_element<ctype, true, true>(
                        src_ptr + iw, dst_ptr + (iw * 2 + 1), IW, OW, alpha);
            }
            compute_linear_2x2_element<ctype, false, true>(
                    src_ptr + (IW - 1), dst_ptr + (OW - 1), IW, OW, alpha);

            src_ptr += IW;
            dst_ptr += 2 * OW;
        }

        compute_linear_2x2_element<ctype, false, false>(
                src_ptr, dst_ptr, IW, OW, alpha);
        {
            for (size_t iw = 0; iw + 1 < IW; ++iw) {
                compute_linear_2x2_element<ctype, true, false>(
                        src_ptr + iw, dst_ptr + (iw * 2 + 1), IW, OW, alpha);
            }
        }
        compute_linear_2x2_element<ctype, false, false>(
                src_ptr + (IW - 1), dst_ptr + (OW - 1), IW, OW, alpha);
        src_ptr += IW;
        dst_ptr += OW;
    }
}

template <typename ctype>
void nearest_upsample2_nchw(
        const ctype* src_ptr, ctype* dst_ptr, size_t N, size_t IH, size_t IW) {
    using simd_helper = SIMDHelper<ctype>;
    size_t OW = IW * 2;
    constexpr size_t PC = simd_helper::simd_width;

    for (size_t i = 0; i < N; ++i) {
        for (size_t ih = 0; ih < IH; ++ih) {
            size_t iw = 0;
            for (; iw + PC - 1 < IW; iw += PC) {
                typename simd_helper::simd_type r0 = simd_helper::load(src_ptr + iw);

                simd_helper::store2_interleave(dst_ptr + (iw * 2), r0, r0);
                simd_helper::store2_interleave(dst_ptr + (OW + iw * 2), r0, r0);
            }
            for (; iw < IW; iw += 1) {
                ctype v = src_ptr[iw];
                dst_ptr[iw * 2] = v;
                dst_ptr[iw * 2 + 1] = v;
                dst_ptr[OW + iw * 2] = v;
                dst_ptr[OW + iw * 2 + 1] = v;
            }
            src_ptr += IW;
            dst_ptr += 2 * OW;
        }
    }
}

}  // namespace

void megdnn::fallback::resize_linear_upsample2_nchw_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    linear_upsample2_nchw(
            kern_param.src(), kern_param.dst(), kern_param.n * kern_param.c,
            kern_param.ih, kern_param.iw);
}

void megdnn::fallback::resize_nearest_upsample2_nchw_gi_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    nearest_upsample2_nchw(
            kern_param.src(), kern_param.dst(), kern_param.n * kern_param.c,
            kern_param.ih, kern_param.iw);
}
