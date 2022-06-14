#pragma once
#include <limits>
#include "megdnn/opr_param_defs.h"
#include "src/common/unroll_macro.h"
#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/gi_intrinsic_helper.h"

namespace megdnn {
namespace fallback {
namespace {

template <
        int filter, int stride, int ow_step, PoolingBase::Mode mode, typename T1,
        typename T2>
struct CalXsXNchw44 {
    static void impl(T1 result, T2 src);
};

struct GiD1Qf32 {
    static GI_FORCEINLINE GI_FLOAT32_t impl(const float32_t* ptr) {
        return GiLoadFloat32(ptr);
    }
};

template <
        int filter, int stride, int ow_step, PoolingBase::Mode mode, typename T1,
        typename T2>
void calculate_xsx_nchw44(T1 result, T2 src) {
    CalXsXNchw44<filter, stride, ow_step, mode, T1, T2>::impl(result, src);
};

#define CALCULATE_MAX_CB(step)                                    \
    result[0] = GiFloat32Type2FixLenType(GiMaximumFloat32(        \
            GiFixLenType2GiFloat32Type(result[0]),                \
            GiFixLenType2GiFloat32Type(src[0 * stride + step]))); \
    result[1] = GiFloat32Type2FixLenType(GiMaximumFloat32(        \
            GiFixLenType2GiFloat32Type(result[1]),                \
            GiFixLenType2GiFloat32Type(src[1 * stride + step]))); \
    result[2] = GiFloat32Type2FixLenType(GiMaximumFloat32(        \
            GiFixLenType2GiFloat32Type(result[2]),                \
            GiFixLenType2GiFloat32Type(src[2 * stride + step]))); \
    result[3] = GiFloat32Type2FixLenType(GiMaximumFloat32(        \
            GiFixLenType2GiFloat32Type(result[3]),                \
            GiFixLenType2GiFloat32Type(src[3 * stride + step])));

#define CALCULATE_AVG_CB(step)                                    \
    result[0] = GiFloat32Type2FixLenType(GiAddFloat32(            \
            GiFixLenType2GiFloat32Type(result[0]),                \
            GiFixLenType2GiFloat32Type(src[0 * stride + step]))); \
    result[1] = GiFloat32Type2FixLenType(GiAddFloat32(            \
            GiFixLenType2GiFloat32Type(result[1]),                \
            GiFixLenType2GiFloat32Type(src[1 * stride + step]))); \
    result[2] = GiFloat32Type2FixLenType(GiAddFloat32(            \
            GiFixLenType2GiFloat32Type(result[2]),                \
            GiFixLenType2GiFloat32Type(src[2 * stride + step]))); \
    result[3] = GiFloat32Type2FixLenType(GiAddFloat32(            \
            GiFixLenType2GiFloat32Type(result[3]),                \
            GiFixLenType2GiFloat32Type(src[3 * stride + step])));

#define INSTANCE_CAL(filter)                                                     \
    template <int stride, typename T1, typename T2>                              \
    struct CalXsXNchw44<filter, stride, 4, PoolingBase::Mode::MAX, T1, T2> {     \
        static void impl(T1 result, T2 src) {                                    \
            UNROLL_CALL_RAW(filter, CALCULATE_MAX_CB);                           \
        }                                                                        \
    };                                                                           \
    template <int stride, typename T1, typename T2>                              \
    struct CalXsXNchw44<filter, stride, 4, PoolingBase::Mode::AVERAGE, T1, T2> { \
        static void impl(T1 result, T2 src) {                                    \
            UNROLL_CALL_RAW(filter, CALCULATE_AVG_CB);                           \
        }                                                                        \
    };

INSTANCE_CAL(2)
INSTANCE_CAL(3)
INSTANCE_CAL(4)
INSTANCE_CAL(5)
INSTANCE_CAL(9)
INSTANCE_CAL(13)

#undef INSTANCE_CAL
#undef CALCULATE_AVG_CB
#undef CALCULATE_MAX_CB

template <int filter, int stride, int ow_step, PoolingBase::Mode mode>
struct KerPoolingFilterXStrideXNchw44 {
    static void impl(const float32_t* src_ptr, float32_t* dst_ptr, size_t iw);
};

template <int filter, int stride, int ow_step>
struct KerPoolingFilterXStrideXNchw44<filter, stride, ow_step, PoolingBase::Mode::MAX> {
    static void impl(const float32_t* src_ptr, float32_t* dst_ptr, size_t iw) {
        constexpr int src_reg_size = ow_step * stride + filter - stride;
        constexpr int packed_ic = 4;
        constexpr int simd_len = 4;
        constexpr float default_float = std::numeric_limits<float>::lowest();
        GI_FLOAT32_FIXLEN_t result[ow_step];
        GI_FLOAT32_FIXLEN_t src[src_reg_size];

        result[0] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));
        result[1] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));
        result[2] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));
        result[3] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));

        for (int fh_idx = 0; fh_idx < filter; ++fh_idx) {
            load_helper<src_reg_size, 0, simd_len, 0, GiD1Qf32>(
                    src, src_ptr + fh_idx * iw * packed_ic, 0);
            calculate_xsx_nchw44<filter, stride, ow_step, PoolingBase::Mode::MAX>(
                    result, src);
        }

        GiStoreFloat32(dst_ptr + 0 * packed_ic, GiFixLenType2GiFloat32Type(result[0]));
        GiStoreFloat32(dst_ptr + 1 * packed_ic, GiFixLenType2GiFloat32Type(result[1]));
        GiStoreFloat32(dst_ptr + 2 * packed_ic, GiFixLenType2GiFloat32Type(result[2]));
        GiStoreFloat32(dst_ptr + 3 * packed_ic, GiFixLenType2GiFloat32Type(result[3]));
    }
};

template <int filter, int stride, int ow_step>
struct KerPoolingFilterXStrideXNchw44<
        filter, stride, ow_step, PoolingBase::Mode::AVERAGE> {
    static void impl(const float32_t* src_ptr, float32_t* dst_ptr, size_t iw) {
        constexpr int src_reg_size = ow_step * stride + filter - stride;
        constexpr int packed_ic = 4;
        constexpr int simd_len = 4;
        constexpr float default_float = 0;
        constexpr float div_filter_size = 1.f / (filter * filter);
        const GI_FLOAT32_t div_filter_size_vec = GiBroadcastFloat32(div_filter_size);
        GI_FLOAT32_FIXLEN_t result[ow_step];
        GI_FLOAT32_FIXLEN_t src[src_reg_size];

        result[0] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));
        result[1] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));
        result[2] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));
        result[3] = GiFloat32Type2FixLenType(GiBroadcastFloat32(default_float));

        for (int fh_idx = 0; fh_idx < filter; ++fh_idx) {
            load_helper<src_reg_size, 0, simd_len, 0, GiD1Qf32>(
                    src, src_ptr + fh_idx * iw * packed_ic, 0);
            calculate_xsx_nchw44<filter, stride, ow_step, PoolingBase::Mode::AVERAGE>(
                    result, src);
        };
        GiStoreFloat32(
                dst_ptr + 0 * packed_ic,
                GiMultiplyFloat32(
                        GiFixLenType2GiFloat32Type(result[0]), div_filter_size_vec));
        GiStoreFloat32(
                dst_ptr + 1 * packed_ic,
                GiMultiplyFloat32(
                        GiFixLenType2GiFloat32Type(result[1]), div_filter_size_vec));
        GiStoreFloat32(
                dst_ptr + 2 * packed_ic,
                GiMultiplyFloat32(
                        GiFixLenType2GiFloat32Type(result[2]), div_filter_size_vec));
        GiStoreFloat32(
                dst_ptr + 3 * packed_ic,
                GiMultiplyFloat32(
                        GiFixLenType2GiFloat32Type(result[3]), div_filter_size_vec));
    }
};

template <PoolingBase::Mode mode>
void ker_pooling_nchw44_remain_pad(
        const float32_t* src_ptr, float32_t* dst_ptr, const int iw, const int pad_top,
        const int pad_bottom, const int pad_left, const int pad_right,
        const int filter);
template <>
void ker_pooling_nchw44_remain_pad<PoolingBase::Mode::MAX>(
        const float32_t* src_ptr, float32_t* dst_ptr, const int iw, const int pad_top,
        const int pad_bottom, const int pad_left, const int pad_right,
        const int filter) {
    constexpr int ic_step = 4;
    const int ih_end = filter - pad_bottom;
    const int iw_end = filter - pad_right;
    GI_FLOAT32_t result = GiBroadcastFloat32(std::numeric_limits<float>::lowest());
    for (int ih_idx = pad_top; ih_idx < ih_end; ++ih_idx) {
        for (int iw_idx = pad_left; iw_idx < iw_end; ++iw_idx) {
            GI_FLOAT32_t src = GiLoadFloat32(src_ptr + (iw_idx - pad_left) * ic_step);
            result = GiMaximumFloat32(result, src);
        }
        src_ptr += iw * ic_step;
    }
    GiStoreFloat32(dst_ptr, result);
}

template <>
void ker_pooling_nchw44_remain_pad<PoolingBase::Mode::AVERAGE>(
        const float32_t* src_ptr, float32_t* dst_ptr, const int iw, const int pad_top,
        const int pad_bottom, const int pad_left, const int pad_right,
        const int filter) {
    constexpr int ic_step = 4;
    const int ih_end = filter - pad_bottom;
    const int iw_end = filter - pad_right;
    const float div_filter_size = 1.f / (filter * filter);
    const GI_FLOAT32_t div_filter_size_vec = GiBroadcastFloat32(div_filter_size);
    GI_FLOAT32_t result = GiBroadcastFloat32(0.f);

    for (int ih_idx = pad_top; ih_idx < ih_end; ++ih_idx) {
        for (int iw_idx = pad_left; iw_idx < iw_end; ++iw_idx) {
            GI_FLOAT32_t src = GiLoadFloat32(src_ptr + (iw_idx - pad_left) * ic_step);
            result = GiAddFloat32(result, src);
        }
        src_ptr += iw * ic_step;
    }
    result = GiMultiplyFloat32(result, div_filter_size_vec);
    GiStoreFloat32(dst_ptr, result);
}

template <PoolingBase::Mode mode>
static inline void kern_pooling_with_pad_nchw44(
        const float32_t* src, float32_t* dst, const int filter, const int ow_start,
        const int ow_end, const int iw, const int ow, const int stride_w, const int pw,
        const int real_ih_idx, const int oh_idx, const int pad_top,
        const int pad_bottom) {
    constexpr int ic_step = 4;
    constexpr int oc_step = 4;
    for (int ow_idx = ow_start; ow_idx < ow_end; ++ow_idx) {
        const int iw_idx = ow_idx * stride_w;
        const int real_iw_idx = std::max(iw_idx - pw, 0);
        const int pad_left = std::max(0, pw - iw_idx);
        const int pad_right = std::max(0, iw_idx - pw + filter - iw);
        const int src_offset = (real_ih_idx * iw + real_iw_idx) * ic_step;
        const int dst_offset = (oh_idx * ow + ow_idx) * oc_step;
        ker_pooling_nchw44_remain_pad<mode>(
                src + src_offset, dst + dst_offset, iw, pad_top, pad_bottom, pad_left,
                pad_right, filter);
    }
}

template <int filter, int stride, PoolingBase::Mode mode>
static inline void pooling_fp32_nchw44_pad(
        const float32_t* src, float32_t* dst, int ih, int iw, int oh, int ow, int ph,
        int pw) {
    constexpr int stride_h = stride;
    constexpr int stride_w = stride;
    constexpr int ic_step = 4;
    constexpr int oc_step = 4;
    constexpr int ow_step = 4;
    const int ow_pad_left_end = div_ceil(pw, stride_w);
    const int ow_pad_right_end = (iw - filter + pw - 1) / stride_w;
    const int ow_pad_right_step_end =
            (ow_pad_right_end - ow_pad_left_end) / ow_step * ow_step + ow_pad_left_end;

    rep(oh_idx, oh) {
        const int ih_idx = oh_idx * stride_h;
        const int real_ih_idx = std::max(ih_idx - ph, 0);
        const int pad_top = std::max(0, ph - ih_idx);
        const int pad_bottom = std::max(0, ih_idx - ph + filter - ih);
        if (pad_top > 0 || pad_bottom > 0) {
            kern_pooling_with_pad_nchw44<mode>(
                    src, dst, filter, 0, ow, iw, ow, stride_w, pw, real_ih_idx, oh_idx,
                    pad_top, pad_bottom);

        } else {
            kern_pooling_with_pad_nchw44<mode>(
                    src, dst, filter, 0, ow_pad_left_end, iw, ow, stride_w, pw,
                    real_ih_idx, oh_idx, pad_top, pad_bottom);
            for (int ow_idx = ow_pad_left_end; ow_idx < ow_pad_right_step_end;
                 ow_idx += ow_step) {
                const int iw_idx = ow_idx * stride_w;
                const int real_iw_idx = std::max(iw_idx - pw, 0);
                const int src_offset = (real_ih_idx * iw + real_iw_idx) * ic_step;
                const int dst_offset = (oh_idx * ow + ow_idx) * oc_step;
                KerPoolingFilterXStrideXNchw44<filter, stride, ow_step, mode>::impl(
                        src + src_offset, dst + dst_offset, iw);
            }
            kern_pooling_with_pad_nchw44<mode>(
                    src, dst, filter, ow_pad_right_step_end, ow, iw, ow, stride_w, pw,
                    real_ih_idx, oh_idx, pad_top, pad_bottom);
        }
    }
}

template <int filter, int stride, PoolingBase::Mode mode>
static inline void pooling_fp32_nchw44_no_pad(
        const float32_t* src, float32_t* dst, int, int iw, int oh, int ow) {
    constexpr int stride_h = stride;
    constexpr int stride_w = stride;
    constexpr int ic_step = 4;
    constexpr int oc_step = 4;
    constexpr int ow_step = 4;
    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;

    rep(oh_idx, oh) {
        const int ih_idx = oh_idx * stride_h;
        const int src_ih_offset = ih_idx * iw;
        const int dst_oh_offset = oh_idx * ow;
        for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
            const int iw_idx = ow_idx * stride_w;
            const int src_offset = (src_ih_offset + iw_idx) * ic_step;
            const int dst_offset = (dst_oh_offset + ow_idx) * oc_step;
            KerPoolingFilterXStrideXNchw44<filter, stride, ow_step, mode>::impl(
                    src + src_offset, dst + dst_offset, iw);
        }
        if (ow_remain > 0) {
            kern_pooling_with_pad_nchw44<mode>(
                    src, dst, filter, ow_end, ow, iw, ow, stride_w, 0, ih_idx, oh_idx,
                    0, 0);
        }
    }
}

template <int filter, int stride, PoolingBase::Mode mode>
static inline void pooling_fp32_nchw44(
        const float32_t* src, float32_t* dst, int ih, int iw, int oh, int ow, int ph,
        int pw) {
    if (ph > 0 || pw > 0) {
        pooling_fp32_nchw44_pad<filter, stride, mode>(src, dst, ih, iw, oh, ow, ph, pw);
    } else {
        pooling_fp32_nchw44_no_pad<filter, stride, mode>(src, dst, ih, iw, oh, ow);
    }
}

}  // namespace
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
