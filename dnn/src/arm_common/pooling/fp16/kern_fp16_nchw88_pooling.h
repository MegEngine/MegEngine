#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#pragma once
#include <arm_neon.h>
#include <limits>
#include "src/arm_common/pooling/opr_impl.h"
#include "src/common/unroll_macro.h"

namespace megdnn {
namespace arm_common {
namespace {
#if MEGDNN_AARCH64
#define OW_STEP 4
#else
#define OW_STEP 2
#endif

template <int filter, int stride, PoolingBase::Mode mode, typename T1, typename T2>
struct CalXsXNchw88 {
    static void impl(T1 result, T2 src);
};

#define CAL_MAX_CB(step, ow_step) \
    result[ow_step] = vmaxq_f16(result[ow_step], src[ow_step * stride + step]);

#define CAL_AVE_CB(step, ow_step) \
    result[ow_step] = vaddq_f16(result[ow_step], src[ow_step * stride + step]);

#define INSTANCE_CAL(filter, ow_step)                                         \
    template <int stride, typename T1, typename T2>                           \
    struct CalXsXNchw88<filter, stride, PoolingBase::Mode::MAX, T1, T2> {     \
        static void impl(T1 result, T2 src) {                                 \
            UNROLL_CALL_NOWRAPPER_D2(filter, ow_step, CAL_MAX_CB);            \
        }                                                                     \
    };                                                                        \
    template <int stride, typename T1, typename T2>                           \
    struct CalXsXNchw88<filter, stride, PoolingBase::Mode::AVERAGE, T1, T2> { \
        static void impl(T1 result, T2 src) {                                 \
            UNROLL_CALL_NOWRAPPER_D2(filter, ow_step, CAL_AVE_CB);            \
        }                                                                     \
    };

INSTANCE_CAL(2, OW_STEP)
INSTANCE_CAL(3, OW_STEP)
INSTANCE_CAL(4, OW_STEP)
INSTANCE_CAL(5, OW_STEP)
INSTANCE_CAL(9, OW_STEP)
INSTANCE_CAL(13, OW_STEP)

#undef INSTANCE_CAL
#undef CAL_AVE_CB
#undef CAL_MAX_CB

template <int filter, int stride, PoolingBase::Mode mode, typename T1, typename T2>
void calculate_xsx_nchw88(T1 result, T2 src) {
    CalXsXNchw88<filter, stride, mode, T1, T2>::impl(result, src);
}

template <int filter, int stride, PoolingBase::Mode mode>
struct KerPoolingFilterXStrideXNchw88 {
    static void impl(const __fp16* src_ptr, __fp16* dst_ptr, size_t iw);
};

template <int filter, int stride>
struct KerPoolingFilterXStrideXNchw88<filter, stride, PoolingBase::Mode::MAX> {
    static void impl(const __fp16* src_ptr, __fp16* dst_ptr, size_t iw) {
        constexpr int src_reg_size = stride * (OW_STEP - 1) + filter;
        constexpr int packed_ic = 8;
        constexpr int simd_len = 8;
        constexpr dt_float16 min_float16 = std::numeric_limits<dt_float16>::lowest();

        float16x8_t result[OW_STEP], src[src_reg_size];
#define cb(i) result[i] = vdupq_n_f16(min_float16);
        UNROLL_CALL_NOWRAPPER(OW_STEP, cb);
#undef cb

        for (int fh_idx = 0; fh_idx < filter; ++fh_idx) {
            auto src_base_ptr = src_ptr + fh_idx * iw * packed_ic;
            rep(i, src_reg_size) { src[i] = vld1q_f16(src_base_ptr + i * simd_len); }
            calculate_xsx_nchw88<filter, stride, PoolingBase::Mode::MAX>(result, src);
        }

#define cb(i) vst1q_f16(dst_ptr + i * packed_ic, result[i]);
        UNROLL_CALL_NOWRAPPER(OW_STEP, cb)
#undef cb
    }
};

template <int filter, int stride>
struct KerPoolingFilterXStrideXNchw88<filter, stride, PoolingBase::Mode::AVERAGE> {
    static void impl(const __fp16* src_ptr, __fp16* dst_ptr, size_t iw) {
        constexpr int src_reg_size = stride * (OW_STEP - 1) + filter;
        constexpr int packed_ic = 8;
        constexpr int simd_len = 8;
        const __fp16 zero = static_cast<__fp16>(0);
        const __fp16 div_filter_pow = static_cast<__fp16>(1.0 / (filter * filter));
        const float16x8_t div_filter_pow_vec = vdupq_n_f16(div_filter_pow);

        float16x8_t result[OW_STEP], src[src_reg_size];
#define cb(i) result[i] = vdupq_n_f16(zero);
        UNROLL_CALL_NOWRAPPER(OW_STEP, cb)
#undef cb
        rep(fh, filter) {
            auto src_base_ptr = src_ptr + fh * iw * packed_ic;
            rep(i, src_reg_size) { src[i] = vld1q_f16(src_base_ptr + i * simd_len); }
            calculate_xsx_nchw88<filter, stride, PoolingBase::Mode::AVERAGE>(
                    result, src);
        }
#define cb(i) \
    vst1q_f16(dst_ptr + i * simd_len, vmulq_f16(result[i], div_filter_pow_vec));
        UNROLL_CALL_NOWRAPPER(OW_STEP, cb)
#undef cb
    }
};

template <PoolingBase::Mode mode>
void kern_pooling_nchw88_remain_pad(
        const __fp16* src, __fp16* dst, const int iw, const int pad_top,
        const int pad_left, const int pad_bottom, const int pad_right,
        const int filter);

template <>
void kern_pooling_nchw88_remain_pad<PoolingBase::Mode::MAX>(
        const __fp16* src, __fp16* dst, const int iw, const int pad_top,
        const int pad_left, const int pad_bottom, const int pad_right,
        const int filter) {
    constexpr int ic_step = 8;
    const int fh_end = filter - pad_bottom;
    const int fw_end = filter - pad_right;
    float16x8_t result = vdupq_n_f16(std::numeric_limits<dt_float16>::lowest());
    for (int fh_idx = pad_top; fh_idx < fh_end; ++fh_idx) {
        for (int fw_idx = pad_left; fw_idx < fw_end; ++fw_idx) {
            float16x8_t s = vld1q_f16(src + (fw_idx - pad_left) * ic_step);
            result = vmaxq_f16(result, s);
        }
        src += iw * ic_step;
    }
    vst1q_f16(dst, result);
}

template <>
void kern_pooling_nchw88_remain_pad<PoolingBase::Mode::AVERAGE>(
        const __fp16* src, __fp16* dst, const int iw, const int pad_top,
        const int pad_left, const int pad_bottom, const int pad_right,
        const int filter) {
    constexpr int ic_step = 8;
    const int fh_end = filter - pad_bottom;
    const int fw_end = filter - pad_right;
    float16x8_t result = vdupq_n_f16(static_cast<dt_float16>(0));
    float16x8_t div_filter_pow_vec = vdupq_n_f16(1.0 / (filter * filter));
    for (int fh_idx = pad_top; fh_idx < fh_end; ++fh_idx) {
        for (int fw_idx = pad_left; fw_idx < fw_end; ++fw_idx) {
            float16x8_t s = vld1q_f16(src + (fw_idx - pad_left) * ic_step);
            result = vaddq_f16(result, s);
        }
        src += iw * ic_step;
    }
    vst1q_f16(dst, vmulq_f16(result, div_filter_pow_vec));
}

template <PoolingBase::Mode mode>
static inline void kern_pooling_with_pad_nchw88(
        const __fp16* src, __fp16* dst, const int filter, const int ow_start,
        const int ow_end, const int iw, const int ow, const int stride_w, const int pw,
        const int real_ih_idx, const int oh_idx, const int pad_top,
        const int pad_bottom) {
    constexpr int ic_step = 8;
    constexpr int oc_step = 8;
    for (int ow_idx = ow_start; ow_idx < ow_end; ++ow_idx) {
        const int iw_idx = ow_idx * stride_w;
        const int real_iw_idx = std::max(0, iw_idx - pw);
        const int pad_left = std::max(0, pw - iw_idx);
        const int pad_right = std::max(0, iw_idx - pw + filter - iw);
        const int src_offset = (real_ih_idx * iw + real_iw_idx) * ic_step;
        const int dst_offset = (oh_idx * ow + ow_idx) * oc_step;
        kern_pooling_nchw88_remain_pad<mode>(
                src + src_offset, dst + dst_offset, iw, pad_top, pad_left, pad_bottom,
                pad_right, filter);
    }
}

template <int filter, int stride, PoolingBase::Mode mode>
static inline void pooling_fp16_nchw88_pad(
        const __fp16* src, __fp16* dst, int ih, int iw, int oh, int ow, int ph,
        int pw) {
    constexpr int stride_h = stride;
    constexpr int stride_w = stride;
    constexpr int ic_step = 8;
    constexpr int oc_step = 8;
    constexpr int ow_step = OW_STEP;
    const int ow_pad_left_end = div_ceil(pw, stride_w);
    const int ow_pad_right_start = (iw + pw - filter) / stride_w + 1;  //!!!! CHECK
    const int ow_pad_right_step_end =
            (ow_pad_right_start - ow_pad_left_end) / ow_step * ow_step +
            ow_pad_left_end;

    rep(oh_idx, oh) {
        const int ih_idx = oh_idx * stride_h;
        const int real_ih_idx = std::max(ih_idx - ph, 0);
        const int pad_top = std::max(0, ph - ih_idx);
        const int pad_bottom = std::max(0, ih_idx - ph + filter - ih);
        if (pad_top > 0 || pad_bottom > 0) {
            kern_pooling_with_pad_nchw88<mode>(
                    src, dst, filter, 0, ow, iw, ow, stride_w, pw, real_ih_idx, oh_idx,
                    pad_top, pad_bottom);
        } else {
            kern_pooling_with_pad_nchw88<mode>(
                    src, dst, filter, 0, ow_pad_left_end, iw, ow, stride_w, pw,
                    real_ih_idx, oh_idx, pad_bottom, pad_bottom);
            for (int ow_idx = ow_pad_left_end; ow_idx < ow_pad_right_step_end;
                 ow_idx += ow_step) {
                const int iw_idx = ow_idx * stride_w;
                const int real_iw_idx = std::max(0, iw_idx - pw);
                const int src_offset = (real_ih_idx * iw + real_iw_idx) * ic_step;
                const int dst_offset = (oh_idx * ow + ow_idx) * oc_step;
                KerPoolingFilterXStrideXNchw88<filter, stride, mode>::impl(
                        src + src_offset, dst + dst_offset, iw);
            }
            kern_pooling_with_pad_nchw88<mode>(
                    src, dst, filter, ow_pad_right_step_end, ow, iw, ow, stride_w, pw,
                    real_ih_idx, oh_idx, pad_top, pad_bottom);
        }
    }
}

template <int filter, int stride, PoolingBase::Mode mode>
static inline void pooling_fp16_nchw88_no_pad(
        const __fp16* src, __fp16* dst, const int iw, const int oh, const int ow) {
    constexpr int stride_h = stride;
    constexpr int stride_w = stride;
    constexpr int ic_step = 8;
    constexpr int oc_step = 8;
    constexpr int ow_step = OW_STEP;
    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;

    rep(oh_idx, oh) {
        const int ih_idx = oh_idx * stride_h;
        for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
            const int iw_idx = ow_idx * stride_w;
            const int src_offset = (ih_idx * iw + iw_idx) * ic_step;
            const int dst_offset = (oh_idx * ow + ow_idx) * oc_step;
            KerPoolingFilterXStrideXNchw88<filter, stride, mode>::impl(
                    src + src_offset, dst + dst_offset, iw);
        }

        if (ow_remain > 0) {
            kern_pooling_with_pad_nchw88<mode>(
                    src, dst, filter, ow_end, ow, iw, ow, stride_w, 0, ih_idx, oh_idx,
                    0, 0);
        }
    }
}

template <int filter, int stride, PoolingBase::Mode mode>
static inline void pooling_fp16_nchw88(
        const __fp16* src, __fp16* dst, const int ih, const int iw, const int oh,
        const int ow, const int ph, const int pw) {
    if (ph > 0 || pw > 0) {
        pooling_fp16_nchw88_pad<filter, stride, mode>(src, dst, ih, iw, oh, ow, ph, pw);
    } else {
        pooling_fp16_nchw88_no_pad<filter, stride, mode>(src, dst, iw, oh, ow);
    }
}

#undef OW_STEP
}  // namespace
}  // namespace arm_common
}  // namespace megdnn
#endif