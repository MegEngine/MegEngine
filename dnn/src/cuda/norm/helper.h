#pragma once
#include "megdnn/dtype.h"

#if MEGDNN_CC_HOST
#include "megdnn/basic_types.h"
#endif

namespace megdnn {
namespace device_reduce {

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct NormOp;

template <>
struct NormOp<dt_float32, dt_float32, dt_float32> {
    typedef dt_float32 wtype;
    typedef dt_float32 src_ctype;
    typedef dt_float32 dst_ctype;
    typedef wtype p_type;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;
    const p_type p;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        return powf(fabsf(src[idx]), p);
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) {
        dst[idx] = powf(val, 1.f / p);
    }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormOp(src_ctype* src, dst_ctype* dst, size_t B, p_type p)
            : INIT(wtype(0)), src(src), dst(dst), B(B), p(static_cast<wtype>(p)) {}
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct NormOp<dt_float16, dt_float16, dt_float16> {
    typedef dt_float16 wtype;
    typedef dt_float16 src_ctype;
    typedef dt_float16 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;
    const wtype p;

    // HALF_FLOAT API has dispatch host and device.
    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        return half_float::detail::pow(half_float::detail::abs(src[idx]), p);
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) {
        dst[idx] = half_float::detail::pow(val, static_cast<wtype>(1.f) / p);
    }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE
    NormOp(src_ctype* src, dst_ctype* dst, size_t B, dt_float32 p)
            : INIT(wtype(0)), src(src), dst(dst), B(B), p(static_cast<wtype>(p)) {}
};
#endif

// TODO: 0Norm impl need understand reduceop
template <typename src_ctype, typename dst_ctype, typename wtype_>
struct NormZeroOp;

template <>
struct NormZeroOp<dt_float32, dt_float32, dt_float32> {
    typedef dt_float32 wtype;
    typedef dt_float32 src_ctype;
    typedef dt_float32 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;
    const wtype epsilon = 0.00001f;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        return fabsf(src[idx] - 0.0f) <= epsilon ? 0.0f : 1.0f;
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }

    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormZeroOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct NormZeroOp<dt_float16, dt_float16, dt_float16> {
    typedef dt_float16 wtype;
    typedef dt_float16 src_ctype;
    typedef dt_float16 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;
    const wtype epsilon = half_float::half(0.00001f);

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        return half_float::detail::fabs(src[idx] - half_float::half()) <= epsilon
                     ? half_float::half(0.0f)
                     : half_float::half(1.0f);
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }

    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormZeroOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};
#endif

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct NormOneOp;

template <>
struct NormOneOp<dt_float32, dt_float32, dt_float32> {
    typedef dt_float32 wtype;
    typedef dt_float32 src_ctype;
    typedef dt_float32 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return fabsf(src[idx]); }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }

    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormOneOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct NormOneOp<dt_float16, dt_float16, dt_float16> {
    typedef dt_float16 wtype;
    typedef dt_float16 src_ctype;
    typedef dt_float16 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        return half_float::detail::abs(src[idx]);
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }

    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormOneOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};
#endif

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct NormTwoOp;

template <>
struct NormTwoOp<dt_float32, dt_float32, dt_float32> {
    typedef dt_float32 wtype;
    typedef dt_float32 src_ctype;
    typedef dt_float32 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx] * src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) {
        dst[idx] = sqrtf(val);
    }

    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormTwoOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

#if !MEGDNN_DISABLE_FLOAT16
template <>
struct NormTwoOp<dt_float16, dt_float16, dt_float16> {
    typedef dt_float16 wtype;
    typedef dt_float16 src_ctype;
    typedef dt_float16 dst_ctype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx] * src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) {
        dst[idx] = half_float::detail::sqrt(val);
    }

    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE NormTwoOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};
#endif

}  // namespace device_reduce
}  // namespace megdnn
