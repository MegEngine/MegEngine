#pragma once
#include "megdnn/dtype.h"

#if MEGDNN_CC_HOST
#include "megdnn/basic_types.h"
#endif

namespace megdnn {
namespace device_reduce {

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct SumOp {
    typedef wtype_ wtype;

    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE SumOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MeanOp {
    typedef wtype_ wtype;

    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) {
        dst[idx] = val / static_cast<wtype>(B);
    }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE MeanOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct SumSqrOp {
    typedef wtype_ wtype;

    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        return static_cast<wtype>(src[idx]) * static_cast<wtype>(src[idx]);
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs + rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE SumSqrOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(0)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct ProdOp {
    typedef wtype_ wtype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs * rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE ProdOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(1)), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MinOp {
    typedef wtype_ wtype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
#if defined(__CUDA_ARCH__)
        return lhs < rhs ? lhs : rhs;
#else
        return std::min(lhs, rhs);
#endif
    }
    MEGDNN_HOST MEGDNN_DEVICE MinOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::max())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype>
struct MinOp<src_ctype, dst_ctype, dt_float32> {
    typedef dt_float32 wtype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
#if defined(__CUDA_ARCH__)
        return (isnan(lhs) || lhs < rhs) ? lhs : rhs;
#else
        return (std::isnan(lhs) || lhs < rhs) ? lhs : rhs;
#endif
    }
    MEGDNN_HOST MEGDNN_DEVICE MinOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::max())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype, typename wtype_>
struct MaxOp {
    typedef wtype_ wtype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
#if defined(__CUDA_ARCH__)
        return lhs > rhs ? lhs : rhs;
#else
        return std::max(lhs, rhs);
#endif
    }
    MEGDNN_HOST MEGDNN_DEVICE MaxOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::min())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename dst_ctype>
struct MaxOp<src_ctype, dst_ctype, dt_float32> {
    typedef dt_float32 wtype;
    const wtype INIT;

    src_ctype* src;
    dst_ctype* dst;
    const size_t B;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) { return src[idx]; }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
#if defined(__CUDA_ARCH__)
        return (isnan(lhs) || lhs > rhs) ? lhs : rhs;
#else
        return (std::isnan(lhs) || lhs > rhs) ? lhs : rhs;
#endif
    }
    MEGDNN_HOST MEGDNN_DEVICE MaxOp(src_ctype* src, dst_ctype* dst, size_t B)
            : INIT(wtype(DTypeTrait<wtype>::min())), src(src), dst(dst), B(B) {}
};

template <typename src_ctype, typename index_ctype, typename dst_ctype, typename wtype_>
struct CheckNonFiniteOp {
    typedef wtype_ wtype;
    const wtype INIT;

    src_ctype** srcs;
    index_ctype* srcs_total_nr_elems;
    dst_ctype* dst;
    const size_t B;
    const src_ctype scale;

    MEGDNN_HOST MEGDNN_DEVICE wtype read(uint32_t idx) {
        size_t x = idx / B;
        size_t y = idx % B;
        if (y < srcs_total_nr_elems[x]) {
#if defined(__CUDA_ARCH__)
            wtype val = isfinite(srcs[x][y]);
#else
            wtype val = std::isfinite(srcs[x][y]);
#endif
            if (val)
                srcs[x][y] *= scale;
            return !val;
        }
        return 0;
    }
    MEGDNN_HOST MEGDNN_DEVICE void write(uint32_t idx, wtype val) { dst[idx] = val; }
    static MEGDNN_HOST MEGDNN_DEVICE wtype apply(wtype lhs, wtype rhs) {
        return lhs | rhs;
    }
    MEGDNN_HOST MEGDNN_DEVICE CheckNonFiniteOp(
            src_ctype** srcs, index_ctype* srcs_total_nr_elems, dst_ctype* dst,
            size_t B, src_ctype scale)
            : INIT(wtype(0)),
              srcs(srcs),
              srcs_total_nr_elems(srcs_total_nr_elems),
              dst(dst),
              B(B),
              scale(scale) {}
};

}  // namespace device_reduce

namespace reduce {
#if MEGDNN_CC_HOST
void get_ABC(const TensorShape& shape, size_t& A, size_t& B, size_t& C, size_t axis);
#endif
}  // namespace reduce

}  // namespace megdnn

// vim: syntax=cpp.doxygen
