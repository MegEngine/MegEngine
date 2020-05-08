/**
 * \file dnn/src/common/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/arch.h"
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "megdnn/handle.h"
#include "megdnn/thin/small_vector.h"

#include "src/common/hash_ct.h"
#include "src/common/utils.cuh"

#include <cmath>
#include <cstdarg>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#endif

#if __cplusplus >= 201703L || __clang_major__ >= 4
    #define MEGDNN_FALLTHRU [[fallthrough]];
#elif __GNUC__ >= 7
    #define MEGDNN_FALLTHRU __attribute__ ((fallthrough));
#else
    #define MEGDNN_FALLTHRU
#endif

#define rep(i, n) for (auto i = decltype(n){0}; i < (n); ++i)
#define rep_step(i, n, step) for (auto i = decltype(n){0}; i < (n); i += (step))

#define megdnn_assert_contiguous(layout)                              \
    do {                                                              \
        megdnn_assert((layout).is_contiguous(), "%s is %s.", #layout, \
                      (layout).to_string().c_str());                  \
    } while (0)

#define megdnn_assert_non_overlapping_strong(layout)                     \
    do {                                                                 \
        megdnn_assert((layout).is_non_overlapping_strong(), "%s is %s.", \
                      #layout, (layout).to_string().c_str());            \
    } while (0)

#define megdnn_assert_eq_size_t(lhs_, rhs_)                                   \
    do {                                                                      \
        size_t lhs = lhs_, rhs = rhs_;                                        \
        megdnn_assert(lhs == rhs, "%s is %zu, %s is %zu.", #lhs_, lhs, #rhs_, \
                      rhs);                                                   \
    } while (0)

#define megdnn_assert_eq_layout(lhs, rhs)                                      \
    do {                                                                       \
        megdnn_assert(lhs.eq_layout(rhs), "%s is %s, %s is %s.", #lhs,         \
                      lhs.to_string().c_str(), #rhs, rhs.to_string().c_str()); \
    } while (0)

#define megdnn_assert_eq_shape(lhs, rhs)                                       \
    do {                                                                       \
        megdnn_assert(lhs.eq_shape(rhs), "%s is %s, %s is %s.", #lhs,          \
                      lhs.to_string().c_str(), #rhs, rhs.to_string().c_str()); \
    } while (0)

#define megdnn_assert_eq_dtype(lhs, rhs)                                   \
    do {                                                                   \
        megdnn_assert(lhs.dtype == rhs.dtype, "%s is %s, %s is %s.", #lhs, \
                      lhs.dtype.name(), #rhs, rhs.dtype.name());           \
    } while (0)

#define megdnn_layout_msg(layout) \
    std::string(megdnn_mangle(#layout "=" + (layout).to_string()))

#define MEGDNN_LOCK_GUARD(var) \
    std::lock_guard<std::remove_cv_t<decltype(var)>> _lock_guard_##var { var }

namespace megdnn {

/* ================ logging ================  */
#define megdnn_log_debug(fmt...) \
    _megdnn_do_log(::megdnn::LogLevel::DEBUG, __FILE__, __func__, __LINE__, fmt)
#define megdnn_log(fmt...) \
    _megdnn_do_log(::megdnn::LogLevel::INFO, __FILE__, __func__, __LINE__, fmt)
#define megdnn_log_warn(fmt...) \
    _megdnn_do_log(::megdnn::LogLevel::WARN, __FILE__, __func__, __LINE__, fmt)
#define megdnn_log_error(fmt...) \
    _megdnn_do_log(::megdnn::LogLevel::ERROR, __FILE__, __func__, __LINE__, fmt)

#if MEGDNN_ENABLE_LOGGING
void __log__(LogLevel level, const char* file, const char* func, int line,
             const char* fmt, ...) __attribute__((format(printf, 5, 6)));

#define _megdnn_do_log ::megdnn::__log__
#else
#define _megdnn_do_log(...) \
    do {                    \
    } while (0)
#endif  // megdnn_ENABLE_LOGGING

/* helper functions */
/**
 * \brief Get the next `stride' index lexicographically.
 *
 * stride must be divisible by the last dimension shape.
 * \return true if index is updated successfully, false otherwise (index is
 * already the last one, next index does not exist)
 */
bool get_next_addr(size_t* index, const size_t* shape, size_t n,
                   size_t stride = 1);
size_t get_linear_addr(size_t* index, const size_t* shape, size_t n);
int get_linear_addr_noncont(size_t* index, const TensorLayout& layout);
size_t infer_conv_shape(size_t inp, size_t flt, size_t stride, size_t pad,
                        bool is_floor = true);
void infer_conv_shape2d(size_t ih, size_t iw, size_t fh, size_t fw, size_t sh,
                        size_t sw, size_t ph, size_t pw, size_t& oh, size_t& ow,
                        bool is_floor = true);
template <typename T, typename S, typename Func>
SmallVector<T> apply_vector(Func&& func, const SmallVector<S>& vec);
std::string ssprintf(const char* fmt, ...)
        __attribute__((format(printf, 1, 2)));

/*!
 * \brief transpose (m*n) matrix to (n*m) matrix
 *
 * -1 in \p lds and \p ldd means default leading dimensions (= nr. columns)
 *
 * Note that transpose and transpose_knc2nsck are implemented in x86/utils.cpp
 * and arm_common/util.cpp, subject to the target platform.
 *
 */
template <typename dtype>
void transpose(const dtype* src, dtype* dst, size_t m, size_t n,
               ptrdiff_t lds = -1, ptrdiff_t ldd = -1);

/*!
 * transpose src with contiguous layout (k, n, c) into dst with shape
 * (n, c, k), with given stride (\p n_stride) on first dimension
 */
template <typename dtype>
void transpose_knc2nsck(const dtype* src, dtype* dst, size_t k, size_t n,
                        size_t c, size_t n_stride);

/*!
 * \brief divide get result ceiled to int; both dividend and divisor shoud be
 * non-negative
 */
template <typename int_t>
int_t div_ceil(int_t dividend, int_t divisor);

/*!
 * \brief divide get result floored to int; both dividend and divisor shoud be
 * non-negative
 */
template <typename int_t>
int_t div_floor(int_t dividend, int_t divisor);

/*!
 * \brief get geometric mean of a and b
 */
inline dt_float32 geometric_mean(dt_float32 a, dt_float32 b) {
    return std::sqrt(a * b);
}

/*!
 * \brief calculate x*x
 */
template <typename num_t>
num_t sqr(num_t x) {
    return x * x;
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/**
 * \brief Aligned workspace bundle.
 *
 * Each individual workspace is aligned to align_in_bytes.
 */
class WorkspaceBundle {
public:
    WorkspaceBundle(void* ptr, SmallVector<size_t> sizes_in_bytes,
                    size_t align_in_bytes = 512);
    /**
     * \returns raw workspace ptr.
     *
     * Note that ptr() is different than get(0), in that
     * the result of ptr() is possibly not aligned.
     */
    void* ptr() const;
    /**
     * \returns the i-th workspace ptr (aligned)
     */
    void* get(size_t i) const;
    /**
     * \returns total size taking into account paddings to solve alignment
     * issue.
     */
    size_t total_size_in_bytes() const;
    size_t get_size(size_t i) const;
    size_t nr_workspace() const;
    void set(void* ptr);

    Workspace get_workspace(size_t i) const {
        return {static_cast<dt_byte*>(get(i)), get_size(i)};
    }

private:
    void* m_ptr;
    SmallVector<size_t> m_sizes;
    SmallVector<size_t> m_aligned_sizes;
    size_t m_align_in_bytes;
};

MEGDNN_CONSTEXPR std::size_t operator"" _z(unsigned long long n) {
    return n;
}

constexpr uint32_t operator"" _hash(char const* str, size_t count) {
    return XXHash64CT::hash(str, count, 20160701);
}

template <typename Vec>
std::string vec2str(Vec&& vec) {
    std::string res;
    res.append("{");
    for (size_t i = 0; i < vec.size(); ++i) {
        res.append(std::to_string(vec[i]));
        if (i + 1 < vec.size())
            res.append(",");
    }
    res.append("}");
    return res;
}

// facilitate tile and repeat
size_t count_not_ones_in_shape(const TensorShape& shape);

/*!
 * \brief whether a TensorLayout is of NHWC format and contiguous on the W and
 *  C dimensions.
 *
 * if true, it implies that a TensorND with given layout is convertible to
 * a Mat for the use of cv algorithms.
 */
bool is_nhwc_contig_wc(const TensorLayout& layout);

static inline void copy_plane_in_bytes(void* dst, const void* src,
                                       size_t height, size_t width,
                                       size_t stride_dst, size_t stride_src) {
    for (size_t h = 0; h < height; ++h) {
        std::memcpy(static_cast<unsigned char*>(dst) + h * stride_dst,
                    static_cast<const unsigned char*>(src) + h * stride_src,
                    width);
    }
}

megcoreDeviceHandle_t get_device_handle(Handle* handle);

static inline void incr_voidp(void*& ptr, ptrdiff_t delta) {
    ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) + delta);
}

/*!
 * \brief align *val* to be multiples of *align*
 * \param align required alignment, which must be power of 2
 */
template <typename T>
static inline T get_aligned_power2(T val, T align) {
    auto d = val & (align - 1);
    val += (align - d) & (align - 1);
    return val;
}

template <typename T, typename S>
inline T saturate(S x, S lower, S upper) {
    //! in(nan) -> out(lower) :
    //! match the meaning with fmax(in dtype.h) when dealing with nan
    S val = x > upper ? upper : (x >= lower ? x : lower);
    return static_cast<T>(val);
}

/*!
 * \brief divide get result ceiled to int; both dividend and divisor shoud be
 * non-negative
 */
template <typename int_t>
int_t div_ceil(int_t dividend, int_t divisor) {
    static_assert(std::is_integral<int_t>::value, "must be integers");
    megdnn_assert_internal(dividend >= 0);
    megdnn_assert_internal(divisor > 0);
    return (dividend + divisor - 1) / divisor;
}

/*!
 * \brief divide get result floored to int; both dividend and divisor shoud be
 * non-negative
 */
template <typename int_t>
int_t div_floor(int_t dividend, int_t divisor) {
    static_assert(std::is_integral<int_t>::value, "must be integers");
    megdnn_assert_internal(dividend >= 0);
    megdnn_assert_internal(divisor > 0);
    return dividend / divisor;
}

/*!
 * \brief round result to multiply of divisor; both dividend and divisor shoud
 * be non-negative
 */
template <typename int_t>
int_t round_up(int_t dividend, int_t divisor) {
    static_assert(std::is_integral<int_t>::value, "must be integers");
    megdnn_assert_internal(dividend >= 0);
    megdnn_assert_internal(divisor > 0);
    return ((dividend + divisor - 1) / divisor) * divisor;
}

template <typename T, typename S, typename Func>
SmallVector<T> apply_vector(Func&& func, const SmallVector<S>& vec) {
    SmallVector<T> res(vec.size());
    std::transform(vec.begin(), vec.end(), res.begin(), func);
    return res;
}

template <typename T>
struct SafeMultiplies;

template <typename T>
struct _SafeMultipliesImplUnsigned : public std::binary_function<T, T, T> {
    static MEGDNN_CONSTEXPR size_t nbits = sizeof(T) * 8;

    static size_t clz(unsigned x) {
        size_t n;
#if defined(_MSC_VER)
        DWORD leading_zero;
        _BitScanReverse(&leading_zero, x);
        n = 31 - leading_zero;
#else
        n = __builtin_clz(x);
#endif
        return x ? n : nbits;
    }

    static size_t clz(unsigned long x) {
        size_t n;
#if defined(_MSC_VER)
        DWORD leading_zero;
        _BitScanReverse(&leading_zero, x);
        n = 31 - leading_zero;
#else
        n = __builtin_clzl(x);
#endif
        return x ? n : nbits;
    }

    static size_t clz(unsigned long long x) {
        size_t n;
#if defined(_MSC_VER)
        DWORD leading_zero;
        _BitScanReverse64(&leading_zero, x);
        n = 63 - leading_zero;
#else
        n = __builtin_clzll(x);
#endif
        return x ? n : nbits;
    }

    T operator()(const T& x, const T& y) const {
        int overflow = clz(x) + clz(y) + 2 <= nbits;
        T t = x * (y >> 1);  // clz(x)+clz(y/2) >= nbits, t must not overflow
        overflow |= t >> (nbits - 1);
        t <<= 1;
        auto yodd = y & 1;
        t += yodd ? x : 0;
        overflow |= yodd & (t < x);

        megdnn_assert(!overflow, "multiply overflow: %s %s",
                      std::to_string(x).c_str(), std::to_string(y).c_str());
        return t;
    }

    template <typename U, typename V>
    U operator()(const U&, const V&) const {
        static_assert(
                // can not be true
                std::is_same<U, T>::value && std::is_same<V, T>::value,
                "implicit conversion disallowed in SafeMultiplies");
        megdnn_trap();
    }
};

template <>
struct SafeMultiplies<size_t> : public _SafeMultipliesImplUnsigned<size_t> {};

template <typename T>
bool vec_contains(const std::vector<T>& vec, const T& elem) {
    return std::find(vec.begin(), vec.end(), elem) != vec.end();
}

template <typename T>
bool vec_contains(const SmallVector<T>& vec, const T& elem) {
    return std::find(vec.begin(), vec.end(), elem) != vec.end();
}

float mul_scale(DType lhs, DType rhs);

template <typename stype, typename dtype>
dtype convert(stype src, dtype dst, size_t offset);

template <>
uint8_t convert<dt_quint4, uint8_t>(dt_quint4 src, uint8_t dst, size_t offset);

template <>
dt_quint4 convert<uint8_t, dt_quint4>(uint8_t src, dt_quint4 dst, size_t offset);

template <>
int8_t convert<dt_qint4, int8_t>(dt_qint4 src, int8_t dst, size_t offset);

template <>
dt_qint4 convert<int8_t, dt_qint4>(int8_t src, dt_qint4 dst, size_t offset);

/*!
 * \brief check float equal within given ULP(unit in the last place)
 */
template <class T>
static inline
        typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
        almost_equal(T x, T y, int unit_last_place = 1) {
    return std::abs(x - y) < (std::numeric_limits<T>::epsilon() *
                              std::abs(x + y) * unit_last_place) ||
           std::abs(x - y) < std::numeric_limits<T>::min();
}

bool dtype_almost_equal(DType lhs, DType rhs);

/**
 * \brief N-dimensional index space
 */
class CpuNDRange {
    static MEGDNN_CONSTEXPR size_t MAX_NDIM = MEGDNN_MAX_NDIM;

private:
    size_t m_dim[MAX_NDIM];
    size_t m_dimension;

public:
    //! \brief Constructs seven-dimensional range.
    CpuNDRange(size_t size0, size_t size1, size_t size2, size_t size3,
               size_t size4, size_t size5, size_t size6)
            : m_dimension(7) {
        m_dim[0] = size0;
        m_dim[1] = size1;
        m_dim[2] = size2;
        m_dim[3] = size3;
        m_dim[4] = size4;
        m_dim[5] = size5;
        m_dim[6] = size6;
    }
    //! \brief Constructs range has zero dimensions.
    CpuNDRange() : CpuNDRange(1, 1, 1, 1, 1, 1, 1) { m_dimension = 0; }

    //! \brief Constructs one-dimensional range.
    CpuNDRange(size_t size0) : CpuNDRange(size0, 1, 1, 1, 1, 1, 1) {
        m_dimension = 1;
    }

    //! \brief Constructs two-dimensional range.
    CpuNDRange(size_t size0, size_t size1)
            : CpuNDRange(size0, size1, 1, 1, 1, 1, 1) {
        m_dimension = 2;
    }

    //! \brief Constructs three-dimensional range.
    CpuNDRange(size_t size0, size_t size1, size_t size2)
            : CpuNDRange(size0, size1, size2, 1, 1, 1, 1) {
        m_dimension = 3;
    }

    //! \brief Constructs four-dimensional range.
    CpuNDRange(size_t size0, size_t size1, size_t size2, size_t size3)
            : CpuNDRange(size0, size1, size2, size3, 1, 1, 1) {
        m_dimension = 4;
    }

    //! \brief Constructs five-dimensional range.
    CpuNDRange(size_t size0, size_t size1, size_t size2, size_t size3,
               size_t size4)
            : CpuNDRange(size0, size1, size2, size3, size4, 1, 1) {
        m_dimension = 5;
    }

    //! \brief Constructs six-dimensional range.
    CpuNDRange(size_t size0, size_t size1, size_t size2, size_t size3,
               size_t size4, size_t size5)
            : CpuNDRange(size0, size1, size2, size3, size4, size5, 1) {
        m_dimension = 6;
    }

    //! \brief Constructs every dim from global
    CpuNDRange(const CpuNDRange& dims, size_t global) {
        m_dimension = dims.dimension();
        for (int i = m_dimension - 1; i >= 0; i--) {
            m_dim[i] = global % dims[i];
            global /= dims[i];
        }
    }

    //! \brief Queries the number of dimensions in the range.
    size_t dimension() const { return m_dimension; }

    //! \brief Returns the size of the object in bytes based on the
    // runtime number of dimensions
    size_t size() const { return m_dimension * sizeof(size_t); }

    size_t* get() { return m_dimension ? m_dim : nullptr; }

    size_t& operator[](size_t idx);
    size_t& operator[](size_t idx) const {
        return const_cast<CpuNDRange*>(this)->operator[](idx);
    };

    const size_t* get() const { return const_cast<CpuNDRange*>(this)->get(); }

    size_t total_size() const {
        size_t ret = 1;
        for (size_t i = 0; i < m_dimension; i++) {
            ret *= m_dim[i];
        }
        return ret;
    }

    //! \brief get the dims string
    std::string to_string() const;
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
