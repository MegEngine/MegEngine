/**
 * half - IEEE 754-based half-precision floating point library.
 *
 * Copyright (c) 2012-2013 Christian Rau <rauy@users.sourceforge.net>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * Version 1.11.0
 * \file
 * Main header file for half precision functionality.
 *
 * --------------------------------------------------------------------------
 * \file include/megdnn/dtype/bfloat16.hpp
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 *
 * --------------------------------------------------------------------------
 */

#ifndef BFLOAT16_BFLOAT16_HPP
#define BFLOAT16_BFLOAT16_HPP
#include "megdnn/arch.h"

#include "megdnn/dtype/half_common_prologue.h"
#include <cstdio>

#if !(HALF_ENABLE_CPP11_CSTDINT & HALF_ENABLE_CPP11_CMATH & \
      HALF_ENABLE_CPP11_HASH)
#error "Should use --std=c++11 option for compile."
#endif

/// Default rounding mode.
/// This specifies the rounding mode used for all conversions between
/// [half](\ref half_bfloat16::bfloat16)s and `float`s as well as for the
/// half_cast() if not specifying a rounding mode explicitly. It can be
/// redefined (before including half.hpp) to one of the standard rounding modes
/// using their respective constants or the equivalent values of
/// `float_round_style`:
///
/// `float_round_style`         | value | rounding
/// ---------------------------------|-------|-------------------------
/// `round_indeterminate`       | -1    | fastest (default)
/// `round_toward_zero`         | 0     | toward zero
/// `round_to_nearest`          | 1     | to nearest
/// `round_toward_infinity`     | 2     | toward positive infinity
/// `round_toward_neg_infinity` | 3     | toward negative infinity
///
/// By default this is set to `1` (`round_to_nearest`). It can even be set to
/// `numeric_limits<float>::round_style` to synchronize the rounding mode with
/// that of the underlying single-precision implementation.
#ifndef BFLOAT16_ROUND_STYLE
#define BFLOAT16_ROUND_STYLE 1  // = to nearest
#endif

/// Tie-breaking behaviour for round to nearest.
/// This specifies if ties in round to nearest should be resolved by rounding to
/// the nearest even value. By default this is defined to `1` resulting of
/// rounding to the nearest even in half-way cases, but can be redefined to
/// `0` (before including bfloat16.hpp). And thus equal to the round() function.
#ifndef BFLOAT16_ROUND_TIES_TO_EVEN
#define BFLOAT16_ROUND_TIES_TO_EVEN 1  // ties round to nearest even.
#endif

#if !BFLOAT16_ROUND_TIES_TO_EVEN
#error "BFloat16 only support round ties to even now."
#endif

/// Value signaling overflow.
/// In correspondence with `HUGE_VAL[F|L]` from `<cmath>` this symbol expands to
/// a positive value signaling the overflow of an operation, in particular it
/// just evaluates to positive infinity.
#define HUGE_VALBH numeric_limits<half_bfloat16::bfloat16>::infinity()

/// Fast half-precision fma function.
/// This symbol is only defined if the fma() function generally executes as fast
/// as, or faster than, a separate half-precision multiplication followed by an
/// addition. Due to the internal single-precision implementation of all
/// arithmetic operations, this is in fact always the case.
#define FP_FAST_FMAH 1

#ifndef FP_ILOGB0
#define FP_ILOGB0 INT_MIN
#endif
#ifndef FP_ILOGBNAN
#define FP_ILOGBNAN INT_MAX
#endif
#ifndef FP_SUBNORMAL
#define FP_SUBNORMAL 0
#endif
#ifndef FP_ZERO
#define FP_ZERO 1
#endif
#ifndef FP_NAN
#define FP_NAN 2
#endif
#ifndef FP_INFINITE
#define FP_INFINITE 3
#endif
#ifndef FP_NORMAL
#define FP_NORMAL 4
#endif

/// Main namespace for bfloat16 functionality.
/// This namespace contains all the functionality provided by the library.
/// Bfloat16 has the following format:
/// Sign bit: 1 bit
/// Exponent width: 8 bits
/// Significand precision: 8 bits (7 explicitly stored), as opposed to 24 bits
/// in a classical single-precision floating-point format
namespace half_bfloat16 {
class bfloat16;
/// \internal
/// \brief Implementation details.
namespace detail {
#if HALF_ENABLE_CPP11_TYPE_TRAITS
/// Conditional type.
template <bool B, typename T, typename F>
struct conditional : std::conditional<B, T, F> {};

/// Helper for tag dispatching.
template <bool B>
struct bool_type : std::integral_constant<bool, B> {};
using std::false_type;
using std::true_type;

/// Type traits for floating point types.
template <typename T>
struct is_float : std::is_floating_point<T> {};
#else
/// Conditional type.
template <bool, typename T, typename>
struct conditional {
    typedef T type;
};
template <typename T, typename F>
struct conditional<false, T, F> {
    typedef F type;
};

/// Helper for tag dispatching.
template <bool>
struct bool_type {};
typedef bool_type<true> true_type;
typedef bool_type<false> false_type;

/// Type traits for floating point types.
template <typename>
struct is_float : false_type {};
template <typename T>
struct is_float<const T> : is_float<T> {};
template <typename T>
struct is_float<volatile T> : is_float<T> {};
template <typename T>
struct is_float<const volatile T> : is_float<T> {};
template <>
struct is_float<float> : true_type {};
template <>
struct is_float<double> : true_type {};
template <>
struct is_float<long double> : true_type {};
#endif

/// Unsigned integer of (at least) 16 bits width.
typedef uint_least16_t uint16;

/// Unsigned integer of (at least) 32 bits width.
typedef uint_least32_t uint32;

/// Fastest signed integer capable of holding all values of type uint16.
typedef int_fast32_t int17;

/// Tag type for binary_t() construction.
struct binary_t {};

/// Temporary bfloat16 expression.
/// This class represents a bfloat16 expression which just stores a
/// single-precision value internally.
struct expr {
    /// Conversion constructor.
    /// \param f single-precision value to convert
    MEGDNN_HOST MEGDNN_DEVICE explicit HALF_CONSTEXPR expr(float f)
            : value_(f) {}

    /// Conversion to single-precision.
    /// \return single precision value representing expression value
    MEGDNN_HOST MEGDNN_DEVICE HALF_CONSTEXPR operator float() const {
        return value_;
    }

private:
    /// Internal expression value stored in single-precision.
    float value_;
};

/// SFINAE helper for generic bfloat16 functions.
/// This class template has to be specialized for each valid combination of
/// argument types to provide a corresponding `type` member equivalent to \a T.
/// \tparam T type to return
template <typename T, typename, typename = void, typename = void>
struct enable {};
template <typename T>
struct enable<T, bfloat16, void, void> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, void, void> {
    typedef T type;
};
template <typename T>
struct enable<T, bfloat16, bfloat16, void> {
    typedef T type;
};
template <typename T>
struct enable<T, bfloat16, expr, void> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, bfloat16, void> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, expr, void> {
    typedef T type;
};
template <typename T>
struct enable<T, bfloat16, bfloat16, bfloat16> {
    typedef T type;
};
template <typename T>
struct enable<T, bfloat16, bfloat16, expr> {
    typedef T type;
};
template <typename T>
struct enable<T, bfloat16, expr, bfloat16> {
    typedef T type;
};
template <typename T>
struct enable<T, bfloat16, expr, expr> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, bfloat16, bfloat16> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, bfloat16, expr> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, expr, bfloat16> {
    typedef T type;
};
template <typename T>
struct enable<T, expr, expr, expr> {
    typedef T type;
};

/// Return type for specialized generic 2-argument bfloat16 functions.
/// This class template has to be specialized for each valid combination of
/// argument types to provide a corresponding `type` member denoting the
/// appropriate return type. \tparam T first argument type \tparam U first
/// argument type
template <typename T, typename U>
struct result : enable<expr, T, U> {};
template <>
struct result<bfloat16, bfloat16> {
    typedef bfloat16 type;
};

/// \name Classification helpers
/// \{

/// Check for infinity.
/// \tparam T argument type (builtin floating point type)
/// \param arg value to query
/// \retval true if infinity
/// \retval false else
template <typename T>
MEGDNN_HOST MEGDNN_DEVICE bool builtin_isinf(T arg) {
#if defined(__CUDA_ARCH__)
    return ::isinf(arg);
#elif HALF_ENABLE_CPP11_CMATH
    return ::std::isinf(arg);
#elif defined(_MSC_VER)
    return !_finite(static_cast<double>(arg)) &&
           !_isnan(static_cast<double>(arg));
#else
    return arg == std::numeric_limits<T>::infinity() ||
           arg == -std::numeric_limits<T>::infinity();
#endif
}

/// Check for NaN.
/// \tparam T argument type (builtin floating point type)
/// \param arg value to query
/// \retval true if not a number
/// \retval false else
template <typename T>
MEGDNN_HOST MEGDNN_DEVICE bool builtin_isnan(T arg) {
#if defined(__CUDA_ARCH__)
    return ::isnan(arg);
#elif HALF_ENABLE_CPP11_CMATH
    return std::isnan(arg);
#elif defined(_MSC_VER)
    return _isnan(static_cast<double>(arg)) != 0;
#else
    return arg != arg;
#endif
}

/// Check sign.
/// \tparam T argument type (builtin floating point type)
/// \param arg value to query
/// \retval true if signbit set
/// \retval false else
template <typename T>
MEGDNN_HOST MEGDNN_DEVICE bool builtin_signbit(T arg) {
#if defined(__CUDA_ARCH__)
    return ::signbit(arg);
#elif HALF_ENABLE_CPP11_CMATH
    return std::signbit(arg);
#else
    return arg < T() || (arg == T() && T(1) / arg < T());
#endif
}

/// \}
/// \name Conversion
/// \{

/// Convert single-precision to bfloat16.
/// \param value single-precision value
/// \return binary_t() representation of bfloat16-precision value
template <std::float_round_style R>
MEGDNN_HOST MEGDNN_DEVICE uint16 float2bfloat16(float value) {
#if HALF_ENABLE_CPP11_STATIC_ASSERT
    static_assert(std::numeric_limits<float>::is_iec559,
                  "float to bfloat16 conversion needs IEEE 754 "
                  "conformant 'float' type");
    static_assert(sizeof(uint32) == sizeof(float),
                  "float to bfloat16 conversion needs unsigned integer "
                  "type of exactly the size of a 'float'");
    static_assert(R == std::round_to_nearest, "Only support rouding-mode "
            "round-to-nearst currently.");
#endif

    union {
        float fraw;
        uint32_t int_raw;
    } r = {value};
    if (~r.int_raw & 0x7f800000) {
        //! When the exponent bits are not all 1s, then the value is zero,
        //! normal, or subnormal.
        r.int_raw += 0x7fff + ((r.int_raw >> 16) & 1);
    } else if (r.int_raw & 0xffff) {
        //! When all of the exponent bits are 1, the value is Inf or NaN.
        //! Preserve signaling NaN here.
        r.int_raw |= 0x10000;
    }
    return uint16(r.int_raw >> 16);
}

/// Convert integer to bfloat16 floating point.
/// \tparam R rounding mode to use, `round_indeterminate` for fastest rounding
/// \tparam T type to convert (builtin integer type)
/// \param value integral value
/// \return binary_t() representation of bfloat16-precision value
template <std::float_round_style R, typename T>
MEGDNN_HOST MEGDNN_DEVICE uint16 int2bfloat16(T value) {
    return float2bfloat16<std::round_to_nearest>(static_cast<float>(value));
}

/// Convert bfloat16 to single-precision.
/// \param value binary_t() representation of bfloat16 value
/// \return single-precision value
MEGDNN_HOST MEGDNN_DEVICE inline float bfloat162float(uint16 value) {
#if HALF_ENABLE_CPP11_STATIC_ASSERT
    static_assert(std::numeric_limits<float>::is_iec559,
                  "bfloat16 to float conversion needs IEEE 754 conformant "
                  "'float' type");
    static_assert(sizeof(uint32) == sizeof(float),
                  "bfloat16 to float conversion needs unsigned integer type of "
                  "exactly the size of a 'float'");
#endif
    union {
        uint32_t int_raw;
        float fraw;
    } r = {uint32_t(value) << 16};
    return r.fraw;
}

/// Convert bfloat16 floating point to integer.
/// \tparam T type to convert to (buitlin integer type with at least 16 bits
/// precision, excluding any implicit sign bits) \param value binary_t()
/// representation of bfloat16-precision value \return integral value
template <typename T>
MEGDNN_HOST MEGDNN_DEVICE T bfloat162int(uint16 value) {
    return static_cast<T>(bfloat162float(value));
}

/// Round bfloat16 number to nearest integer value.
/// \tparam R rounding mode to use, `round_indeterminate` for fastest rounding
/// \tparam E `true` for round to even, `false` for round away from zero
/// \param value binary_t() representation of bfloat16-precision value
/// \return bfloat16 bits for nearest integral value
template <std::float_round_style R, bool E>
MEGDNN_HOST MEGDNN_DEVICE uint16 round_bfloat16_impl(uint16 value) {
    unsigned int e = value & 0x7FFF;
    uint16 result = value;
    if (e < 0x3F80) {
        result &= 0x8000;
        if (R == std::round_to_nearest)
            result |= 0x3F80U & -(e >= (0x3F00 + E));
        else if (R == std::round_toward_infinity)
            result |= 0x3F80U & -(~(value >> 15) & (e != 0));
        else if (R == std::round_toward_neg_infinity)
            result |= 0x3F80U & -(value > 0x8000);
    } else if (e < 0x4300) {
        e = 134 - (e >> 7);
        unsigned int mask = (1 << e) - 1;
        if (R == std::round_to_nearest)
            result += (1 << (e - 1)) - (~(result >> e) & E);
        else if (R == std::round_toward_infinity)
            result += mask & ((value >> 15) - 1);
        else if (R == std::round_toward_neg_infinity)
            result += mask & -(value >> 15);
        result &= ~mask;
    }
    return result;
}

/// Round bfloat16 number to nearest integer value.
/// \tparam R rounding mode to use, `round_indeterminate` for fastest rounding
/// \param value binary_t() representation of bfloat16-precision value
/// \return bfloat16 bits for nearest integral value
template <std::float_round_style R>
MEGDNN_HOST MEGDNN_DEVICE uint16 round_bfloat16(uint16 value) {
    return round_bfloat16_impl<R, BFLOAT16_ROUND_TIES_TO_EVEN>(value);
}

/// Round bfloat16 number to nearest integer value using
/// round-to-nearest-away-from-zero. \param value binary_t() representation of
/// bfloat16-precision value \return bfloat16-precision bits for nearest
/// integral value
MEGDNN_HOST MEGDNN_DEVICE inline uint16 round_bfloat16_up(uint16 value) {
    return round_bfloat16_impl<std::round_to_nearest, 0>(value);
}
/// \}

struct functions;
template <typename>
struct unary_specialized;
template <typename, typename>
struct binary_specialized;
template <typename, typename, std::float_round_style>
struct bfloat16_caster;
}

/// bfloat16 floating point type.
class bfloat16 {
    friend struct detail::functions;
    friend struct detail::unary_specialized<bfloat16>;
    friend struct detail::binary_specialized<bfloat16, bfloat16>;
    template <typename, typename, std::float_round_style>
    friend struct detail::bfloat16_caster;
#if HALF_ENABLE_CPP11_HASH
    friend struct std::hash<bfloat16>;
#endif

public:
    /// Default constructor.
    MEGDNN_HOST MEGDNN_DEVICE bfloat16() {}

    /// Copy constructor.
    /// \tparam T type of concrete bfloat16 expression
    /// \param rhs bfloat16 expression to copy from
    MEGDNN_HOST MEGDNN_DEVICE bfloat16(detail::expr rhs)
            : data_(detail::float2bfloat16<round_style>(rhs)) {}

    MEGDNN_HOST MEGDNN_DEVICE HALF_CONSTEXPR bfloat16(const bfloat16& rhs)
            : data_(rhs.data_) {}

    MEGDNN_HOST MEGDNN_DEVICE bfloat16(const volatile bfloat16& rhs)
            : data_(rhs.data_) {}

    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator=(const bfloat16& rhs) {
        data_ = rhs.data_;
        return *this;
    }

    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator=(
            const volatile bfloat16& rhs) {
        data_ = rhs.data_;
        return *this;
    }

    MEGDNN_HOST MEGDNN_DEVICE volatile bfloat16& operator=(
            const bfloat16& rhs) volatile {
        data_ = rhs.data_;
        return *this;
    }

    /// Conversion constructor.
    /// \param rhs float to convert
    // MEGDNN_HOST MEGDNN_DEVICE explicit bfloat16(float rhs)
    // : data_(detail::float2bfloat16<round_style>(rhs)) {}

    MEGDNN_HOST MEGDNN_DEVICE explicit bfloat16(float rhs) {
        data_ = detail::float2bfloat16<round_style>(rhs);
    }

    /// Conversion to single-precision.
    /// \return single precision value representing expression value
    MEGDNN_HOST MEGDNN_DEVICE operator float() const {
        return detail::bfloat162float(data_);
    }

    /// Assignment operator.
    /// \tparam T type of concrete bfloat16 expression
    /// \param rhs bfloat16 expression to copy from
    /// \return reference to this bfloat16
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator=(detail::expr rhs) {
        return *this = static_cast<float>(rhs);
    }

    /// Arithmetic assignment.
    /// \tparam T type of concrete bfloat16 expression
    /// \param rhs bfloat16 expression to add
    /// \return reference to this bfloat16
    template <typename T>
    MEGDNN_HOST MEGDNN_DEVICE typename detail::enable<bfloat16&, T>::type
    operator+=(T rhs) {
        return *this += static_cast<float>(rhs);
    }

    /// Arithmetic assignment.
    /// \tparam T type of concrete bfloat16 expression
    /// \param rhs bfloat16 expression to subtract
    /// \return reference to this bfloat16
    template <typename T>
    MEGDNN_HOST MEGDNN_DEVICE typename detail::enable<bfloat16&, T>::type
    operator-=(T rhs) {
        return *this -= static_cast<float>(rhs);
    }

    /// Arithmetic assignment.
    /// \tparam T type of concrete bfloat16 expression
    /// \param rhs bfloat16 expression to multiply with
    /// \return reference to this bfloat16
    template <typename T>
    MEGDNN_HOST MEGDNN_DEVICE typename detail::enable<bfloat16&, T>::type
    operator*=(T rhs) {
        return *this *= static_cast<float>(rhs);
    }

    /// Arithmetic assignment.
    /// \tparam T type of concrete bfloat16 expression
    /// \param rhs bfloat16 expression to divide by
    /// \return reference to this bfloat16
    template <typename T>
    MEGDNN_HOST MEGDNN_DEVICE typename detail::enable<bfloat16&, T>::type
    operator/=(T rhs) {
        return *this /= static_cast<float>(rhs);
    }

    /// Assignment operator.
    /// \param rhs single-precision value to copy from
    /// \return reference to this bfloat16
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator=(float rhs) {
        data_ = detail::float2bfloat16<round_style>(rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to add
    /// \return reference to this bfloat16
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator+=(float rhs) {
        data_ = detail::float2bfloat16<round_style>(
                detail::bfloat162float(data_) + rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to subtract
    /// \return reference to this bfloat16
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator-=(float rhs) {
        data_ = detail::float2bfloat16<round_style>(
                detail::bfloat162float(data_) - rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to multiply with
    /// \return reference to this bfloat16
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator*=(float rhs) {
        data_ = detail::float2bfloat16<round_style>(
                detail::bfloat162float(data_) * rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to divide by
    /// \return reference to this bfloat16
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator/=(float rhs) {
        data_ = detail::float2bfloat16<round_style>(
                detail::bfloat162float(data_) / rhs);
        return *this;
    }

    /// Prefix increment.
    /// \return incremented bfloat16 value
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator++() { return *this += 1.0f; }

    /// Prefix decrement.
    /// \return decremented bfloat16 value
    MEGDNN_HOST MEGDNN_DEVICE bfloat16& operator--() { return *this -= 1.0f; }

    /// Postfix increment.
    /// \return non-incremented bfloat16 value
    MEGDNN_HOST MEGDNN_DEVICE bfloat16 operator++(int) {
        bfloat16 out(*this);
        ++*this;
        return out;
    }

    /// Postfix decrement.
    /// \return non-decremented bfloat16 value
    MEGDNN_HOST MEGDNN_DEVICE bfloat16 operator--(int) {
        bfloat16 out(*this);
        --*this;
        return out;
    }

    /// Constructor.
    /// \param bits binary_t() representation to set bfloat16 to
    MEGDNN_HOST MEGDNN_DEVICE HALF_CONSTEXPR bfloat16(detail::binary_t,
                                                      detail::uint16 bits)
            : data_(bits) {}

    /// Rounding mode to use (always `round_to_nearest` with
    /// BFLOAT16_ROUND_TIES_TO_EVEN on)
    static HALF_CONSTEXPR_CONST std::float_round_style round_style =
            (std::float_round_style)(BFLOAT16_ROUND_STYLE);

    // private:
    /// Internal binary_t() representation
    detail::uint16 data_;
};

#if HALF_ENABLE_CPP11_USER_LITERALS
/// Library-defined bfloat16 literals.
/// Import this namespace to enable bfloat16 floating point literals:
/// ~~~~{.cpp}
/// using namespace half_bfloat16::literal;
/// half_bfloat16::bfloat16 = 4.2_h;
/// ~~~~
namespace literal {
/// Half literal.
/// While this returns an actual bfloat16-precision value, bfloat16 literals can
/// unfortunately not be constant expressions due to rather involved
/// single-to-bfloat16 conversion. \param value literal value \return bfloat16
/// with given value (if representable)
inline bfloat16 operator"" _h(long double value) {
    return bfloat16(static_cast<float>(value));
}
}  // namespace literal
#endif

namespace detail {
/// Wrapper implementing unspecialized bfloat16 functions.
struct functions {
    /// Addition implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return bfloat16 sum stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr plus(float x, float y) {
        return expr(x + y);
    }

    /// Subtraction implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return bfloat16 difference stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr minus(float x, float y) {
        return expr(x - y);
    }

    /// Multiplication implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return bfloat16 product stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr multiplies(float x, float y) {
        return expr(x * y);
    }

    /// Division implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return bfloat16 quotient stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr divides(float x, float y) {
        return expr(x / y);
    }

    /// Output implementation.
    /// \param out stream to write to
    /// \param arg value to write
    /// \return reference to stream
    template <typename charT, typename traits>
    static std::basic_ostream<charT, traits>& write(
            std::basic_ostream<charT, traits>& out, float arg) {
        return out << arg;
    }

    /// Input implementation.
    /// \param in stream to read from
    /// \param arg bfloat16 to read into
    /// \return reference to stream
    template <typename charT, typename traits>
    static std::basic_istream<charT, traits>& read(
            std::basic_istream<charT, traits>& in, bfloat16& arg) {
        float f;
        if (in >> f)
            arg = f;
        return in;
    }

    /// Modulo implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return bfloat16 division remainder stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr fmod(float x, float y) {
#if defined(__CUDA_ARCH__)
        return expr(fmodf(x, y));
#else
        return expr(std::fmod(x, y));
#endif
    }

    /// Remainder implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return bfloat16 division remainder stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr remainder(float x, float y) {
#if defined(__CUDA_ARCH__)
        return expr(remainderf(x, y));
#else
        return expr(std::remainder(x, y));
#endif
    }

    /// Positive difference implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return Positive difference stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr fdim(float x, float y) {
#if defined(__CUDA_ARCH__)
        return expr(fdimf(x, y));
#else
        return expr(std::fdim(x, y));
#endif
    }

    /// Fused multiply-add implementation.
    /// \param x first operand
    /// \param y second operand
    /// \param z third operand
    /// \return \a x * \a y + \a z stored in single-precision
    MEGDNN_HOST MEGDNN_DEVICE static expr fma(float x, float y, float z) {
#if defined(__CUDA_ARCH__)
        return expr(fmaf(x, y, z));
#elif defined(FP_FAST_FMAF)
        return expr(std::fma(x, y, z));
#else
        return expr(x * y + z);
#endif
    }

    /// Get NaN.
    /// \return bfloat16 quiet NaN
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 nanh(const char*) {
        return bfloat16(binary_t(), 0x7FFF);
    }

    /// Exponential implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr exp(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(expf(arg));
#else
        return expr(std::exp(arg));
#endif
    }

    /// Exponential implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr expm1(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(expm1f(arg));
#else
        return expr(std::expm1(arg));
#endif
    }

    /// Binary exponential implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr exp2(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(exp2f(arg));
#else
        return expr(std::exp2(arg));
#endif
    }

    /// Logarithm implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr log(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(logf(arg));
#else
        return expr(std::log(arg));
#endif
    }

    /// Common logarithm implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr log10(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(log10f(arg));
#else
        return expr(std::log10(arg));
#endif
    }

    /// Logarithm implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr log1p(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(log1pf(arg));
#else
        return expr(std::log1p(arg));
#endif
    }

    /// Binary logarithm implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr log2(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(log2f(arg));
#else
        return expr(std::log2(arg));
#endif
    }

    /// Square root implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr sqrt(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(sqrtf(arg));
#else
        return expr(std::sqrt(arg));
#endif
    }

    /// Cubic root implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr cbrt(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(cbrtf(arg));
#else
        return expr(std::cbrt(arg));
#endif
    }

    /// Hypotenuse implementation.
    /// \param x first argument
    /// \param y second argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr hypot(float x, float y) {
#if defined(__CUDA_ARCH__)
        return expr(hypotf(x, y));
#else
        return expr(std::hypot(x, y));
#endif
    }

    /// Power implementation.
    /// \param base value to exponentiate
    /// \param exp power to expontiate to
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr pow(float base, float exp) {
#if defined(__CUDA_ARCH__)
        return expr(powf(base, exp));
#else
        return expr(std::pow(base, exp));
#endif
    }

    /// Sine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr sin(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(sinf(arg));
#else
        return expr(std::sin(arg));
#endif
    }

    /// Cosine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr cos(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(cosf(arg));
#else
        return expr(std::cos(arg));
#endif
    }

    /// Tan implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr tan(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(tanf(arg));
#else
        return expr(std::tan(arg));
#endif
    }

    /// Arc sine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr asin(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(asinf(arg));
#else
        return expr(std::asin(arg));
#endif
    }

    /// Arc cosine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr acos(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(acosf(arg));
#else
        return expr(std::acos(arg));
#endif
    }

    /// Arc tangent implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr atan(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(atanf(arg));
#else
        return expr(std::atan(arg));
#endif
    }

    /// Arc tangent implementation.
    /// \param x first argument
    /// \param y second argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr atan2(float x, float y) {
#if defined(__CUDA_ARCH__)
        return expr(atan2f(x, y));
#else
        return expr(std::atan2(x, y));
#endif
    }

    /// Hyperbolic sine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr sinh(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(sinhf(arg));
#else
        return expr(std::sinh(arg));
#endif
    }

    /// Hyperbolic cosine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr cosh(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(coshf(arg));
#else
        return expr(std::cosh(arg));
#endif
    }

    /// Hyperbolic tangent implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr tanh(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(tanhf(arg));
#else
        return expr(std::tanh(arg));
#endif
    }

    /// Hyperbolic area sine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr asinh(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(asinhf(arg));
#else
        return expr(std::asinh(arg));
#endif
    }

    /// Hyperbolic area cosine implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr acosh(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(acoshf(arg));
#else
        return expr(std::acosh(arg));
#endif
    }

    /// Hyperbolic area tangent implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr atanh(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(atanhf(arg));
#else
        return expr(std::atanh(arg));
#endif
    }

    /// Error function implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr erf(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(erff(arg));
#else
        return expr(std::erf(arg));
#endif
    }

    /// Complementary implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr erfc(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(erfcf(arg));
#else
        return expr(std::erfc(arg));
#endif
    }

    /// Gamma logarithm implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr lgamma(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(lgammaf(arg));
#else
        return expr(std::lgamma(arg));
#endif
    }

    /// Gamma implementation.
    /// \param arg function argument
    /// \return function value stored in single-preicision
    MEGDNN_HOST MEGDNN_DEVICE static expr tgamma(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(tgammaf(arg));
#else
        return expr(std::tgamma(arg));
#endif
    }

    /// Minimum implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return minimum value
    MEGDNN_HOST MEGDNN_DEVICE static expr fmin(float x, float y) {
        return expr(::fmin(x, y));
    }

    /// Maximum implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return maximum value
    MEGDNN_HOST MEGDNN_DEVICE static expr fmax(float x, float y) {
        return expr(::fmax(x, y));
    }

    /// Floor implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 floor(bfloat16 arg) {
        return bfloat16(
                binary_t(),
                round_bfloat16<std::round_toward_neg_infinity>(arg.data_));
    }

    /// Ceiling implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 ceil(bfloat16 arg) {
        return bfloat16(binary_t(),
                        round_bfloat16<std::round_toward_infinity>(arg.data_));
    }

    /// Truncation implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 trunc(bfloat16 arg) {
        return bfloat16(binary_t(),
                        round_bfloat16<std::round_toward_zero>(arg.data_));
    }

    /// Nearest integer implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 round(bfloat16 arg) {
        return bfloat16(binary_t(), round_bfloat16_up(arg.data_));
    }

    /// Nearest integer implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 rint(bfloat16 arg) {
        return bfloat16(binary_t(),
                        round_bfloat16<bfloat16::round_style>(arg.data_));
    }

    /// Nearest integer implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static long lrint(bfloat16 arg) {
        return detail::bfloat162int<long>(arg.data_);
    }

#if HALF_ENABLE_CPP11_LONG_LONG
    /// Nearest integer implementation.
    /// \param arg value to round
    /// \return rounded value
    MEGDNN_HOST MEGDNN_DEVICE static long long llrint(bfloat16 arg) {
        return detail::bfloat162int<long long>(
                arg.data_);
    }
#endif

    /// Decompression implementation.
    /// \param arg number to decompress
    /// \param exp address to store exponent at
    /// \return normalized significant
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 frexp(float arg, int* exp) {
        return bfloat16(binary_t(), float2bfloat16<bfloat16::round_style>(
                                            std::frexp(arg, exp)));
    }

    /// Decompression implementation.
    /// \param arg number to decompress
    /// \param iptr address to store integer part at
    /// \return fractional part
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 modf(float arg, bfloat16* iptr) {
        float fptr = 0.f;
        bfloat16 ret = bfloat16(
                binary_t(),
                float2bfloat16<bfloat16::round_style>(std::modf(arg, &fptr)));
        *iptr = fptr;
        return ret;
    }

    /// Scaling implementation.
    /// \param arg number to scale
    /// \param exp power of two to scale by
    /// \return scaled number
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 scalbln(float arg, long exp) {
        return bfloat16(binary_t(), float2bfloat16<bfloat16::round_style>(
                                            std::scalbln(arg, exp)));
    }

    /// Exponent implementation.
    /// \param arg number to query
    /// \return floating point exponent
    MEGDNN_HOST MEGDNN_DEVICE static int ilogb(float arg) {
        return std::ilogb(arg);
    }

    /// Exponent implementation.
    /// \param arg number to query
    /// \return floating point exponent
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 logb(bfloat16 arg) {
        return bfloat16(binary_t(),
                        float2bfloat16<bfloat16::round_style>(std::logb(arg)));
    }

    /// Enumeration implementation.
    /// \param from number to increase/decrease
    /// \param to direction to enumerate into
    /// \return next representable number
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 nextafter(bfloat16 from,
                                                        bfloat16 to) {
        uint16 fabs = from.data_ & 0x7FFF, tabs = to.data_ & 0x7FFF;
        if (fabs > 0x7F80)
            return from;
        if (tabs > 0x7F80 || from.data_ == to.data_ || !(fabs | tabs))
            return to;
        if (!fabs)
            return bfloat16(binary_t(), (to.data_ & 0x8000) + 1);
        bool lt = (signbit(from) ? (static_cast<int17>(0x8000) - from.data_)
                                 : static_cast<int17>(from.data_)) <
                  (signbit(to) ? (static_cast<int17>(0x8000) - to.data_)
                               : static_cast<int17>(to.data_));
        return bfloat16(
                binary_t(),
                from.data_ +
                        (((from.data_ >> 15) ^ static_cast<uint16>(lt)) << 1) -
                        1);
    }

    /// Sign implementation
    /// \param x first operand
    /// \param y second operand
    /// \return composed value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 copysign(bfloat16 x, bfloat16 y) {
        return bfloat16(binary_t(), x.data_ ^ ((x.data_ ^ y.data_) & 0x8000));
    }

    /// Classification implementation.
    /// \param arg value to classify
    /// \retval true if infinite number
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static int fpclassify(bfloat16 arg) {
        unsigned int abs = arg.data_ & 0x7FFF;
        if (abs > 0x7F80)
            return FP_NAN;
        if (abs == 0x7F80)
            return FP_INFINITE;
        if (abs > 0x7F)
            return FP_NORMAL;
        return abs ? FP_SUBNORMAL : FP_ZERO;
    }

    /// Classification implementation.
    /// \param arg value to classify
    /// \retval true if finite number
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isfinite(float arg) {
        return std::isfinite(arg);
    }

    /// Classification implementation.
    /// \param arg value to classify
    /// \retval true if infinite number
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isinf(float arg) {
        return std::isinf(arg);
    }

    /// Classification implementation.
    /// \param arg value to classify
    /// \retval true if not a number
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isnan(bfloat16 arg) {
        return (arg.data_ & 0x7FFF) > 0x7F80;
    }

    /// Classification implementation.
    /// \param arg value to classify
    /// \retval true if normal number
    /// \retval false else

    MEGDNN_HOST MEGDNN_DEVICE static bool isnormal(bfloat16 arg) {
        return ((arg.data_ & 0x7F80) != 0) & ((arg.data_ & 0x7F80) != 0x7F80);
    }

    /// Sign bit implementation.
    /// \param arg value to check
    /// \retval true if signed
    /// \retval false if unsigned
    MEGDNN_HOST MEGDNN_DEVICE static bool signbit(bfloat16 arg) {
        return (arg.data_ & 0x8000) != 0;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if operands equal
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isequal(float x, float y) {
        return x == y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if operands not equal
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isnotequal(float x, float y) {
        return x != y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x > \a y
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isgreater(float x, float y) {
        return x > y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x >= \a y
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isgreaterequal(float x, float y) {
        return x >= y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x < \a y
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isless(float x, float y) {
        return x < y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x <= \a y
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool islessequal(float x, float y) {
        return x <= y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true neither \a x > \a y nor \a x < \a y
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool islessgreater(float x, float y) {
        return x < y || x > y;
    }

    /// Comparison implementation.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if operand unordered
    /// \retval false else
    MEGDNN_HOST MEGDNN_DEVICE static bool isunordered(bfloat16 x, bfloat16 y) {
        return isnan(x) || isnan(y);
    }
};

/// Wrapper for unary bfloat16 functions needing specialization for
/// individual argument types. \tparam T argument type
template <typename T>
struct unary_specialized {
    /// Negation implementation.
    /// \param arg value to negate
    /// \return negated value
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR bfloat16
    negate(bfloat16 arg) {
        return bfloat16(binary_t(), arg.data_ ^ 0x8000);
    }

    /// Absolute value implementation.
    /// \param arg function argument
    /// \return absolute value
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 fabs(bfloat16 arg) {
        return bfloat16(binary_t(), arg.data_ & 0x7FFF);
    }
};
template <>
struct unary_specialized<expr> {
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR expr negate(float arg) {
        return expr(-arg);
    }
    MEGDNN_HOST MEGDNN_DEVICE static expr fabs(float arg) {
#if defined(__CUDA_ARCH__)
        return expr(fabsf(arg));
#else
        return expr(std::fabs(arg));
#endif
    }
};

/// Wrapper for binary_t() bfloat16-precision functions needing
/// specialization for individual argument types. \tparam T first argument
/// type \tparam U first argument type
template <typename T, typename U>
struct binary_specialized {
    /// Minimum implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return minimum value
    MEGDNN_HOST MEGDNN_DEVICE static expr fmin(float x, float y) {
        return detail::functions::fmin(x, y);
    }

    /// Maximum implementation.
    /// \param x first operand
    /// \param y second operand
    /// \return maximum value
    MEGDNN_HOST MEGDNN_DEVICE static expr fmax(float x, float y) {
        return detail::functions::fmax(x, y);
    }
};
template <>
struct binary_specialized<bfloat16, bfloat16> {
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 fmin(bfloat16 x, bfloat16 y) {
        return bfloat16(binary_t(),
                        float2bfloat16<bfloat16::round_style>(
                                static_cast<float>(functions::fmin(x, y))));
    }
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 fmax(bfloat16 x, bfloat16 y) {
        return bfloat16(binary_t(),
                        float2bfloat16<bfloat16::round_style>(
                                static_cast<float>(functions::fmax(x, y))));
    }
};

/// Helper class for bfloat16 casts.
/// This class template has to be specialized for all valid cast argument to
/// define an appropriate static `cast` member function and a corresponding
/// `type` member denoting its return type. \tparam T destination type
/// \tparam U source type \tparam R rounding mode to use
template <typename T, typename U,
          std::float_round_style R =
                  (std::float_round_style)(BFLOAT16_ROUND_STYLE)>
struct bfloat16_caster {};
template <typename U, std::float_round_style R>
struct bfloat16_caster<bfloat16, U, R> {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
    static_assert(std::is_arithmetic<U>::value,
                  "bfloat16_cast from non-arithmetic type unsupported");
#endif

    typedef bfloat16 type;
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 cast(U arg) {
        return cast_impl(arg, is_float<U>());
    };

private:
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 cast_impl(U arg, true_type) {
        return bfloat16(binary_t(), float2bfloat16<R>(static_cast<float>(arg)));
    }
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 cast_impl(U arg, false_type) {
        return bfloat16(binary_t(), int2bfloat16<R>(arg));
    }
};
template <typename T, std::float_round_style R>
struct bfloat16_caster<T, bfloat16, R> {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
    static_assert(std::is_arithmetic<T>::value,
                  "bfloat16_cast to non-arithmetic type unsupported");
#endif

    typedef T type;
    template <typename U>
    MEGDNN_HOST MEGDNN_DEVICE static T cast(U arg) {
        return cast_impl(arg, is_float<T>());
    }

private:
    MEGDNN_HOST MEGDNN_DEVICE static T cast_impl(float arg, true_type) {
        return static_cast<T>(arg);
    }
    MEGDNN_HOST MEGDNN_DEVICE static T cast_impl(bfloat16 arg, false_type) {
        return bfloat162int<T>(arg.data_);
    }
};
template <typename T, std::float_round_style R>
struct bfloat16_caster<T, expr, R> : public bfloat16_caster<T, bfloat16, R> {};
template <std::float_round_style R>
struct bfloat16_caster<bfloat16, bfloat16, R> {
    typedef bfloat16 type;
    MEGDNN_HOST MEGDNN_DEVICE static bfloat16 cast(bfloat16 arg) { return arg; }
};
template <std::float_round_style R>
struct bfloat16_caster<bfloat16, expr, R>
        : public bfloat16_caster<bfloat16, bfloat16, R> {};

/// \name Comparison operators
/// \{

/// Comparison for equality.
/// \param x first operand
/// \param y second operand
/// \retval true if operands equal
/// \retval false else
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<bool, T, U>::type operator==(T x,
                                                                       U y) {
    return functions::isequal(x, y);
}

/// Comparison for inequality.
/// \param x first operand
/// \param y second operand
/// \retval true if operands not equal
/// \retval false else
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<bool, T, U>::type operator!=(T x,
                                                                       U y) {
    return functions::isnotequal(x, y);
}

/// Comparison for less than.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x less than \a y
/// \retval false else
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<bool, T, U>::type operator<(T x,
                                                                      U y) {
    return functions::isless(x, y);
}

/// Comparison for greater than.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x greater than \a y
/// \retval false else
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<bool, T, U>::type operator>(T x,
                                                                      U y) {
    return functions::isgreater(x, y);
}

/// Comparison for less equal.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x less equal \a y
/// \retval false else
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<bool, T, U>::type operator<=(T x,
                                                                       U y) {
    return functions::islessequal(x, y);
}

/// Comparison for greater equal.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x greater equal \a y
/// \retval false else
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<bool, T, U>::type operator>=(T x,
                                                                       U y) {
    return functions::isgreaterequal(x, y);
}

/// \}
/// \name Arithmetic operators
/// \{

/// Add bfloat16s.
/// \param x left operand
/// \param y right operand
/// \return sum of bfloat16 expressions
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<expr, T, U>::type operator+(T x,
                                                                      U y) {
    return functions::plus(x, y);
}

/// Subtract bfloat16s.
/// \param x left operand
/// \param y right operand
/// \return difference of bfloat16 expressions
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<expr, T, U>::type operator-(T x,
                                                                      U y) {
    return functions::minus(x, y);
}

/// Multiply bfloat16s.
/// \param x left operand
/// \param y right operand
/// \return product of bfloat16 expressions
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<expr, T, U>::type operator*(T x,
                                                                      U y) {
    return functions::multiplies(x, y);
}

/// Divide bfloat16s.
/// \param x left operand
/// \param y right operand
/// \return quotient of bfloat16 expressions
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename enable<expr, T, U>::type operator/(T x,
                                                                      U y) {
    return functions::divides(x, y);
}

/// Identity.
/// \param arg operand
/// \return uncahnged operand
template <typename T>
MEGDNN_HOST MEGDNN_DEVICE HALF_CONSTEXPR typename enable<T, T>::type operator+(
        T arg) {
    return arg;
}

/// Negation.
/// \param arg operand
/// \return negated operand
template <typename T>
MEGDNN_HOST MEGDNN_DEVICE HALF_CONSTEXPR typename enable<T, T>::type operator-(
        T arg) {
    return unary_specialized<T>::negate(arg);
}

/// \}
/// \name Input and output
/// \{

/// Output operator.
/// \param out output stream to write into
/// \param arg bfloat16 expression to write
/// \return reference to output stream
template <typename T, typename charT, typename traits>
typename enable<std::basic_ostream<charT, traits>&, T>::type operator<<(
        std::basic_ostream<charT, traits>& out, T arg) {
    return functions::write(out, arg);
}

/// Input operator.
/// \param in input stream to read from
/// \param arg bfloat16 to read into
/// \return reference to input stream
template <typename charT, typename traits>
std::basic_istream<charT, traits>& operator>>(
        std::basic_istream<charT, traits>& in, bfloat16& arg) {
    return functions::read(in, arg);
}

/// \}
/// \name Basic mathematical operations
/// \{

/// Absolute value.
/// \param arg operand
/// \return absolute value of \a arg
//		template<typename T> typename enable<T,T>::type abs(T arg) {
// return unary_specialized<T>::fabs(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 abs(bfloat16 arg) {
    return unary_specialized<bfloat16>::fabs(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr abs(expr arg) {
    return unary_specialized<expr>::fabs(arg);
}

/// Absolute value.
/// \param arg operand
/// \return absolute value of \a arg
//		template<typename T> typename enable<T,T>::type fabs(T arg) {
// return unary_specialized<T>::fabs(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 fabs(bfloat16 arg) {
    return unary_specialized<bfloat16>::fabs(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fabs(expr arg) {
    return unary_specialized<expr>::fabs(arg);
}

/// Remainder of division.
/// \param x first operand
/// \param y second operand
/// \return remainder of floating point division.
//		template<typename T,typename U> typename enable<expr,T,U>::type
// fmod(T x, U y) { return functions::fmod(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline expr fmod(bfloat16 x, bfloat16 y) {
    return functions::fmod(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmod(bfloat16 x, expr y) {
    return functions::fmod(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmod(expr x, bfloat16 y) {
    return functions::fmod(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmod(expr x, expr y) {
    return functions::fmod(x, y);
}

/// Remainder of division.
/// \param x first operand
/// \param y second operand
/// \return remainder of floating point division.
//		template<typename T,typename U> typename enable<expr,T,U>::type
// remainder(T x, U y) { return functions::remainder(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline expr remainder(bfloat16 x, bfloat16 y) {
    return functions::remainder(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr remainder(bfloat16 x, expr y) {
    return functions::remainder(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr remainder(expr x, bfloat16 y) {
    return functions::remainder(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr remainder(expr x, expr y) {
    return functions::remainder(x, y);
}

/// Fused multiply add.
/// \param x first operand
/// \param y second operand
/// \param z third operand
/// \return ( \a x * \a y ) + \a z rounded as one operation.
//		template<typename T,typename U,typename V> typename
// enable<expr,T,U,V>::type fma(T x, U y, V z) { return functions::fma(x, y,
// z);
//}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(bfloat16 x, bfloat16 y, bfloat16 z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(bfloat16 x, bfloat16 y, expr z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(bfloat16 x, expr y, bfloat16 z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(bfloat16 x, expr y, expr z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(expr x, bfloat16 y, bfloat16 z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(expr x, bfloat16 y, expr z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(expr x, expr y, bfloat16 z) {
    return functions::fma(x, y, z);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fma(expr x, expr y, expr z) {
    return functions::fma(x, y, z);
}

/// Maximum of bfloat16 expressions.
/// \param x first operand
/// \param y second operand
/// \return maximum of operands
//		template<typename T,typename U> typename result<T,U>::type
// fmax(T  x, U y) { return binary_specialized<T,U>::fmax(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 fmax(bfloat16 x, bfloat16 y) {
    return binary_specialized<bfloat16, bfloat16>::fmax(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmax(bfloat16 x, expr y) {
    return binary_specialized<bfloat16, expr>::fmax(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmax(expr x, bfloat16 y) {
    return binary_specialized<expr, bfloat16>::fmax(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmax(expr x, expr y) {
    return binary_specialized<expr, expr>::fmax(x, y);
}

/// Minimum of bfloat16 expressions.
/// \param x first operand
/// \param y second operand
/// \return minimum of operands
//		template<typename T,typename U> typename result<T,U>::type
// fmin(T  x, U y) { return binary_specialized<T,U>::fmin(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 fmin(bfloat16 x, bfloat16 y) {
    return binary_specialized<bfloat16, bfloat16>::fmin(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmin(bfloat16 x, expr y) {
    return binary_specialized<bfloat16, expr>::fmin(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmin(expr x, bfloat16 y) {
    return binary_specialized<expr, bfloat16>::fmin(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fmin(expr x, expr y) {
    return binary_specialized<expr, expr>::fmin(x, y);
}

/// Positive difference.
/// \param x first operand
/// \param y second operand
/// \return \a x - \a y or 0 if difference negative
//		template<typename T,typename U> typename enable<expr,T,U>::type
// fdim(T x, U y) { return functions::fdim(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline expr fdim(bfloat16 x, bfloat16 y) {
    return functions::fdim(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fdim(bfloat16 x, expr y) {
    return functions::fdim(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fdim(expr x, bfloat16 y) {
    return functions::fdim(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr fdim(expr x, expr y) {
    return functions::fdim(x, y);
}

/// Get NaN value.
/// \param arg descriptive string (ignored)
/// \return quiet NaN
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nanh(const char* arg) {
    return functions::nanh(arg);
}

/// \}
/// \name Exponential functions
/// \{

/// Exponential function.
/// \param arg function argument
/// \return e raised to \a arg
//		template<typename T> typename enable<expr,T>::type exp(T arg) {
// return functions::exp(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr exp(bfloat16 arg) {
    return functions::exp(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr exp(expr arg) {
    return functions::exp(arg);
}

/// Exponential minus one.
/// \param arg function argument
/// \return e raised to \a arg subtracted by 1
//		template<typename T> typename enable<expr,T>::type expm1(T arg)
//{  return functions::expm1(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr expm1(bfloat16 arg) {
    return functions::expm1(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr expm1(expr arg) {
    return functions::expm1(arg);
}

/// Binary exponential.
/// \param arg function argument
/// \return 2 raised to \a arg
//		template<typename T> typename enable<expr,T>::type exp2(T arg) {
// return functions::exp2(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr exp2(bfloat16 arg) {
    return functions::exp2(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr exp2(expr arg) {
    return functions::exp2(arg);
}

/// Natural logorithm.
/// \param arg function argument
/// \return logarithm of \a arg to base e
//		template<typename T> typename enable<expr,T>::type log(T arg) {
// return functions::log(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr log(bfloat16 arg) {
    return functions::log(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr log(expr arg) {
    return functions::log(arg);
}

/// Common logorithm.
/// \param arg function argument
/// \return logarithm of \a arg to base 10
//		template<typename T> typename enable<expr,T>::type log10(T arg)
//{  return functions::log10(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr log10(bfloat16 arg) {
    return functions::log10(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr log10(expr arg) {
    return functions::log10(arg);
}

/// Natural logorithm.
/// \param arg function argument
/// \return logarithm of \a arg plus 1 to base e
//		template<typename T> typename enable<expr,T>::type log1p(T arg)
//{  return functions::log1p(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr log1p(bfloat16 arg) {
    return functions::log1p(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr log1p(expr arg) {
    return functions::log1p(arg);
}

/// Binary logorithm.
/// \param arg function argument
/// \return logarithm of \a arg to base 2
//		template<typename T> typename enable<expr,T>::type log2(T arg) {
// return functions::log2(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr log2(bfloat16 arg) {
    return functions::log2(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr log2(expr arg) {
    return functions::log2(arg);
}

/// \}
/// \name Power functions
/// \{

/// Square root.
/// \param arg function argument
/// \return square root of \a arg
//		template<typename T> typename enable<expr,T>::type sqrt(T arg) {
// return functions::sqrt(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr sqrt(bfloat16 arg) {
    return functions::sqrt(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr sqrt(expr arg) {
    return functions::sqrt(arg);
}

/// Cubic root.
/// \param arg function argument
/// \return cubic root of \a arg
//		template<typename T> typename enable<expr,T>::type cbrt(T arg) {
// return functions::cbrt(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr cbrt(bfloat16 arg) {
    return functions::cbrt(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr cbrt(expr arg) {
    return functions::cbrt(arg);
}

/// Hypotenuse function.
/// \param x first argument
/// \param y second argument
/// \return square root of sum of squares without internal over- or
/// underflows
//		template<typename T,typename U> typename enable<expr,T,U>::type
// hypot(T x, U y) { return functions::hypot(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline expr hypot(bfloat16 x, bfloat16 y) {
    return functions::hypot(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr hypot(bfloat16 x, expr y) {
    return functions::hypot(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr hypot(expr x, bfloat16 y) {
    return functions::hypot(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr hypot(expr x, expr y) {
    return functions::hypot(x, y);
}

/// Power function.
/// \param base first argument
/// \param exp second argument
/// \return \a base raised to \a exp
//		template<typename T,typename U> typename enable<expr,T,U>::type
// pow(T base, U exp) { return functions::pow(base, exp); }
MEGDNN_HOST MEGDNN_DEVICE inline expr pow(bfloat16 base, bfloat16 exp) {
    return functions::pow(base, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr pow(bfloat16 base, expr exp) {
    return functions::pow(base, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr pow(expr base, bfloat16 exp) {
    return functions::pow(base, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr pow(expr base, expr exp) {
    return functions::pow(base, exp);
}

/// \}
/// \name Trigonometric functions
/// \{

/// Sine function.
/// \param arg function argument
/// \return sine value of \a arg
//		template<typename T> typename enable<expr,T>::type sin(T arg) {
// return functions::sin(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr sin(bfloat16 arg) {
    return functions::sin(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr sin(expr arg) {
    return functions::sin(arg);
}

/// Cosine function.
/// \param arg function argument
/// \return cosine value of \a arg
//		template<typename T> typename enable<expr,T>::type cos(T arg) {
// return functions::cos(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr cos(bfloat16 arg) {
    return functions::cos(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr cos(expr arg) {
    return functions::cos(arg);
}

/// Tangent function.
/// \param arg function argument
/// \return tangent value of \a arg
//		template<typename T> typename enable<expr,T>::type tan(T arg) {
// return functions::tan(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr tan(bfloat16 arg) {
    return functions::tan(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr tan(expr arg) {
    return functions::tan(arg);
}

/// Arc sine.
/// \param arg function argument
/// \return arc sine value of \a arg
//		template<typename T> typename enable<expr,T>::type asin(T arg) {
// return functions::asin(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr asin(bfloat16 arg) {
    return functions::asin(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr asin(expr arg) {
    return functions::asin(arg);
}

/// Arc cosine function.
/// \param arg function argument
/// \return arc cosine value of \a arg
//		template<typename T> typename enable<expr,T>::type acos(T arg) {
// return functions::acos(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr acos(bfloat16 arg) {
    return functions::acos(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr acos(expr arg) {
    return functions::acos(arg);
}

/// Arc tangent function.
/// \param arg function argument
/// \return arc tangent value of \a arg
//		template<typename T> typename enable<expr,T>::type atan(T arg) {
// return functions::atan(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr atan(bfloat16 arg) {
    return functions::atan(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr atan(expr arg) {
    return functions::atan(arg);
}

/// Arc tangent function.
/// \param x first argument
/// \param y second argument
/// \return arc tangent value
//		template<typename T,typename U> typename enable<expr,T,U>::type
// atan2(T x, U y) { return functions::atan2(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline expr atan2(bfloat16 x, bfloat16 y) {
    return functions::atan2(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr atan2(bfloat16 x, expr y) {
    return functions::atan2(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr atan2(expr x, bfloat16 y) {
    return functions::atan2(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr atan2(expr x, expr y) {
    return functions::atan2(x, y);
}

/// \}
/// \name Hyperbolic functions
/// \{

/// Hyperbolic sine.
/// \param arg function argument
/// \return hyperbolic sine value of \a arg
//		template<typename T> typename enable<expr,T>::type sinh(T arg) {
// return functions::sinh(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr sinh(bfloat16 arg) {
    return functions::sinh(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr sinh(expr arg) {
    return functions::sinh(arg);
}

/// Hyperbolic cosine.
/// \param arg function argument
/// \return hyperbolic cosine value of \a arg
//		template<typename T> typename enable<expr,T>::type cosh(T arg) {
// return functions::cosh(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr cosh(bfloat16 arg) {
    return functions::cosh(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr cosh(expr arg) {
    return functions::cosh(arg);
}

/// Hyperbolic tangent.
/// \param arg function argument
/// \return hyperbolic tangent value of \a arg
//		template<typename T> typename enable<expr,T>::type tanh(T arg) {
// return functions::tanh(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr tanh(bfloat16 arg) {
    return functions::tanh(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr tanh(expr arg) {
    return functions::tanh(arg);
}

/// Hyperbolic area sine.
/// \param arg function argument
/// \return area sine value of \a arg
//		template<typename T> typename enable<expr,T>::type asinh(T arg)
//{  return functions::asinh(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr asinh(bfloat16 arg) {
    return functions::asinh(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr asinh(expr arg) {
    return functions::asinh(arg);
}

/// Hyperbolic area cosine.
/// \param arg function argument
/// \return area cosine value of \a arg
//		template<typename T> typename enable<expr,T>::type acosh(T arg)
//{  return functions::acosh(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr acosh(bfloat16 arg) {
    return functions::acosh(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr acosh(expr arg) {
    return functions::acosh(arg);
}

/// Hyperbolic area tangent.
/// \param arg function argument
/// \return area tangent value of \a arg
//		template<typename T> typename enable<expr,T>::type atanh(T arg)
//{  return functions::atanh(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr atanh(bfloat16 arg) {
    return functions::atanh(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr atanh(expr arg) {
    return functions::atanh(arg);
}

/// \}
/// \name Error and gamma functions
/// \{

/// Error function.
/// \param arg function argument
/// \return error function value of \a arg
//		template<typename T> typename enable<expr,T>::type erf(T arg) {
// return functions::erf(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr erf(bfloat16 arg) {
    return functions::erf(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr erf(expr arg) {
    return functions::erf(arg);
}

/// Complementary error function.
/// \param arg function argument
/// \return 1 minus error function value of \a arg
//		template<typename T> typename enable<expr,T>::type erfc(T arg) {
// return functions::erfc(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr erfc(bfloat16 arg) {
    return functions::erfc(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr erfc(expr arg) {
    return functions::erfc(arg);
}

/// Natural logarithm of gamma function.
/// \param arg function argument
/// \return natural logarith of gamma function for \a arg
//		template<typename T> typename enable<expr,T>::type lgamma(T arg)
//{  return functions::lgamma(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr lgamma(bfloat16 arg) {
    return functions::lgamma(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr lgamma(expr arg) {
    return functions::lgamma(arg);
}

/// Gamma function.
/// \param arg function argument
/// \return gamma function value of \a arg
//		template<typename T> typename enable<expr,T>::type tgamma(T arg)
//{  return functions::tgamma(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline expr tgamma(bfloat16 arg) {
    return functions::tgamma(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline expr tgamma(expr arg) {
    return functions::tgamma(arg);
}

/// \}
/// \name Rounding
/// \{

/// Nearest integer not less than bfloat16 value.
/// \param arg bfloat16 to round
/// \return nearest integer not less than \a arg
//		template<typename T> typename enable<bfloat16,T>::type ceil(T
// arg) { return functions::ceil(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 ceil(bfloat16 arg) {
    return functions::ceil(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 ceil(expr arg) {
    return functions::ceil(arg);
}

/// Nearest integer not greater than bfloat16 value.
/// \param arg bfloat16 to round
/// \return nearest integer not greater than \a arg
//		template<typename T> typename enable<bfloat16,T>::type floor(T
// arg) { return functions::floor(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 floor(bfloat16 arg) {
    return functions::floor(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 floor(expr arg) {
    return functions::floor(arg);
}

/// Nearest integer not greater in magnitude than bfloat16 value.
/// \param arg bfloat16 to round
/// \return nearest integer not greater in magnitude than \a arg
//		template<typename T> typename enable<bfloat16,T>::type trunc(T
// arg) { return functions::trunc(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 trunc(bfloat16 arg) {
    return functions::trunc(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 trunc(expr arg) {
    return functions::trunc(arg);
}

/// Nearest integer.
/// \param arg bfloat16 to round
/// \return nearest integer, rounded away from zero in bfloat16-way cases
//		template<typename T> typename enable<bfloat16,T>::type round(T
// arg) { return functions::round(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 round(bfloat16 arg) {
    return functions::round(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 round(expr arg) {
    return functions::round(arg);
}

/// Nearest integer using bfloat16's internal rounding mode.
/// \param arg bfloat16 expression to round
/// \return nearest integer using default rounding mode
//		template<typename T> typename enable<bfloat16,T>::type
// nearbyint(T  arg) { return functions::nearbyint(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nearbyint(bfloat16 arg) {
    return functions::rint(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nearbyint(expr arg) {
    return functions::rint(arg);
}

/// Nearest integer using bfloat16's internal rounding mode.
/// \param arg bfloat16 expression to round
/// \return nearest integer using default rounding mode
//		template<typename T> typename enable<bfloat16,T>::type rint(T
// arg) { return functions::rint(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 rint(bfloat16 arg) {
    return functions::rint(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 rint(expr arg) {
    return functions::rint(arg);
}

/// Nearest integer using bfloat16's internal rounding mode.
/// \param arg bfloat16 expression to round
/// \return nearest integer using default rounding mode
//		template<typename T> typename enable<long,T>::type lrint(T arg)
//{  return functions::lrint(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline long lrint(bfloat16 arg) {
    return functions::lrint(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline long lrint(expr arg) {
    return functions::lrint(arg);
}
#if HALF_ENABLE_CPP11_LONG_LONG
/// Nearest integer using bfloat16's internal rounding mode.
/// \param arg bfloat16 expression to round
/// \return nearest integer using default rounding mode
//		template<typename T> typename enable<long long,T>::type llrint(T
// arg) { return functions::llrint(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline long long llrint(bfloat16 arg) {
    return functions::llrint(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline long long llrint(expr arg) {
    return functions::llrint(arg);
}
#endif

/// \}
/// \name Floating point manipulation
/// \{

/// Decompress floating point number.
/// \param arg number to decompress
/// \param exp address to store exponent at
/// \return significant in range [0.5, 1)
//		template<typename T> typename enable<bfloat16,T>::type frexp(T
// arg, int *exp) { return functions::frexp(arg, exp); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 frexp(bfloat16 arg, int* exp) {
    return functions::frexp(arg, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 frexp(expr arg, int* exp) {
    return functions::frexp(arg, exp);
}

/// Multiply by power of two.
/// \param arg number to modify
/// \param exp power of two to multiply with
/// \return \a arg multplied by 2 raised to \a exp
//		template<typename T> typename enable<bfloat16,T>::type ldexp(T
// arg, int exp) { return functions::scalbln(arg, exp); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 ldexp(bfloat16 arg, int exp) {
    return functions::scalbln(arg, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 ldexp(expr arg, int exp) {
    return functions::scalbln(arg, exp);
}

/// Extract integer and fractional parts.
/// \param arg number to decompress
/// \param iptr address to store integer part at
/// \return fractional part
//		template<typename T> typename enable<bfloat16,T>::type modf(T
// arg,  bfloat16 *iptr) { return functions::modf(arg, iptr); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 modf(bfloat16 arg, bfloat16* iptr) {
    return functions::modf(arg, iptr);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 modf(expr arg, bfloat16* iptr) {
    return functions::modf(arg, iptr);
}

/// Multiply by power of two.
/// \param arg number to modify
/// \param exp power of two to multiply with
/// \return \a arg multplied by 2 raised to \a exp
//		template<typename T> typename enable<bfloat16,T>::type scalbn(T
// arg, int exp) { return functions::scalbln(arg, exp); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 scalbn(bfloat16 arg, int exp) {
    return functions::scalbln(arg, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 scalbn(expr arg, int exp) {
    return functions::scalbln(arg, exp);
}

/// Multiply by power of two.
/// \param arg number to modify
/// \param exp power of two to multiply with
/// \return \a arg multplied by 2 raised to \a exp
//		template<typename T> typename enable<bfloat16,T>::type scalbln(T
// arg, long exp) { return functions::scalbln(arg, exp); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 scalbln(bfloat16 arg, long exp) {
    return functions::scalbln(arg, exp);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 scalbln(expr arg, long exp) {
    return functions::scalbln(arg, exp);
}

/// Extract exponent.
/// \param arg number to query
/// \return floating point exponent
/// \retval FP_ILOGB0 for zero
/// \retval FP_ILOGBNAN for NaN
/// \retval MAX_INT for infinity
//		template<typename T> typename enable<int,T>::type ilogb(T arg) {
// return functions::ilogb(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline int ilogb(bfloat16 arg) {
    return functions::ilogb(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline int ilogb(expr arg) {
    return functions::ilogb(arg);
}

/// Extract exponent.
/// \param arg number to query
/// \return floating point exponent
//		template<typename T> typename enable<bfloat16,T>::type logb(T
// arg) { return functions::logb(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 logb(bfloat16 arg) {
    return functions::logb(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 logb(expr arg) {
    return functions::logb(arg);
}

/// Next representable value.
/// \param from value to compute next representable value for
/// \param to direction towards which to compute next value
/// \return next representable value after \a from in direction towards \a
/// to
//		template<typename T,typename U> typename
// enable<bfloat16,T,U>::type nextafter(T from, U to) { return
// functions::nextafter(from, to); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nextafter(bfloat16 from,
                                                    bfloat16 to) {
    return functions::nextafter(from, to);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nextafter(bfloat16 from, expr to) {
    return functions::nextafter(from, to);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nextafter(expr from, bfloat16 to) {
    return functions::nextafter(from, to);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 nextafter(expr from, expr to) {
    return functions::nextafter(from, to);
}

/// Take sign.
/// \param x value to change sign for
/// \param y value to take sign from
/// \return value equal to \a x in magnitude and to \a y in sign
//		template<typename T,typename U> typename
// enable<bfloat16,T,U>::type copysign(T x, U y) { return
// functions::copysign(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 copysign(bfloat16 x, bfloat16 y) {
    return functions::copysign(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 copysign(bfloat16 x, expr y) {
    return functions::copysign(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 copysign(expr x, bfloat16 y) {
    return functions::copysign(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bfloat16 copysign(expr x, expr y) {
    return functions::copysign(x, y);
}

/// \}
/// \name Floating point classification
/// \{

/// Classify floating point value.
/// \param arg number to classify
/// \retval FP_ZERO for positive and negative zero
/// \retval FP_SUBNORMAL for subnormal numbers
/// \retval FP_INFINITY for positive and negative infinity
/// \retval FP_NAN for NaNs
/// \retval FP_NORMAL for all other (normal) values
//		template<typename T> typename enable<int,T>::type fpclassify(T
// arg) { return functions::fpclassify(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline int fpclassify(bfloat16 arg) {
    return functions::fpclassify(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline int fpclassify(expr arg) {
    return functions::fpclassify(arg);
}

/// Check if finite number.
/// \param arg number to check
/// \retval true if neither infinity nor NaN
/// \retval false else
//		template<typename T> typename enable<bool,T>::type isfinite(T
// arg) { return functions::isfinite(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isfinite(bfloat16 arg) {
    return functions::isfinite(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isfinite(expr arg) {
    return functions::isfinite(arg);
}

/// Check for infinity.
/// \param arg number to check
/// \retval true for positive or negative infinity
/// \retval false else
//		template<typename T> typename enable<bool,T>::type isinf(T arg)
//{  return functions::isinf(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isinf(bfloat16 arg) {
    return functions::isinf(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isinf(expr arg) {
    return functions::isinf(arg);
}

/// Check for NaN.
/// \param arg number to check
/// \retval true for NaNs
/// \retval false else
//		template<typename T> typename enable<bool,T>::type isnan(T arg)
//{  return functions::isnan(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isnan(bfloat16 arg) {
    return functions::isnan(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isnan(expr arg) {
    return functions::isnan(arg);
}

/// Check if normal number.
/// \param arg number to check
/// \retval true if normal number
/// \retval false if either subnormal, zero, infinity or NaN
//		template<typename T> typename enable<bool,T>::type isnormal(T
// arg) { return functions::isnormal(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isnormal(bfloat16 arg) {
    return functions::isnormal(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isnormal(expr arg) {
    return functions::isnormal(arg);
}

/// Check sign.
/// \param arg number to check
/// \retval true for negative number
/// \retval false for positive number
//		template<typename T> typename enable<bool,T>::type signbit(T
// arg) { return functions::signbit(arg); }
MEGDNN_HOST MEGDNN_DEVICE inline bool signbit(bfloat16 arg) {
    return functions::signbit(arg);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool signbit(expr arg) {
    return functions::signbit(arg);
}

/// \}
/// \name Comparison
/// \{

/// Comparison for greater than.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x greater than \a y
/// \retval false else
//		template<typename T,typename U> typename enable<bool,T,U>::type
// isgreater(T x, U y) { return functions::isgreater(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreater(bfloat16 x, bfloat16 y) {
    return functions::isgreater(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreater(bfloat16 x, expr y) {
    return functions::isgreater(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreater(expr x, bfloat16 y) {
    return functions::isgreater(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreater(expr x, expr y) {
    return functions::isgreater(x, y);
}

/// Comparison for greater equal.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x greater equal \a y
/// \retval false else
//		template<typename T,typename U> typename enable<bool,T,U>::type
// isgreaterequal(T x, U y) { return functions::isgreaterequal(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreaterequal(bfloat16 x, bfloat16 y) {
    return functions::isgreaterequal(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreaterequal(bfloat16 x, expr y) {
    return functions::isgreaterequal(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreaterequal(expr x, bfloat16 y) {
    return functions::isgreaterequal(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isgreaterequal(expr x, expr y) {
    return functions::isgreaterequal(x, y);
}

/// Comparison for less than.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x less than \a y
/// \retval false else
//		template<typename T,typename U> typename enable<bool,T,U>::type
// isless(T x, U y) { return functions::isless(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isless(bfloat16 x, bfloat16 y) {
    return functions::isless(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isless(bfloat16 x, expr y) {
    return functions::isless(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isless(expr x, bfloat16 y) {
    return functions::isless(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isless(expr x, expr y) {
    return functions::isless(x, y);
}

/// Comparison for less equal.
/// \param x first operand
/// \param y second operand
/// \retval true if \a x less equal \a y
/// \retval false else
//		template<typename T,typename U> typename enable<bool,T,U>::type
// islessequal(T x, U y) { return functions::islessequal(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bool islessequal(bfloat16 x, bfloat16 y) {
    return functions::islessequal(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool islessequal(bfloat16 x, expr y) {
    return functions::islessequal(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool islessequal(expr x, bfloat16 y) {
    return functions::islessequal(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool islessequal(expr x, expr y) {
    return functions::islessequal(x, y);
}

/// Comarison for less or greater.
/// \param x first operand
/// \param y second operand
/// \retval true if either less or greater
/// \retval false else
//		template<typename T,typename U> typename enable<bool,T,U>::type
// islessgreater(T x, U y) { return functions::islessgreater(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bool islessgreater(bfloat16 x, bfloat16 y) {
    return functions::islessgreater(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool islessgreater(bfloat16 x, expr y) {
    return functions::islessgreater(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool islessgreater(expr x, bfloat16 y) {
    return functions::islessgreater(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool islessgreater(expr x, expr y) {
    return functions::islessgreater(x, y);
}

/// Check if unordered.
/// \param x first operand
/// \param y second operand
/// \retval true if unordered (one or two NaN operands)
/// \retval false else
//		template<typename T,typename U> typename enable<bool,T,U>::type
// isunordered(T x, U y) { return functions::isunordered(x, y); }
MEGDNN_HOST MEGDNN_DEVICE inline bool isunordered(bfloat16 x, bfloat16 y) {
    return functions::isunordered(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isunordered(bfloat16 x, expr y) {
    return functions::isunordered(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isunordered(expr x, bfloat16 y) {
    return functions::isunordered(x, y);
}
MEGDNN_HOST MEGDNN_DEVICE inline bool isunordered(expr x, expr y) {
    return functions::isunordered(x, y);
}

/// \name Casting
/// \{

/// Cast to or from bfloat16-precision floating point number.
/// This casts between [bfloat16](\ref bfloat16_float::bfloat16) and any
/// built-in arithmetic type. Floating point types are converted via an
/// explicit cast to/from `float` (using the rounding mode of the built-in
/// single precision implementation) and thus any possible warnings due to
/// an otherwise implicit conversion to/from `float` will be suppressed.
/// Integer types are converted directly using the given rounding mode,
/// without any roundtrip over `float` that a `static_cast` would otherwise
/// do. It uses the default rounding mode.
///
/// Using this cast with neither of the two types being a [bfloat16](\ref
/// bfloat16_float::bfloat16) or with any of the two types not being a
/// built-in arithmetic type (apart from [bfloat16](\ref
/// bfloat16_float::bfloat16), of course) results in a compiler error and
/// casting between [bfloat16](\ref bfloat16_float::bfloat16)s is just a
/// no-op. \tparam T destination type (bfloat16 or built-in arithmetic type)
/// \tparam U source type (bfloat16 or built-in arithmetic type) \param arg
/// value to cast \return \a arg converted to destination type
template <typename T, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename bfloat16_caster<T, U>::type bfloat16_cast(
        U arg) {
    return bfloat16_caster<T, U>::cast(arg);
}

/// Cast to or from bfloat16-precision floating point number.
/// This casts between [bfloat16](\ref bfloat16_float::bfloat16) and any
/// built-in arithmetic type. Floating point types are converted via an
/// explicit cast to/from `float` (using the rounding mode of the built-in
/// single precision implementation) and thus any possible warnings due to
/// an otherwise implicit conversion to/from `float` will be suppressed.
/// Integer types are converted directly using the given rounding mode,
/// without any roundtrip over `float` that a `static_cast` would otherwise
/// do.
///
/// Using this cast with neither of the two types being a [bfloat16](\ref
/// bfloat16_float::bfloat16) or with any of the two types not being a
/// built-in arithmetic type (apart from [bfloat16](\ref
/// bfloat16_float::bfloat16), of course) results in a compiler error and
/// casting between [bfloat16](\ref bfloat16_float::bfloat16)s is just a
/// no-op. \tparam T destination type (bfloat16 or built-in arithmetic type)
/// \tparam R rounding mode to use. \tparam U source type (bfloat16 or
/// built-in arithmetic type) \param arg value to cast \return \a arg
/// converted to destination type
template <typename T, std::float_round_style R, typename U>
MEGDNN_HOST MEGDNN_DEVICE typename bfloat16_caster<T, U, R>::type bfloat16_cast(
        U arg) {
    return bfloat16_caster<T, U, R>::cast(arg);
}
/// \}
}  // namespace detail

using detail::operator==;
using detail::operator!=;
using detail::operator<;
using detail::operator>;
using detail::operator<=;
using detail::operator>=;
using detail::operator+;
using detail::operator-;
using detail::operator*;
using detail::operator/;
using detail::operator<<;
using detail::operator>>;

using detail::abs;
using detail::acos;
using detail::acosh;
using detail::asin;
using detail::asinh;
using detail::atan;
using detail::atan2;
using detail::atanh;
using detail::cbrt;
using detail::ceil;
using detail::cos;
using detail::cosh;
using detail::erf;
using detail::erfc;
using detail::exp;
using detail::exp2;
using detail::expm1;
using detail::fabs;
using detail::fdim;
using detail::floor;
using detail::fma;
using detail::fmax;
using detail::fmin;
using detail::fmod;
using detail::hypot;
using detail::lgamma;
using detail::log;
using detail::log10;
using detail::log1p;
using detail::log2;
using detail::lrint;
using detail::nanh;
using detail::nearbyint;
using detail::pow;
using detail::remainder;
using detail::rint;
using detail::round;
using detail::sin;
using detail::sinh;
using detail::sqrt;
using detail::tan;
using detail::tanh;
using detail::tgamma;
using detail::trunc;
#if HALF_ENABLE_CPP11_LONG_LONG
using detail::llrint;
#endif
using detail::copysign;
using detail::fpclassify;
using detail::frexp;
using detail::ilogb;
using detail::isfinite;
using detail::isgreater;
using detail::isgreaterequal;
using detail::isinf;
using detail::isless;
using detail::islessequal;
using detail::islessgreater;
using detail::isnan;
using detail::isnormal;
using detail::isunordered;
using detail::ldexp;
using detail::logb;
using detail::modf;
using detail::nextafter;
using detail::scalbln;
using detail::scalbn;
using detail::signbit;

using detail::bfloat16_cast;
}  // namespace half_bfloat16

/// Extensions to the C++ standard library.
namespace std {
/// Numeric limits for bfloat16-precision floats.
/// Because of the underlying single-precision implementation of many
/// operations, it inherits some properties from `numeric_limits<float>`.
#if !defined(__HIPCC__)
template <>
class numeric_limits<half_bfloat16::bfloat16> : public numeric_limits<float> {
public:
    /// Supports signed values.
    static HALF_CONSTEXPR_CONST bool is_signed = true;

    /// Is not exact.
    static HALF_CONSTEXPR_CONST bool is_exact = false;

    /// Doesn't provide modulo arithmetic.
    static HALF_CONSTEXPR_CONST bool is_modulo = false;

    /// IEEE conformant.
    static HALF_CONSTEXPR_CONST bool is_iec559 = false;

    /// Supports infinity.
    static HALF_CONSTEXPR_CONST bool has_infinity = true;

    /// Supports quiet NaNs.
    static HALF_CONSTEXPR_CONST bool has_quiet_NaN = true;

    /// Supports subnormal values.
    static HALF_CONSTEXPR_CONST float_denorm_style has_denorm = denorm_present;

    /// Rounding mode.
    /// Due to the mix of internal single-precision computations (using the
    /// rounding mode of the underlying single-precision implementation) with
    /// explicit truncation of the single-to-bfloat16 conversions, the actual
    /// rounding mode is indeterminate.
    static HALF_CONSTEXPR_CONST float_round_style round_style =
            (numeric_limits<float>::round_style ==
             half_bfloat16::bfloat16::round_style)
                    ? half_bfloat16::bfloat16::round_style
                    : round_indeterminate;

    /// Significant digits.
    static HALF_CONSTEXPR_CONST int digits = 8;

    /// Significant decimal digits.
    static HALF_CONSTEXPR_CONST int digits10 = 2;

    /// Required decimal digits to represent all possible values.
    static HALF_CONSTEXPR_CONST int max_digits10 = 4;

    /// Number base.
    static HALF_CONSTEXPR_CONST int radix = 2;

    /// One more than smallest exponent.
    static HALF_CONSTEXPR_CONST int min_exponent = -125;

    /// Smallest normalized representable power of 10.
    static HALF_CONSTEXPR_CONST int min_exponent10 = -37;

    /// One more than largest exponent
    static HALF_CONSTEXPR_CONST int max_exponent = 128;

    /// Largest finitely representable power of 10.
    static HALF_CONSTEXPR_CONST int max_exponent10 = 38;

    /// Smallest positive normal value.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    min() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x0080);
    }

    /// Smallest finite value.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    lowest() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0xFF7F);
    }

    /// Largest finite value.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    max() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x7F7F);
    }

    /// Difference between one and next representable value.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    epsilon() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x3C00);
    }

    /// Maximum rounding error.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    round_error() HALF_NOTHROW {
        return half_bfloat16::bfloat16(
                half_bfloat16::detail::binary_t(),
                (round_style == round_to_nearest) ? 0x3F00 : 0x3F80);
    }

    /// Positive infinity.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    infinity() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x7F80);
    }

    /// Quiet NaN.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    quiet_NaN() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x7FFF);
    }

    /// Signalling NaN.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    signaling_NaN() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x7FBF);
    }

    /// Smallest positive subnormal value.
    MEGDNN_HOST MEGDNN_DEVICE static HALF_CONSTEXPR half_bfloat16::bfloat16
    denorm_min() HALF_NOTHROW {
        return half_bfloat16::bfloat16(half_bfloat16::detail::binary_t(),
                                       0x0001);
    }
};
#endif

#ifdef MEGDNN_CC_HOST
#if HALF_ENABLE_CPP11_HASH
/// Hash function for bfloat16-precision floats.
/// This is only defined if C++11 `hash` is supported and enabled.
template <>
struct hash<half_bfloat16::bfloat16>
{
    /// Type of function argument.
    typedef half_bfloat16::bfloat16 argument_type;

    /// Function return type.
    typedef size_t result_type;

    /// Compute hash function.
    /// \param arg bfloat16 to hash
    /// \return hash value
    MEGDNN_HOST MEGDNN_DEVICE result_type operator()(argument_type arg) const {
        return hash<half_bfloat16::detail::uint16>()(
                static_cast<unsigned int>(arg.data_) & -(arg.data_ != 0x8000));
    }
};
#endif
#endif
}  // namespace std

#include "megdnn/dtype/half_common_epilogue.h"

#endif

// vim: syntax=cpp.doxygen
