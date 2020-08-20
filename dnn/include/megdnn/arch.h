/**
 * \file dnn/include/megdnn/arch.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

// include general build configurations
#include "megdnn/config/config.h"

#if defined(__GNUC__) || defined(__clang__)
 #if !defined (__clang__)
  // gcc specific
  #define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
  #if GCC_VERSION < 40800
   #error "GCC version should be at least 4.8.0."
  #endif // GCC_VERSION < 40800
 #endif // !defined(__clang__)

 #ifndef megdnn_trap
 #define megdnn_trap() __builtin_trap()
 #endif

 #define megdnn_likely(v) __builtin_expect(bool(v), 1)
 #define megdnn_unlikely(v) __builtin_expect(bool(v), 0)

#if !defined(__clang__) && MEGDNN_ARMV7 && !defined(NDEBUG)
//! Thumb2 limit code length
#define MEGDNN_ALWAYS_INLINE
#else
#define MEGDNN_ALWAYS_INLINE inline __attribute__((__always_inline__))
#endif

 #define MEGDNN_DEPRECATED __attribute__((deprecated))
 #define MEGDNN_PACKED __attribute__((packed))
 #define MEGDNN_CONSTEXPR constexpr
 #define MEGDNN_NOEXCEPT noexcept
 #define MEGDNN_STATIC_ASSERT static_assert
 #define MEGDNN_FINAL final
 #define MEGDNN_NORETURN __attribute__((noreturn))
 #define MEGDNN_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
 #define MEGDNN_ATTRIBUTE_TARGET(simd) __attribute__((target(simd)))
 #if defined(__clang_major__) && (__clang_major__ >= 7)
   #define MEGDNN_LAMBDA_ATTRIBUTE_TARGET(simd) __attribute__((target(simd)))
 #else
   #define MEGDNN_LAMBDA_ATTRIBUTE_TARGET(simd) [[gnu::target(simd)]]
 #endif
 #define MEGDNN_NOINLINE __attribute__((noinline))

 #define megdnn_isatty(x) isatty(x)
#elif defined(__INTEL_COMPILER) || defined(_MSC_VER)

#ifndef megdnn_trap
#define megdnn_trap() __debugbreak()
#endif

#define megdnn_likely(v) (bool(v))
#define megdnn_unlikely(v) (bool(v))

#define MEGDNN_DEPRECATED
#define MEGDNN_PACKED
#define MEGDNN_CONSTEXPR constexpr
#define MEGDNN_NOEXCEPT noexcept
#define MEGDNN_STATIC_ASSERT static_assert
#define MEGDNN_FINAL final

#if defined(_MSC_VER)
 #define MEGDNN_NORETURN __declspec(noreturn)
 #define MEGDNN_NOINLINE __declspec(noinline)
#else
 #define MEGDNN_NORETURN
 #define MEGDNN_FORCE_NOINLINE
#endif // _MSC_VER

#define MEGDNN_WARN_UNUSED_RESULT

#define megdnn_isatty(x) _isatty(x)

#else
  #error "unknown compiler"
#endif // __GNUC__

// __cpp_exceptions and __cpp_rtti is referred from
// https://isocpp.org/std/standing-documentssd-6-sg10-feature-test-recommendations
// gcc < 5 does not define __cpp_exceptions but __EXCEPTIONS, 
// similar for __GXX_RTTI
// _CPPUNWIND and _CPPRTTI is used by MSVC, see
// https://docs.microsoft.com/en-us/cpp/preprocessor/predefined-macrosview=vs-2019
#ifndef MEGDNN_ENABLE_EXCEPTIONS
 #if __cpp_exceptions || __EXCEPTIONS || \
     (defined(_MSC_VER) && defined(_CPPUNWIND))
  #define MEGDNN_ENABLE_EXCEPTIONS 1
 #else
  #define MEGDNN_ENABLE_EXCEPTIONS 0
 #endif
#endif
#ifndef MEGDNN_ENABLE_RTTI
 #if __cpp_rtti || __GXX_RTTI || (defined(_MSC_VER) && defined(_CPPRTTI))
  #define MEGDNN_ENABLE_RTTI 1
 #else
  #define MEGDNN_ENABLE_RTTI 0
 #endif
#endif

#ifdef __CUDACC__
 #define MEGDNN_CC_CUDA      1
 #undef MEGDNN_CONSTEXPR
 #define MEGDNN_CONSTEXPR  const

#if defined(__CUDACC_VER_MAJOR__)
#if __CUDACC_VER_MAJOR__ >= 9
 #undef MEGDNN_STATIC_ASSERT
 #define MEGDNN_STATIC_ASSERT(cond, msg) static_assert(cond, msg);
#else
 #undef MEGDNN_STATIC_ASSERT
 #define MEGDNN_STATIC_ASSERT(cond, msg)
#endif
#endif

 #define nullptr NULL
 #undef MEGDNN_FINAL
 #define MEGDNN_FINAL
#elif defined(__HIPCC__)
 #define MEGDNN_CC_CUDA 1
#else
 #define MEGDNN_CC_HOST      1
#endif // __CUDACC__

// MEGDNN_HOST and MEGDNN_DEVICE
#if MEGDNN_CC_CUDA
 #define MEGDNN_HOST __host__
 #define MEGDNN_DEVICE __device__
#else
 #define MEGDNN_HOST
 #define MEGDNN_DEVICE
#endif

#if defined(_MSC_VER) || defined(WIN32)
 #define ATTR_ALIGNED(v) __declspec(align(v))
#else
 #define ATTR_ALIGNED(v) __attribute__((aligned(v)))
#endif
// vim: syntax=cpp.doxygen
