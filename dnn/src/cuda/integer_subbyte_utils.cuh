/**
 * \file dnn/src/cuda/integer_subbyte_utils.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#if MEGDNN_CC_CUDA
#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace integer_subbyte {
template <bool signedness>
struct integer_trait;

template <>
struct integer_trait<true> {
    using type = int;
};

template <>
struct integer_trait<false> {
    using type = unsigned;
};

MEGDNN_DEVICE __forceinline__ static int transform_int8_to_int4x8(
        int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
    unsigned out;
#if __CUDA_ARCH__ >= 750 &&             \
        ((__CUDACC_VER_MAJOR__ > 10) || \
         ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))
    asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.s4.s32.b32    r4, %8, %7, 0;"
            "cvt.pack.sat.s4.s32.b32    r4, %6, %5, r4;"
            "cvt.pack.sat.s4.s32.b32    r4, %4, %3, r4;"
            "cvt.pack.sat.s4.s32.b32    %0, %2, %1, r4;"
            "}"
            : "=r"(out)
            : "r"(s0), "r"(s1), "r"(s2), "r"(s3), "r"(s4), "r"(s5), "r"(s6),
              "r"(s7));
#else
#define CVT_SAT_S4_S32(r, bits) \
    r = r <= -8 ? -8 : r;       \
    r = r > 7 ? 7 : r;          \
    r = (((unsigned)r & 0xf) << bits);
    CVT_SAT_S4_S32(s0, 0)
    CVT_SAT_S4_S32(s1, 4)
    CVT_SAT_S4_S32(s2, 8)
    CVT_SAT_S4_S32(s3, 12)
    CVT_SAT_S4_S32(s4, 16)
    CVT_SAT_S4_S32(s5, 20)
    CVT_SAT_S4_S32(s6, 24)
    CVT_SAT_S4_S32(s7, 28)
    out = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
#undef CVT_SAT_S4_S32
#endif
    return reinterpret_cast<int const&>(out);
}

MEGDNN_DEVICE __forceinline__ static int transform_int8_to_uint4x8(
        int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
    unsigned out;
#if __CUDA_ARCH__ >= 750 &&             \
        ((__CUDACC_VER_MAJOR__ > 10) || \
         ((__CUDACC_VER_MAJOR__ >= 10) && (__CUDACC_VER_MINOR__ >= 2)))
    asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.u4.s32.b32    r4, %8, %7, 0;"
            "cvt.pack.sat.u4.s32.b32    r4, %6, %5, r4;"
            "cvt.pack.sat.u4.s32.b32    r4, %4, %3, r4;"
            "cvt.pack.sat.u4.s32.b32    %0, %2, %1, r4;"
            "}"
            : "=r"(out)
            : "r"(s0), "r"(s1), "r"(s2), "r"(s3), "r"(s4), "r"(s5), "r"(s6),
              "r"(s7));
#else
#define CVT_SAT_U4_S32(r, bits) \
    r = r <= 0 ? 0 : r;         \
    r = r > 15 ? 15 : r;        \
    r = (((unsigned)r & 0xf) << bits);
    CVT_SAT_U4_S32(s0, 0)
    CVT_SAT_U4_S32(s1, 4)
    CVT_SAT_U4_S32(s2, 8)
    CVT_SAT_U4_S32(s3, 12)
    CVT_SAT_U4_S32(s4, 16)
    CVT_SAT_U4_S32(s5, 20)
    CVT_SAT_U4_S32(s6, 24)
    CVT_SAT_U4_S32(s7, 28)
    out = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
#undef CVT_SAT_U4_S32
#endif
    return reinterpret_cast<int const&>(out);
}

template <bool signedness, typename T>
MEGDNN_DEVICE __forceinline__ static int unpack_integer_4bits(T storage,
                                                              int bits) {
    //! size in bits of 32 bit integer - 4 bits
    static constexpr int shift = 28;
    using type = typename integer_trait<signedness>::type;
    unsigned intermediate = static_cast<unsigned>(storage);
    type result = reinterpret_cast<type&>(intermediate);
    return (result << (shift - bits)) >> shift;
}

MEGDNN_DEVICE __forceinline__ static void transform_int4x8_to_int8(
        int (&result)[8], const int& source) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = unpack_integer_4bits<true>(
                reinterpret_cast<unsigned const&>(source), (i << 2));
    }
}

MEGDNN_DEVICE __forceinline__ static void transform_uint4x8_to_int8(
        int (&result)[8], const int& source) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = unpack_integer_4bits<false>(
                reinterpret_cast<unsigned const&>(source), (i << 2));
    }
}

MEGDNN_DEVICE __forceinline__ static void transform_int4x2_to_int8(
        int (&result)[2], const uint8_t& source) {
    result[0] = unpack_integer_4bits<true>(source, 0);
    result[1] = unpack_integer_4bits<true>(source, 4);
}

MEGDNN_DEVICE __forceinline__ static void transform_uint4x2_to_int8(
        int (&result)[2], const uint8_t& source) {
    result[0] = unpack_integer_4bits<false>(source, 0);
    result[1] = unpack_integer_4bits<false>(source, 4);
}
}  // namespace integer_subbyte
}  // namespace cuda
}  // namespace megdnn
#endif
// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
