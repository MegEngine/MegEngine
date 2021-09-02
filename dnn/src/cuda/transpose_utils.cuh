/**
 * \file dnn/src/cuda/memory_utils.cuh
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

MEGDNN_DEVICE __forceinline__ void transpose_int8_4x4_impl(
        const int src0, const int src1, const int src2, const int src3,
        int& dst0, int& dst1, int& dst2, int& dst3) {
    int dst01_lo = __byte_perm(src0, src1, 0x5140);
    int dst01_hi = __byte_perm(src0, src1, 0x7362);
    int dst23_lo = __byte_perm(src2, src3, 0x5140);
    int dst23_hi = __byte_perm(src2, src3, 0x7362);
    dst0 = __byte_perm(dst01_lo, dst23_lo, 0x5410);
    dst1 = __byte_perm(dst01_lo, dst23_lo, 0x7632);
    dst2 = __byte_perm(dst01_hi, dst23_hi, 0x5410);
    dst3 = __byte_perm(dst01_hi, dst23_hi, 0x7632);
}

template <uint32_t interleaved, typename vec_type>
struct transpose_int8_interleavedx4;

template <>
struct transpose_int8_interleavedx4<4, int> {
    static constexpr uint32_t interleaved = 4;
    using vec_type = int;
    using Fragment = array_wrapper<int, interleaved>;
    MEGDNN_DEVICE __forceinline__ void operator()(const Fragment src,
                                                  vec_type (&dst)[4]) {
        transpose_int8_4x4_impl(src[0], src[1], src[2], src[3], dst[0], dst[1],
                                dst[2], dst[3]);
    }
};

template <>
struct transpose_int8_interleavedx4<8, int2> {
    static constexpr uint32_t interleaved = 8;
    using vec_type = int2;
    using Fragment = array_wrapper<int, interleaved>;
    MEGDNN_DEVICE __forceinline__ void operator()(const Fragment src,
                                                  vec_type (&dst)[4]) {
        transpose_int8_4x4_impl(src[0], src[1], src[2], src[3], dst[0].x,
                                dst[1].x, dst[2].x, dst[3].x);
        transpose_int8_4x4_impl(src[4], src[5], src[6], src[7], dst[0].y,
                                dst[1].y, dst[2].y, dst[3].y);
    }
};

template <>
struct transpose_int8_interleavedx4<16, int4> {
    static constexpr uint32_t interleaved = 16;
    using vec_type = int4;
    using Fragment = array_wrapper<int, interleaved>;
    MEGDNN_DEVICE __forceinline__ void operator()(const Fragment src,
                                                  vec_type (&dst)[4]) {
        transpose_int8_4x4_impl(src[0], src[1], src[2], src[3], dst[0].x,
                                dst[1].x, dst[2].x, dst[3].x);
        transpose_int8_4x4_impl(src[4], src[5], src[6], src[7], dst[0].y,
                                dst[1].y, dst[2].y, dst[3].y);
        transpose_int8_4x4_impl(src[8], src[9], src[10], src[11], dst[0].z,
                                dst[1].z, dst[2].z, dst[3].z);
        transpose_int8_4x4_impl(src[12], src[13], src[14], src[15], dst[0].w,
                                dst[1].w, dst[2].w, dst[3].w);
    }
};

}  // namespace cuda
}  // namespace megdnn
#endif

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
