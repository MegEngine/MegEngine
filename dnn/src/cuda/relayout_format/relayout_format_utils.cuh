/**
 * \file dnn/src/cuda/relayout_format/relayout_format_utils.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/cuda/integer_subbyte_utils.cuh"
#include "src/cuda/relayout_format/relayout_format.cuh"

namespace megdnn {
namespace cuda {
namespace relayout_format {
namespace internal {
template <typename cype, int pack_w, typename enable = void>
struct DTypeRWHelper;
template <typename ctype>
struct DTypeRWHelper<
        ctype, 1,
        typename std::enable_if<
                std::is_same<ctype, dt_qint8>::value ||
                std::is_same<ctype, dt_quint8>::value ||
                std::is_same<ctype, dt_uint8>::value>::type> {
    using InnerDtype = char;
    using DstDtype = char4;
};

template <typename ctype>
struct DTypeRWHelper<
        ctype, 4,
        typename std::enable_if<
                std::is_same<ctype, dt_qint8>::value ||
                std::is_same<ctype, dt_quint8>::value ||
                std::is_same<ctype, dt_uint8>::value>::type> {
    using InnerDtype = char4;
    using DstDtype = char4;
};

template <>
struct DTypeRWHelper<dt_qint32, 1> {
    using InnerDtype = int;
    using DstDtype = int4;
};

template <>
struct DTypeRWHelper<dt_qint32, 4> {
    using InnerDtype = int4;
    using DstDtype = int4;
};

template <typename ctype>
struct DTypeRWHelper<
        ctype, 2,
        typename std::enable_if<
                std::is_same<ctype, dt_qint4>::value ||
                std::is_same<ctype, dt_quint4>::value>::type> {
    using InnerDtype = char;
    using DstDtype = array_wrapper<uint8_t, 32>;
};

template <typename ctype>
struct DTypeRWHelper<
        ctype, 8,
        typename std::enable_if<
                std::is_same<ctype, dt_qint4>::value ||
                std::is_same<ctype, dt_quint4>::value>::type> {
    using InnerDtype = unsigned;
    using DstDtype = array_wrapper<uint8_t, 32>;
};

template <typename DstType>
inline __device__ DstType make_zero_pad(const uint8_t zero_point) {
    return zero_point;
}

template <>
inline __device__ char4 make_zero_pad<char4>(const uint8_t zero_point) {
    char izp = reinterpret_cast<const char&>(zero_point);
    return {izp, izp, izp, izp};
}

template <>
inline __device__ int4 make_zero_pad<int4>(const uint8_t zero_point) {
    return {zero_point, zero_point, zero_point, zero_point};
}

template <int size_nbits>
inline __device__ int make_zero(int zero_point);

template <>
inline __device__ int make_zero<4>(int zero_point) {
    return integer_subbyte::transform_int8_to_uint4x8(
            zero_point, zero_point, zero_point, zero_point, zero_point, zero_point,
            zero_point, zero_point);
}

template <typename DstDtype>
inline __device__ void write_helper(DstDtype* ptr, DstDtype val) {
    *ptr = val;
}

template <>
inline __device__ void write_helper<char4>(char4* ptr, char4 val) {
    int32_t* rel_ptr = (int32_t*)ptr;
    *rel_ptr = *(int32_t*)(&val);
}

template <>
inline __device__ void write_helper<array_wrapper<uint8_t, 32>>(
        array_wrapper<uint8_t, 32>* ptr, array_wrapper<uint8_t, 32> val) {
    uint4 const* data = reinterpret_cast<uint4 const*>(&val);
    void* ptr_ = reinterpret_cast<void*>(ptr);
    asm volatile(
            "{\n"
            " st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
            " st.global.v4.u32 [%5], {%6, %7, %8, %9};\n"
            "}\n"
            :
            : "l"(ptr_), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z), "r"(data[0].w),
              "l"(((uint8_t*)ptr_) + 16), "r"(data[1].x), "r"(data[1].y),
              "r"(data[1].z), "r"(data[1].w));
}

}  // namespace internal
}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn
