/**
 * \file dnn/src/cuda/relayout_format/helper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

namespace megdnn {
namespace cuda {
namespace relayout_format {

#define devfunc __forceinline__ __device__
template <int size_nbits>
devfunc int make_zero(int zero_point);

template <>
devfunc int make_zero<4>(int zero_point) {
    return transform_int8_to_uint4x8(zero_point, zero_point, zero_point,
                                     zero_point, zero_point, zero_point,
                                     zero_point, zero_point);
}

template <typename AccessType, int LoadBytes>
struct global_load_with_zero_point;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

// The redundant mov PTX instruction is used to enforce the compiler to
// initialize data to zero before ld.global
template <typename AccessType>
struct global_load_with_zero_point<AccessType, 32> {
    devfunc global_load_with_zero_point(AccessType& D, void const* ptr,
                                        bool pred_guard, int zero_point) {
        uint4* data = reinterpret_cast<uint4*>(&D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %9, 0;\n"
                "  mov.b32 %0, %10;\n"
                "  mov.b32 %1, %10;\n"
                "  mov.b32 %2, %10;\n"
                "  mov.b32 %3, %10;\n"
                "  mov.b32 %4, %10;\n"
                "  mov.b32 %5, %10;\n"
                "  mov.b32 %6, %10;\n"
                "  mov.b32 %7, %10;\n"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%8];\n"
                "  @p ld.global.v4.u32 {%4, %5, %6, %7}, [%11];\n"
                "}\n"
                : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z),
                  "=r"(data[0].w), "=r"(data[1].x), "=r"(data[1].y),
                  "=r"(data[1].z), "=r"(data[1].w)
                : "l"(ptr), "r"((int)pred_guard),
                  "r"(reinterpret_cast<unsigned&>(zero_point)),
                  "l"(((uint8_t*)ptr) + 16));
    }
};

template <typename AccessType>
struct global_load_with_zero_point<AccessType, 16> {
    devfunc global_load_with_zero_point(AccessType& D, void const* ptr,
                                        bool pred_guard, int zero_point) {
        uint4& data = reinterpret_cast<uint4&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %5, 0;\n"
                "  mov.b32 %0, %6;\n"
                "  mov.b32 %1, %6;\n"
                "  mov.b32 %2, %6;\n"
                "  mov.b32 %3, %6;\n"
                "  @p ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                "}\n"
                : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
                : "l"(ptr), "r"((int)pred_guard),
                  "r"(reinterpret_cast<unsigned&>(zero_point)));
    }
};

template <typename AccessType>
struct global_load_with_zero_point<AccessType, 8> {
    devfunc global_load_with_zero_point(AccessType& D, void const* ptr,
                                        bool pred_guard, int zero_point) {
        uint2& data = reinterpret_cast<uint2&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %3, 0;\n"
                "  mov.b32 %0, %4;\n"
                "  mov.b32 %1, %4;\n"
                "  @p ld.global.v2.u32 {%0, %1}, [%2];\n"
                "}\n"
                : "=r"(data.x), "=r"(data.y)
                : "l"(ptr), "r"((int)pred_guard),
                  "r"(reinterpret_cast<unsigned&>(zero_point)));
    }
};

template <typename AccessType>
struct global_load_with_zero_point<AccessType, 4> {
    devfunc global_load_with_zero_point(AccessType& D, void const* ptr,
                                        bool pred_guard, int zero_point) {
        unsigned& data = reinterpret_cast<unsigned&>(D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  mov.b32 %0, %3;\n"
                "  @p ld.global.u32 %0, [%1];\n"
                "}\n"
                : "=r"(data)
                : "l"(ptr), "r"((int)pred_guard),
                  "r"(reinterpret_cast<unsigned&>(zero_point)));
    }
};

template <typename AccessType>
struct global_load_with_zero_point<AccessType, 1> {
    devfunc global_load_with_zero_point(AccessType& D, void const* ptr,
                                        bool pred_guard, int zero_point) {
        if (pred_guard)
            D = *(reinterpret_cast<AccessType const*>(ptr));
        else {
            unsigned uv = reinterpret_cast<unsigned&>(zero_point);
            uint8_t& data = reinterpret_cast<uint8_t&>(D);
            data = uv & 0xff;
        }
    }
};

#undef devfunc
}  // namespace relayout_format
}  // namespace cuda
}  // namespace megdnn
