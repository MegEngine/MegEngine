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
namespace memory {

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename AccessType, int LoadBytes>
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

// The redundant mov PTX instruction is used to enforce the compiler to
// initialize data to zero before ld.global
template <typename AccessType>
struct global_load<AccessType, 32> {
    MEGDNN_DEVICE __forceinline__
    global_load(AccessType& D, void const* ptr, bool pred_guard, int val = 0) {
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
                : "=r"(data[0].x), "=r"(data[0].y), "=r"(data[0].z), "=r"(data[0].w),
                  "=r"(data[1].x), "=r"(data[1].y), "=r"(data[1].z), "=r"(data[1].w)
                : "l"(ptr), "r"((int)pred_guard), "r"(reinterpret_cast<unsigned&>(val)),
                  "l"(((uint8_t*)ptr) + 16));
    }
};

template <typename AccessType>
struct global_load<AccessType, 16> {
    MEGDNN_DEVICE __forceinline__
    global_load(AccessType& D, void const* ptr, bool pred_guard, int val) {
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
                  "r"(reinterpret_cast<unsigned&>(val)));
    }
};

template <typename AccessType>
struct global_load<AccessType, 8> {
    MEGDNN_DEVICE __forceinline__
    global_load(AccessType& D, void const* ptr, bool pred_guard, int val) {
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
                  "r"(reinterpret_cast<unsigned&>(val)));
    }
};

template <typename AccessType>
struct global_load<AccessType, 4> {
    MEGDNN_DEVICE __forceinline__
    global_load(AccessType& D, void const* ptr, bool pred_guard, int val) {
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
                  "r"(reinterpret_cast<unsigned&>(val)));
    }
};

template <typename AccessType>
struct global_load<AccessType, 1> {
    MEGDNN_DEVICE __forceinline__
    global_load(AccessType& D, void const* ptr, bool pred_guard, int val) {
        if (pred_guard)
            D = *(reinterpret_cast<AccessType const*>(ptr));
        else {
            unsigned uv = reinterpret_cast<unsigned&>(val);
            uint8_t& data = reinterpret_cast<uint8_t&>(D);
            data = uv & 0xff;
        }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
        /// Fragment type to store loaded data
        typename AccessType,
        /// The bytes of loading
        int LoadBytes>
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_store<AccessType, 32> {
    MEGDNN_DEVICE __forceinline__
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint4 const* data = reinterpret_cast<uint4 const*>(&D);

        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %5, 0;\n"
                "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                "  @p st.global.v4.u32 [%6], {%7, %8, %9, %10};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data[0].x), "r"(data[0].y), "r"(data[0].z),
                  "r"(data[0].w), "r"((int)pred_guard), "l"(((uint8_t*)ptr) + 16),
                  "r"(data[1].x), "r"(data[1].y), "r"(data[1].z), "r"(data[1].w));
    }
};

template <typename AccessType>
struct global_store<AccessType, 16> {
    MEGDNN_DEVICE __forceinline__
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint4 const& data = reinterpret_cast<uint4 const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %5, 0;\n"
                "  @p st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w),
                  "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 8> {
    MEGDNN_DEVICE __forceinline__
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint2 const& data = reinterpret_cast<uint2 const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %3, 0;\n"
                "  @p st.global.v2.u32 [%0], {%1, %2};\n"
                "}\n"
                :
                : "l"(ptr), "r"(data.x), "r"(data.y), "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 4> {
    MEGDNN_DEVICE __forceinline__
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint32_t const& data = reinterpret_cast<uint32_t const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  @p st.global.u32 [%0], %1;\n"
                "}\n"
                :
                : "l"(ptr), "r"(data), "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 2> {
    MEGDNN_DEVICE __forceinline__
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        uint16_t const& data = reinterpret_cast<uint16_t const&>(D);
        asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  setp.ne.b32 p, %2, 0;\n"
                "  @p st.global.u16 [%0], %1;\n"
                "}\n"
                :
                : "l"(ptr), "h"(data), "r"((int)pred_guard));
    }
};

template <typename AccessType>
struct global_store<AccessType, 1> {
    MEGDNN_DEVICE __forceinline__
    global_store(AccessType const& D, void* ptr, bool pred_guard) {
        if (pred_guard)
            *(reinterpret_cast<AccessType*>(ptr)) = D;
    }
};

}  // namespace memory
}  // namespace cuda
}  // namespace megdnn
#endif

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
