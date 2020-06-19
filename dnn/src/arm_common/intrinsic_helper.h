/**
 * \file dnn/src/arm_common/intrinsic_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/arm_common/neon_struct.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#define __ai inline __attribute__((__always_inline__))
namespace megdnn {
namespace {

template <int weight_number, int base_offset, int ptr_step, int oc_block,
          typename Func, typename T, typename T2, typename... XT>
struct LoadHelper {
    static __ai void impl(T& weight, T2 ptr, int oc_offset, XT... args);
};

#define WEIGHT_CB(step) \
    src[step] = Func::impl(ptr + base_offset + step * ptr_step, args...);

#define LOAD_HELPER(step)                                                   \
    template <int base_offset, int ptr_step, typename Func, typename T,     \
              typename T2, typename... XT>                                  \
    struct LoadHelper<step, base_offset, ptr_step, 0, Func, T, T2, XT...> { \
        static __ai void impl(T& src, T2 ptr, int, XT... args) {            \
            UNROLL_CALL_RAW(step, WEIGHT_CB);                               \
        }                                                                   \
    }

LOAD_HELPER(1);
LOAD_HELPER(2);
LOAD_HELPER(3);
LOAD_HELPER(4);
LOAD_HELPER(5);
LOAD_HELPER(6);
LOAD_HELPER(7);
LOAD_HELPER(8);
LOAD_HELPER(9);
LOAD_HELPER(10);
LOAD_HELPER(11);
LOAD_HELPER(12);
LOAD_HELPER(13);
LOAD_HELPER(14);
LOAD_HELPER(15);
LOAD_HELPER(16);

#undef LOAD_HELPER
#undef WEIGHT_CB

///////////////////////////c_dim = 1/////////////////////////
#define WEIGHT_CB(step) \
    src[0][step] = Func::impl(ptr + base_offset + step * ptr_step);

#define LOAD_HELPER(step)                                               \
    template <int base_offset, int ptr_step, typename Func, typename T, \
              typename T2>                                              \
    struct LoadHelper<step, base_offset, ptr_step, 1, Func, T, T2> {    \
        static __ai void impl(T& src, T2 ptr, int) {                    \
            UNROLL_CALL_RAW(step, WEIGHT_CB);                           \
        }                                                               \
    }

LOAD_HELPER(1);
LOAD_HELPER(2);
LOAD_HELPER(3);
LOAD_HELPER(4);
LOAD_HELPER(5);
LOAD_HELPER(6);
LOAD_HELPER(7);
LOAD_HELPER(8);
LOAD_HELPER(9);

#undef LOAD_HELPER
#undef WEIGHT_CB

/////////////////////////c_dim = 2///////////////////////////////
#define WEIGHT_CB(step)                                             \
    src[0][step] = Func::impl(ptr + base_offset + step * ptr_step); \
    src[1][step] = Func::impl(ptr + base_offset + step * ptr_step + oc_offset);

#define LOAD_HELPER(step)                                               \
    template <int base_offset, int ptr_step, typename Func, typename T, \
              typename T2>                                              \
    struct LoadHelper<step, base_offset, ptr_step, 2, Func, T, T2> {    \
        static __ai void impl(T& src, T2 ptr, int oc_offset) {          \
            UNROLL_CALL_RAW(step, WEIGHT_CB);                           \
        }                                                               \
    }

LOAD_HELPER(1);
LOAD_HELPER(2);
LOAD_HELPER(3);
LOAD_HELPER(4);
LOAD_HELPER(5);
LOAD_HELPER(6);
LOAD_HELPER(7);
LOAD_HELPER(8);

#undef LOAD_HELPER
#undef WEIGHT_CB

template <int weight_number, int base_offset, int ptr_step, int c_dim,
          typename Func, typename T, typename T2>
__ai void load_helper(T& weight, T2 ptr, int oc_offset) {
    LoadHelper<weight_number, base_offset, ptr_step, c_dim, Func, T, T2>::impl(
            weight, ptr, oc_offset);
}

template <int weight_number, int base_offset, int ptr_step, int c_dim,
          typename Func, typename T, typename T2, typename... XT>
__ai void load_helper_x(T& weight, T2 ptr, int oc_offset, XT... args) {
    LoadHelper<weight_number, base_offset, ptr_step, c_dim, Func, T, T2,
               XT...>::impl(weight, ptr, oc_offset, args...);
}

}  // namespace
}  // namespace megdnn
#undef __ai
// vim: syntax=cpp.doxygen