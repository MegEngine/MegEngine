#pragma once
/**
 * \file dnn/src/arm_common/conv_bias/intrinsic_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/conv_bias/neon_struct.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/fallback/conv_bias/common.h"
namespace megdnn {
namespace {

////////////////////Store_OC4_OW8_Remain/////////////////////////
template <int ow_remain, typename Op>
struct Store_OC4_OW8_Remain {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr);
};

template <typename Op>
struct Store_OC4_OW8_Remain<0, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[4], c[5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
        op({{c[6], c[7]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 24));
    }
};

template <typename Op>
struct Store_OC4_OW8_Remain<7, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[4], c[5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
        op(c[6], reinterpret_cast<dt_qint8*>(dst_ptr + 24));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<6, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[4], c[5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<5, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op(c[4], reinterpret_cast<dt_qint8*>(dst_ptr + 16));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<4, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<3, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<2, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<1, Op> {
    static void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op(c[0], reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};

template <int ow_remain, typename Op>
inline void store_oc4_ow8_remain_static(int32x4_t c[8], const Op& op,
                                        int8_t* dst_ptr) {
    Store_OC4_OW8_Remain<ow_remain, Op>::impl(c, op, dst_ptr);
}

template <int c_dim, int ow_remain, typename Op, typename T>
struct StoreOcxOw4Remain {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc);
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 0, Op, T> {
    static void impl(int32x4_t c[2][4], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
    }
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 3, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[0][2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op(c[1][2], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 2, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 1, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[1][0], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 0, Op, T> {
    static void impl(int32x4_t c[2][4], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 3, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[0][2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 2, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 1, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <int c_dim, int ow_remain, typename Op, typename T>
inline void store_ocx_ow4_remain_static(T& c, const Op& op, int8_t* dst_ptr,
                                        int ld_dst_oc) {
    StoreOcxOw4Remain<c_dim, ow_remain, Op, T>::impl(c, op, dst_ptr, ld_dst_oc);
}

////////////////////Store_OC8_OW8_Remain/////////////////////////

template <int ow_remain, typename Op>
struct Store_OC8_OW8_Remain {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc);
};

template <typename Op>
struct Store_OC8_OW8_Remain<0, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 24));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 16));
        op({{c[1][6], c[1][7]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 24));
    }
};

template <typename Op>
struct Store_OC8_OW8_Remain<7, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
        op(c[0][6], reinterpret_cast<dt_qint8*>(dst_ptr + 24));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 16));
        op(c[1][6], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 24));
    }
};

template <typename Op>
struct Store_OC8_OW8_Remain<6, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 16));
    }
};

template <typename Op>
struct Store_OC8_OW8_Remain<5, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op(c[0][4], reinterpret_cast<dt_qint8*>(dst_ptr + 16));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
        op(c[1][4], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 16));
    }
};

template <typename Op>
struct Store_OC8_OW8_Remain<4, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
    }
};

template <typename Op>
struct Store_OC8_OW8_Remain<3, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[0][2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op(c[1][2], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op>
struct Store_OC8_OW8_Remain<2, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};
template <typename Op>
struct Store_OC8_OW8_Remain<1, Op> {
    static void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                     int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[1][0], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};

template <int ow_remain, typename Op>
inline void store_oc8_ow8_remain_static(int32x4_t c[2][8], const Op& op,
                                        int8_t* dst_ptr, int ld_dst_oc) {
    Store_OC8_OW8_Remain<ow_remain, Op>::impl(c, op, dst_ptr, ld_dst_oc);
}

///////////////////////////////////////////////////////

template <BiasMode bias_mode>
inline void init_oc4_ow8(int32x4_t c[8], const int32_t* bias_ptr) {
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
#define BAIS_INIT(step) c[step] = vld1q_s32(bias_ptr);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    } else {
#define BAIS_INIT(step) c[step] = vdupq_n_s32(0);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
}

template <BiasMode bias_mode>
inline void init_oc8_ow8(int32x4_t c[2][8], const int32_t* bias_ptr,
                         int oc_step) {
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
#define BAIS_INIT(step)               \
    c[0][step] = vld1q_s32(bias_ptr); \
    c[1][step] = vld1q_s32(bias_ptr + oc_step);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    } else {
#define BAIS_INIT(step)          \
    c[0][step] = vdupq_n_s32(0); \
    c[1][step] = vdupq_n_s32(0);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
}
template <int c_dim, BiasMode bias_mode, typename T>
struct InitOcxOw4 {
    static void impl(T& c, const int32_t* bias_ptr, int oc_step);
};

template <BiasMode bias_mode, typename T>
struct InitOcxOw4<2, bias_mode, T> {
    static void impl(T& c, const int32_t* bias_ptr, int oc_step) {
        if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
#define BAIS_INIT(step)               \
    c[0][step] = vld1q_s32(bias_ptr); \
    c[1][step] = vld1q_s32(bias_ptr + oc_step);
            UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
        } else {
#define BAIS_INIT(step)          \
    c[0][step] = vdupq_n_s32(0); \
    c[1][step] = vdupq_n_s32(0);
            UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
        }
    }
};

template <BiasMode bias_mode, typename T>
struct InitOcxOw4<1, bias_mode, T> {
    static void impl(T& c, const int32_t* bias_ptr, int oc_step) {
        if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
#define BAIS_INIT(step) c[0][step] = vld1q_s32(bias_ptr);
            UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
        } else {
#define BAIS_INIT(step) c[0][step] = vdupq_n_s32(0);
            UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
        }
    }
};

template <int c_dim, BiasMode bias_mode, typename T>
inline void init_ocx_ow4(T& c, const int32_t* bias_ptr, int oc_step) {
    InitOcxOw4<c_dim, bias_mode, T>::impl(c, bias_ptr, oc_step);
}
///////////////////////////////////////
template <int weight_number, int base_offset, int ptr_step, int oc_block,
          typename Func, typename T, typename... XT>
struct LoadHelper {
    static void impl(T& weight, const int8_t* ptr, int oc_offset, XT... args);
};

#define WEIGHT_CB(step) \
    src[step] = Func::impl(ptr + base_offset + step * ptr_step, args...);

template <int base_offset, int ptr_step, typename Func, typename T,
          typename... XT>
struct LoadHelper<1, base_offset, ptr_step, 0, Func, T, XT...> {
    static void impl(T& src, const int8_t* ptr, int oc_offset, XT... args) {
        UNROLL_CALL_RAW(1, WEIGHT_CB);
    }
};
template <int base_offset, int ptr_step, typename Func, typename T,
          typename... XT>
struct LoadHelper<2, base_offset, ptr_step, 0, Func, T, XT...> {
    static void impl(T& src, const int8_t* ptr, int oc_offset, XT... args) {
        UNROLL_CALL_RAW(2, WEIGHT_CB);
    }
};

template <int base_offset, int ptr_step, typename Func, typename T,
          typename... XT>
struct LoadHelper<3, base_offset, ptr_step, 0, Func, T, XT...> {
    static void impl(T& src, const int8_t* ptr, int oc_offset, XT... args) {
        UNROLL_CALL_RAW(3, WEIGHT_CB);
    }
};
template <int base_offset, int ptr_step, typename Func, typename T,
          typename... XT>
struct LoadHelper<4, base_offset, ptr_step, 0, Func, T, XT...> {
    static void impl(T& src, const int8_t* ptr, int oc_offset, XT... args) {
        UNROLL_CALL_RAW(4, WEIGHT_CB);
    }
};
template <int base_offset, int ptr_step, typename Func, typename T,
          typename... XT>
struct LoadHelper<5, base_offset, ptr_step, 0, Func, T, XT...> {
    static void impl(T& src, const int8_t* ptr, int oc_offset, XT... args) {
        UNROLL_CALL_RAW(5, WEIGHT_CB);
    }
};
template <int base_offset, int ptr_step, typename Func, typename T,
          typename... XT>
struct LoadHelper<6, base_offset, ptr_step, 0, Func, T, XT...> {
    static void impl(T& src, const int8_t* ptr, int oc_offset, XT... args) {
        UNROLL_CALL_RAW(6, WEIGHT_CB);
    }
};
#undef WEIGHT_CB

#define WEIGHT_CB(step) \
    src[0][step] = Func::impl(ptr + base_offset + step * ptr_step);
template <int base_offset, int ptr_step, typename Func, typename T>
struct LoadHelper<1, base_offset, ptr_step, 1, Func, T> {
    static void impl(T& src, const int8_t* ptr, int oc_offset) {
        UNROLL_CALL_RAW(1, WEIGHT_CB);
    }
};
template <int base_offset, int ptr_step, typename Func, typename T>
struct LoadHelper<2, base_offset, ptr_step, 1, Func, T> {
    static void impl(T& src, const int8_t* ptr, int oc_offset) {
        UNROLL_CALL_RAW(2, WEIGHT_CB);
    }
};

template <int base_offset, int ptr_step, typename Func, typename T>
struct LoadHelper<3, base_offset, ptr_step, 1, Func, T> {
    static void impl(T& src, const int8_t* ptr, int oc_offset) {
        UNROLL_CALL_RAW(3, WEIGHT_CB);
    }
};

#undef WEIGHT_CB

#define WEIGHT_CB(step)                                             \
    src[0][step] = Func::impl(ptr + base_offset + step * ptr_step); \
    src[1][step] = Func::impl(ptr + base_offset + step * ptr_step + oc_offset);

template <int base_offset, int ptr_step, typename Func, typename T>
struct LoadHelper<1, base_offset, ptr_step, 2, Func, T> {
    static void impl(T& src, const int8_t* ptr, int oc_offset) {
        UNROLL_CALL_RAW(1, WEIGHT_CB);
    }
};
template <int base_offset, int ptr_step, typename Func, typename T>
struct LoadHelper<2, base_offset, ptr_step, 2, Func, T> {
    static void impl(T& src, const int8_t* ptr, int oc_offset) {
        UNROLL_CALL_RAW(2, WEIGHT_CB);
    }
};

template <int base_offset, int ptr_step, typename Func, typename T>
struct LoadHelper<3, base_offset, ptr_step, 2, Func, T> {
    static void impl(T& src, const int8_t* ptr, int oc_offset) {
        UNROLL_CALL_RAW(3, WEIGHT_CB);
    }
};

#undef WEIGHT_CB

template <int weight_number, int base_offset, int ptr_step, int c_dim,
          typename Func, typename T>
inline void load_helper(T& weight, const int8_t* ptr, int oc_offset) {
    LoadHelper<weight_number, base_offset, ptr_step, c_dim, Func, T>::impl(
            weight, ptr, oc_offset);
}

template <int weight_number, int base_offset, int ptr_step, int c_dim,
          typename Func, typename T, typename... XT>
inline void load_helper_x(T& weight, const int8_t* ptr, int oc_offset,
                          XT... args) {
    LoadHelper<weight_number, base_offset, ptr_step, c_dim, Func, T,
               XT...>::impl(weight, ptr, oc_offset, args...);
}

}  // namespace
}  // namespace megdnn