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
#pragma once
#include "src/arm_common/intrinsic_helper.h"
#include "src/arm_common/neon_struct.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/fallback/conv_bias/common.h"

#define __ai inline __attribute__((__always_inline__))
namespace megdnn {
namespace {

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"

#ifdef __GNUC__
#ifndef __has_warning
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#else
#if __has_warning("-Wmaybe-uninitialized")
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#endif
#endif
////////////////////Store_OC4_OW8_Remain/////////////////////////
template <int ow_remain, typename Op>
struct Store_OC4_OW8_Remain {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr);
};

template <typename Op>
struct Store_OC4_OW8_Remain<0, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[4], c[5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
        op({{c[6], c[7]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 24));
    }
};

template <typename Op>
struct Store_OC4_OW8_Remain<7, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[4], c[5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
        op(c[6], reinterpret_cast<dt_qint8*>(dst_ptr + 24));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<6, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op({{c[4], c[5]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 16));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<5, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
        op(c[4], reinterpret_cast<dt_qint8*>(dst_ptr + 16));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<4, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[2], c[3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<3, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<2, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op({{c[0], c[1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <typename Op>
struct Store_OC4_OW8_Remain<1, Op> {
    static __ai void impl(int32x4_t c[8], const Op& op, int8_t* dst_ptr) {
        op(c[0], reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};

template <int ow_remain, typename Op>
__ai void store_oc4_ow8_remain_static(int32x4_t c[8], const Op& op,
                                      int8_t* dst_ptr) {
    Store_OC4_OW8_Remain<ow_remain, Op>::impl(c, op, dst_ptr);
}

template <int c_dim, int ow_remain, typename Op, typename T>
struct StoreOcxOw4Remain {
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc);
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 0, Op, T> {
    static __ai void impl(int32x4_t c[2][4], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[0][2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
        op(c[1][2], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 2, Op, T> {
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<2, 1, Op, T> {
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[1][0], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 0, Op, T> {
    static __ai void impl(int32x4_t c[2][4], const Op& op, int8_t* dst_ptr,
                          int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 3, Op, T> {
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[0][2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 2, Op, T> {
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 1, Op, T> {
    static __ai void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <int c_dim, int ow_remain, typename Op, typename T>
__ai void store_ocx_ow4_remain_static(T& c, const Op& op, int8_t* dst_ptr,
                                      int ld_dst_oc) {
    StoreOcxOw4Remain<c_dim, ow_remain, Op, T>::impl(c, op, dst_ptr, ld_dst_oc);
}
////////////////////Store_OCX_OW8_Remain/////////////////////////
template <int c_dim, int ow_remain, typename Op, typename T, typename T2,
          typename T3>
struct StoreOcxOw8Remain {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc);
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 0, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<T3>(dst_ptr + 24));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op({{c[1][6], c[1][7]}},
           reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 8, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<T3>(dst_ptr + 24));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op({{c[1][6], c[1][7]}},
           reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 7, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op(c[0][6], reinterpret_cast<T3>(dst_ptr + 24));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
        op(c[1][6], reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 6, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op({{c[1][4], c[1][5]}},
           reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 5, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op(c[0][4], reinterpret_cast<T3>(dst_ptr + 16));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
        op(c[1][4], reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 4, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 3, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op(c[0][2], reinterpret_cast<T3>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(c[1][2], reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 2, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 1, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<T3>(dst_ptr));
        op(c[1][0], reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
    }
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 0, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<T3>(dst_ptr + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 8, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<T3>(dst_ptr + 24));
    }
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 7, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op(c[0][6], reinterpret_cast<T3>(dst_ptr + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 6, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 5, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op(c[0][4], reinterpret_cast<T3>(dst_ptr + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 4, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 3, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op(c[0][2], reinterpret_cast<T3>(dst_ptr + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 2, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 1, Op, T, T2, T3> {
    static __ai void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op(c[0][0], reinterpret_cast<T3>(dst_ptr));
    }
};

template <int c_dim, int ow_remain, typename Op, typename T, typename T2>
__ai void store_ocx_ow8_remain_static(T& c, const Op& op, T2 dst_ptr,
                                      int ld_dst_oc) {
    StoreOcxOw8Remain<c_dim, ow_remain, Op, T, T2, T2>::impl(c, op, dst_ptr,
                                                             ld_dst_oc);
}
template <int c_dim, int ow_remain, typename Op, typename T3, typename T,
          typename T2>
__ai void store_ocx_ow8_remain_static_dt(T& c, const Op& op, T2 dst_ptr,
                                         int ld_dst_oc) {
    StoreOcxOw8Remain<c_dim, ow_remain, Op, T, T2, T3>::impl(c, op, dst_ptr,
                                                             ld_dst_oc);
}
////////////////////Store_OCX_OW8_Remain/////////////////////////
template <int c_dim, int ow_block, int nr_group, int out_group, typename T,
          typename T2, typename T3>
struct StoreOc4Ow8Remain {
    static __ai void impl(T& c, T2 dst_ptr, int ld_dst_oc, const int ow_remain);
};

#define cb(step)                                               \
    vst1q_lane_s64((int64_t*)(dst_ptr + step * 4),             \
                   vreinterpretq_s64_s16(c[0][step]), 0);      \
    vst1q_lane_s64((int64_t*)(dst_ptr + step * 4 + ld_dst_oc), \
                   vreinterpretq_s64_s16(c[0][step]), 1);

#define cb2(step)                                  \
    vst1q_lane_s64((int64_t*)(dst_ptr + step * 4), \
                   vreinterpretq_s64_s16(c[0][step]), 0);

#define cb_case(step)              \
    case step:                     \
        UNROLL_CALL_RAW(step, cb); \
        break;

#define cb_case2(step)              \
    case step:                      \
        UNROLL_CALL_RAW(step, cb2); \
        break;
template <typename T, typename T2, typename T3>
struct StoreOc4Ow8Remain<1, 8, 2, 2, T, T2, T3> {
    static __ai void impl(T& c, T2 dst_ptr, int ld_dst_oc,
                          const int ow_remain) {
        if (ow_remain == 8) {
            UNROLL_CALL_RAW(8, cb)
        } else {
            switch (ow_remain) {
                cb_case(7);
                cb_case(6);
                cb_case(5);
                cb_case(4);
                cb_case(3);
                cb_case(2);
                cb_case(1);

                default:
                    break;
            }
        }
    }
};
template <typename T, typename T2, typename T3>
struct StoreOc4Ow8Remain<1, 8, 2, 1, T, T2, T3> {
    static __ai void impl(T& c, T2 dst_ptr, int, const int ow_remain) {
        if (ow_remain == 8) {
            UNROLL_CALL_RAW(8, cb2)
        } else {
            switch (ow_remain) {
                cb_case2(7);
                cb_case2(6);
                cb_case2(5);
                cb_case2(4);
                cb_case2(3);
                cb_case2(2);
                cb_case2(1);

                default:
                    break;
            }
        }
    }
};

#undef cb
#undef cb2
#undef cb_case
#undef cb_case2

template <int c_dim, int ow_block, int nr_group, int out_group, typename T,
          typename T2>
__ai void store_oc4_ow8_remain_static(T& c, T2 dst_ptr, const int ld_dst_oc,
                                      const int ow_remain) {
    StoreOc4Ow8Remain<c_dim, ow_block, nr_group, out_group, T, T2, T2>::impl(
            c, dst_ptr, ld_dst_oc, ow_remain);
}

////////////////////Store_OC8_OW8_Remain/////////////////////////

template <int ow_remain, typename Op>
struct Store_OC8_OW8_Remain {
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                          int ld_dst_oc);
};

template <typename Op>
struct Store_OC8_OW8_Remain<0, Op> {
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
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
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                          int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[1][0], c[1][1]}},
           reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};
template <typename Op>
struct Store_OC8_OW8_Remain<1, Op> {
    static __ai void impl(int32x4_t c[2][8], const Op& op, int8_t* dst_ptr,
                          int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[1][0], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc));
    }
};

///////////

template <int ow_remain, typename Op, typename T, typename T2>
__ai void store_oc8_ow8_remain_static(T& c, const Op& op, T2 dst_ptr,
                                      int ld_dst_oc) {
    Store_OC8_OW8_Remain<ow_remain, Op>::impl(c, op, dst_ptr, ld_dst_oc);
}
#pragma GCC diagnostic pop

//////////////////////////////////////
template <BiasMode bias_mode>
__ai void init_oc4_ow8(int32x4_t c[8], const int32_t* bias_ptr) {
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
__ai void init_oc8_ow8(int32x4_t c[2][8], const int32_t* bias_ptr,
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

/////////////////////////init_ocx_ow8////////////////////

__ai float32x4_t neon_vdupq_n(float val) {
    return vdupq_n_f32(val);
}

__ai int32x4_t neon_vdupq_n(int val) {
    return vdupq_n_s32(val);
}
__ai int16x8_t neon_vdupq_n(int16_t val) {
    return vdupq_n_s16(val);
}
__ai float32x4_t neon_vld1q(const float* ptr) {
    return vld1q_f32(ptr);
}
__ai int32x4_t neon_vld1q(const int* ptr) {
    return vld1q_s32(ptr);
}
__ai int16x8_t neon_vld1q(const int16_t* ptr) {
    return vld1q_s16(ptr);
}
template <typename T>
struct NeonLdqSimd;
template <>
struct NeonLdqSimd<float> {
    static constexpr int simd_len = 4;
};
template <>
struct NeonLdqSimd<int> {
    static constexpr int simd_len = 4;
};
template <>
struct NeonLdqSimd<int16_t> {
    static constexpr int simd_len = 8;
};
template <int c_dim, BiasMode bias_mode, int ow_remain, typename T, typename T2>
struct InitOcxOw8 {
    static __ai void impl(T& c, const T2* bias_ptr, int oc_step);
};
template <int c_dim, BiasMode bias_mode, typename T, typename T2>
struct InitOcxOw8<c_dim, bias_mode, 0, T, T2> {
    static __ai void impl(T&, const T2*, int) {}
};

#define BAIS_INIT_NO_BIAS_C2(step)                 \
    c[0][step] = neon_vdupq_n(static_cast<T2>(0)); \
    c[1][step] = neon_vdupq_n(static_cast<T2>(0));
#define BAIS_INIT_NO_BIAS_C1(step) \
    c[0][step] = neon_vdupq_n(static_cast<T2>(0));

#define BAIS_INIT_BROADCAST_C2(step)   \
    c[0][step] = neon_vld1q(bias_ptr); \
    c[1][step] = neon_vld1q(bias_ptr + oc_step);
#define BAIS_INIT_BROADCAST_C1(step) c[0][step] = neon_vld1q(bias_ptr);

#define BAIS_INIT_BIAS_C2(step)                          \
    c[0][step] = neon_vld1q(bias_ptr + step * simd_len); \
    c[1][step] = neon_vld1q(bias_ptr + oc_step + step * simd_len);

#define BAIS_INIT_BIAS_C1(step) \
    c[0][step] = neon_vld1q(bias_ptr + step * simd_len);

#define INSTANCE_InitOcxOw8(ow_remain, cdim)                                \
    template <typename T, typename T2>                                      \
    struct InitOcxOw8<cdim, BiasMode::NO_BIAS, ow_remain, T, T2> {          \
        static __ai void impl(T& c, const T2*, int) {                       \
            UNROLL_CALL_RAW(ow_remain, BAIS_INIT_NO_BIAS_C##cdim);          \
        }                                                                   \
    };                                                                      \
    template <typename T, typename T2>                                      \
    struct InitOcxOw8<cdim, BiasMode::BROADCAST_CHANNEL_BIAS, ow_remain, T, \
                      T2> {                                                 \
        static __ai void impl(T& c, const T2* bias_ptr, int oc_step) {      \
            (void)oc_step;                                                  \
            UNROLL_CALL_RAW(ow_remain, BAIS_INIT_BROADCAST_C##cdim);        \
        }                                                                   \
    };                                                                      \
    template <typename T, typename T2>                                      \
    struct InitOcxOw8<cdim, BiasMode::BIAS, ow_remain, T, T2> {             \
        static __ai void impl(T& c, const T2* bias_ptr, int oc_step) {      \
            constexpr int simd_len = NeonLdqSimd<T2>::simd_len;             \
            (void)oc_step;                                                  \
            UNROLL_CALL_RAW(ow_remain, BAIS_INIT_BIAS_C##cdim);             \
        }                                                                   \
    };
#define INSTANCE_InitOcxOw8_C(ow_remain) \
    INSTANCE_InitOcxOw8(ow_remain, 2);   \
    INSTANCE_InitOcxOw8(ow_remain, 1);

INSTANCE_InitOcxOw8_C(1);
INSTANCE_InitOcxOw8_C(2);
INSTANCE_InitOcxOw8_C(3);
INSTANCE_InitOcxOw8_C(4);
INSTANCE_InitOcxOw8_C(5);
INSTANCE_InitOcxOw8_C(6);
INSTANCE_InitOcxOw8_C(7);
INSTANCE_InitOcxOw8_C(8);

#undef INSTANCE_InitOcxOw8
#undef INSTANCE_InitOcxOw8_C
#undef BAIS_INIT_BIAS_C1
#undef BAIS_INIT_BIAS_C2
#undef BAIS_INIT_BROADCAST_C1
#undef BAIS_INIT_BROADCAST_C2
#undef BAIS_INIT_NO_BIAS_C1
#undef BAIS_INIT_NO_BIAS_C2

template <int c_dim, BiasMode bias_mode, int ow_remain, typename T, typename T2>
__ai void init_ocx_ow8(T& c, const T2* bias_ptr, int oc_step) {
    InitOcxOw8<c_dim, bias_mode, ow_remain, T, T2>::impl(c, bias_ptr, oc_step);
}
/////////////////////init_ocx_ow4/////////////////////
template <int c_dim, BiasMode bias_mode, typename T>
struct InitOcxOw4 {
    static __ai void impl(T& c, const int32_t* bias_ptr, int oc_step);
};

template <BiasMode bias_mode, typename T>
struct InitOcxOw4<2, bias_mode, T> {
    static __ai void impl(T& c, const int32_t* bias_ptr, int oc_step) {
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
    static __ai void impl(T& c, const int32_t* bias_ptr, int oc_step) {
        MEGDNN_MARK_USED_VAR(oc_step);
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
__ai void init_ocx_ow4(T& c, const int32_t* bias_ptr, int oc_step) {
    InitOcxOw4<c_dim, bias_mode, T>::impl(c, bias_ptr, oc_step);
}
///////////////////////////////////////

static inline void memcpy_s8_dup(int8_t* outptr, const int8_t* inptr,
                                 int count) {
    constexpr int expand = 8;
    for (; count >= 8; count -= 8) {
        int8x8_t in = vld1_s8(inptr);
        int8x8_t in0 = vdup_lane_s8(in, 0);
        int8x8_t in1 = vdup_lane_s8(in, 1);
        int8x8_t in2 = vdup_lane_s8(in, 2);
        int8x8_t in3 = vdup_lane_s8(in, 3);
        int8x8_t in4 = vdup_lane_s8(in, 4);
        int8x8_t in5 = vdup_lane_s8(in, 5);
        int8x8_t in6 = vdup_lane_s8(in, 6);
        int8x8_t in7 = vdup_lane_s8(in, 7);

        vst1_s8(outptr + 0 * 8, in0);
        vst1_s8(outptr + 1 * 8, in1);
        vst1_s8(outptr + 2 * 8, in2);
        vst1_s8(outptr + 3 * 8, in3);
        vst1_s8(outptr + 4 * 8, in4);
        vst1_s8(outptr + 5 * 8, in5);
        vst1_s8(outptr + 6 * 8, in6);
        vst1_s8(outptr + 7 * 8, in7);

        inptr += 8;
        outptr += 8 * expand;
    }
    for (; count > 0; --count) {
        int8x8_t in0 = vld1_dup_s8(inptr++);
        vst1_s8(outptr, in0);
        outptr += 1 * expand;
    }
}

}  // namespace
}  // namespace megdnn
#undef __ai
// vim: syntax=cpp.doxygen
