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
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};

template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 3, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
        op(c[0][2], reinterpret_cast<dt_qint8*>(dst_ptr + 8));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 2, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op({{c[0][0], c[0][1]}}, reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <typename Op, typename T>
struct StoreOcxOw4Remain<1, 1, Op, T> {
    static void impl(T& c, const Op& op, int8_t* dst_ptr, int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        op(c[0][0], reinterpret_cast<dt_qint8*>(dst_ptr));
    }
};
template <int c_dim, int ow_remain, typename Op, typename T>
inline void store_ocx_ow4_remain_static(T& c, const Op& op, int8_t* dst_ptr,
                                        int ld_dst_oc) {
    StoreOcxOw4Remain<c_dim, ow_remain, Op, T>::impl(c, op, dst_ptr, ld_dst_oc);
}
////////////////////Store_OCX_OW8_Remain/////////////////////////
template <int c_dim, int ow_remain, typename Op, typename T, typename T2,
          typename T3>
struct StoreOcxOw8Remain {
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc);
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 0, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
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
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
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
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
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
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
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
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
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
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op({{c[1][2], c[1][3]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 3, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op(c[0][2], reinterpret_cast<T3>(dst_ptr + 8));

        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
        op(c[1][2], reinterpret_cast<T3>(dst_ptr + ld_dst_oc + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 2, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[1][0], c[1][1]}}, reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<2, 1, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int ld_dst_oc) {
        op(c[0][0], reinterpret_cast<T3>(dst_ptr));
        op(c[1][0], reinterpret_cast<T3>(dst_ptr + ld_dst_oc));
    }
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 0, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<T3>(dst_ptr + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 8, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op({{c[0][6], c[0][7]}}, reinterpret_cast<T3>(dst_ptr + 24));
    }
};

template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 7, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
        op(c[0][6], reinterpret_cast<T3>(dst_ptr + 24));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 6, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op({{c[0][4], c[0][5]}}, reinterpret_cast<T3>(dst_ptr + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 5, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
        op(c[0][4], reinterpret_cast<T3>(dst_ptr + 16));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 4, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op({{c[0][2], c[0][3]}}, reinterpret_cast<T3>(dst_ptr + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 3, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
        op(c[0][2], reinterpret_cast<T3>(dst_ptr + 8));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 2, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op({{c[0][0], c[0][1]}}, reinterpret_cast<T3>(dst_ptr));
    }
};
template <typename Op, typename T, typename T2, typename T3>
struct StoreOcxOw8Remain<1, 1, Op, T, T2, T3> {
    static void impl(T& c, const Op& op, T2 dst_ptr, int) {
        op(c[0][0], reinterpret_cast<T3>(dst_ptr));
    }
};

template <int c_dim, int ow_remain, typename Op, typename T, typename T2>
inline void store_ocx_ow8_remain_static(T& c, const Op& op, T2 dst_ptr,
                                        int ld_dst_oc) {
    StoreOcxOw8Remain<c_dim, ow_remain, Op, T, T2, T2>::impl(c, op, dst_ptr,
                                                             ld_dst_oc);
}
template <int c_dim, int ow_remain, typename Op, typename T3, typename T,
          typename T2>
inline void store_ocx_ow8_remain_static_dt(T& c, const Op& op, T2 dst_ptr,
                                           int ld_dst_oc) {
    StoreOcxOw8Remain<c_dim, ow_remain, Op, T, T2, T3>::impl(c, op, dst_ptr,
                                                             ld_dst_oc);
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

///////////

template <int ow_remain, typename Op, typename T, typename T2>
inline void store_oc8_ow8_remain_static(T& c, const Op& op, T2 dst_ptr,
                                        int ld_dst_oc) {
    Store_OC8_OW8_Remain<ow_remain, Op>::impl(c, op, dst_ptr, ld_dst_oc);
}

//////////////////////////////////////
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

/////////////////////////init_ocx_ow8////////////////////

inline float32x4_t neon_vdupq_n(float val) {
    return vdupq_n_f32(val);
}

inline int32x4_t neon_vdupq_n(int val) {
    return vdupq_n_s32(val);
}
inline float32x4_t neon_vld1q(const float* ptr) {
    return vld1q_f32(ptr);
}

inline int32x4_t neon_vld1q(const int* ptr) {
    return vld1q_s32(ptr);
}

template <int c_dim, BiasMode bias_mode, int ow_block, typename T, typename T2>
struct InitOcxOw8 {
    static void impl(T& c, const T2* bias_ptr, int oc_step);
};
template <typename T, typename T2>
struct InitOcxOw8<2, BiasMode::NO_BIAS, 8, T, T2> {
    static void impl(T& c, const T2*, int) {
#define BAIS_INIT(step)                            \
    c[0][step] = neon_vdupq_n(static_cast<T2>(0)); \
    c[1][step] = neon_vdupq_n(static_cast<T2>(0));
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<2, BiasMode::NO_BIAS, 4, T, T2> {
    static void impl(T& c, const T2*, int) {
#define BAIS_INIT(step)                            \
    c[0][step] = neon_vdupq_n(static_cast<T2>(0)); \
    c[1][step] = neon_vdupq_n(static_cast<T2>(0));
        UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<2, BiasMode::BROADCAST_CHANNEL_BIAS, 8, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int oc_step) {
#define BAIS_INIT(step)                \
    c[0][step] = neon_vld1q(bias_ptr); \
    c[1][step] = neon_vld1q(bias_ptr + oc_step);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<2, BiasMode::BROADCAST_CHANNEL_BIAS, 4, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int oc_step) {
#define BAIS_INIT(step)                \
    c[0][step] = neon_vld1q(bias_ptr); \
    c[1][step] = neon_vld1q(bias_ptr + oc_step);
        UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<2, BiasMode::BIAS, 8, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int oc_step) {
        constexpr int simd_len = 4;
#define BAIS_INIT(step)                                  \
    c[0][step] = neon_vld1q(bias_ptr + step * simd_len); \
    c[1][step] = neon_vld1q(bias_ptr + oc_step + step * simd_len);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<2, BiasMode::BIAS, 4, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int oc_step) {
        constexpr int simd_len = 4;
#define BAIS_INIT(step)                                  \
    c[0][step] = neon_vld1q(bias_ptr + step * simd_len); \
    c[1][step] = neon_vld1q(bias_ptr + oc_step + step * simd_len);
        UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
    }
};

template <typename T, typename T2>
struct InitOcxOw8<1, BiasMode::NO_BIAS, 8, T, T2> {
    static void impl(T& c, const T2*, int) {
#define BAIS_INIT(step) c[0][step] = neon_vdupq_n(static_cast<T2>(0));
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<1, BiasMode::NO_BIAS, 4, T, T2> {
    static void impl(T& c, const T2*, int) {
#define BAIS_INIT(step) c[0][step] = neon_vdupq_n(static_cast<T2>(0));
        UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<1, BiasMode::BROADCAST_CHANNEL_BIAS, 8, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int) {
#define BAIS_INIT(step) c[0][step] = neon_vld1q(bias_ptr);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<1, BiasMode::BROADCAST_CHANNEL_BIAS, 4, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int) {
#define BAIS_INIT(step) c[0][step] = neon_vld1q(bias_ptr);
        UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<1, BiasMode::BIAS, 8, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int) {
        constexpr int simd_len = 4;
#define BAIS_INIT(step) c[0][step] = neon_vld1q(bias_ptr + step * simd_len);
        UNROLL_CALL_RAW(8, BAIS_INIT);
#undef BAIS_INIT
    }
};
template <typename T, typename T2>
struct InitOcxOw8<1, BiasMode::BIAS, 4, T, T2> {
    static void impl(T& c, const T2* bias_ptr, int) {
        constexpr int simd_len = 4;
#define BAIS_INIT(step) c[0][step] = neon_vld1q(bias_ptr + step * simd_len);
        UNROLL_CALL_RAW(4, BAIS_INIT);
#undef BAIS_INIT
    }
};

template <int c_dim, BiasMode bias_mode, int ow_block, typename T, typename T2>
inline void init_ocx_ow8(T& c, const T2* bias_ptr, int oc_step) {
    InitOcxOw8<c_dim, bias_mode, ow_block, T, T2>::impl(c, bias_ptr, oc_step);
}
/////////////////////init_ocx_ow4/////////////////////
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
inline void init_ocx_ow4(T& c, const int32_t* bias_ptr, int oc_step) {
    InitOcxOw4<c_dim, bias_mode, T>::impl(c, bias_ptr, oc_step);
}
///////////////////////////////////////

}  // namespace
}  // namespace megdnn

// vim: syntax=cpp.doxygen
