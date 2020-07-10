/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw44_common.h
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
#if __ARM_FEATURE_DOTPROD
#include "megdnn/arch.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/intrinsic_helper.h"
#include "src/arm_common/neon_struct.h"
#include "src/common/unroll_macro.h"

namespace megdnn {
namespace arm_common {
namespace direct_dotprod_nchw44 {

constexpr int SIMD_LEN = 16;
constexpr int IC_PACK_SIZE = 4;
constexpr int OC_PACK_SIZE = 4;
constexpr int filter_next_col =
        IC_PACK_SIZE * OC_PACK_SIZE;  //! [OC/4, IC/4, FH, FW, 4OC, 4IC]

template <int row, BiasMode bias_mode>
MEGDNN_ALWAYS_INLINE void init_ocx_ow8(int32x4_t c[][8],
                                       const int32_t* bias_ptr, int oc_step) {
    static_assert(row == 1 || row == 2 || row == 3, "Invalid OC number.");
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
#define BIAS_INIT(step, i) c[i][step] = vld1q_s32(bias_ptr + i * oc_step);
        switch (row) {
            case 3:
                UNROLL_CALL_RAW(8, BIAS_INIT, 2);
            case 2:
                UNROLL_CALL_RAW(8, BIAS_INIT, 1);
            default:
                UNROLL_CALL_RAW(8, BIAS_INIT, 0);
        }
#undef BIAS_INIT
    } else {
#define BIAS_INIT(step, i) c[i][step] = vdupq_n_s32(0);
        switch (row) {
            case 3:
                UNROLL_CALL_RAW(8, BIAS_INIT, 2);
            case 2:
                UNROLL_CALL_RAW(8, BIAS_INIT, 1);
            default:
                UNROLL_CALL_RAW(8, BIAS_INIT, 0);
        }
#undef BIAS_INIT
    }
}

#define cb11(col) \
    op(res[0][col], reinterpret_cast<dt_qint8*>(dst_ptr + col / 2 * 8));

#define cb21(col)                                                        \
    op(res[0][col], reinterpret_cast<dt_qint8*>(dst_ptr + col / 2 * 8)); \
    op(res[1][col],                                                      \
       reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + col / 2 * 8));

#define cb31(col)                                                        \
    op(res[0][col], reinterpret_cast<dt_qint8*>(dst_ptr + col / 2 * 8)); \
    op(res[1][col],                                                      \
       reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + col / 2 * 8));  \
    op(res[2][col], reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc +    \
                                                ld_dst_oc + col / 2 * 8));

#define cb12(step)                                 \
    op({{res[0][2 * step], res[0][2 * step + 1]}}, \
       reinterpret_cast<dt_qint8*>(dst_ptr + step * 8));

#define cb22(step)                                       \
    op({{res[0][2 * step], res[0][2 * step + 1]}},       \
       reinterpret_cast<dt_qint8*>(dst_ptr + step * 8)); \
    op({{res[1][2 * step], res[1][2 * step + 1]}},       \
       reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + step * 8));

#define cb32(step)                                                   \
    op({{res[0][2 * step], res[0][2 * step + 1]}},                   \
       reinterpret_cast<dt_qint8*>(dst_ptr + step * 8));             \
    op({{res[1][2 * step], res[1][2 * step + 1]}},                   \
       reinterpret_cast<dt_qint8*>(dst_ptr + ld_dst_oc + step * 8)); \
    op({{res[2][2 * step], res[2][2 * step + 1]}},                   \
       reinterpret_cast<dt_qint8*>(dst_ptr + 2 * ld_dst_oc + step * 8));

template <int row, int ow_remain, typename Op, typename T>
struct StoreOCxOWx {
    static MEGDNN_ALWAYS_INLINE void impl(int32x4_t res[][8], const Op& op,
                                          T* dst_ptr, const int ld_dst_oc);
};

template <int ow_remain, typename Op, typename T>
struct StoreOCxOWx<1, ow_remain, Op, T> {
    static void impl(int32x4_t res[][8], const Op& op, T* dst_ptr,
                     const int ld_dst_oc) {
        MEGDNN_MARK_USED_VAR(ld_dst_oc);
        switch (ow_remain) {
            case 8:
                UNROLL_CALL_RAW(4, cb12);
                break;
            case 7:
                cb11(6);
            case 6:
                UNROLL_CALL_RAW(3, cb12);
                break;
            case 5:
                cb11(4);
            case 4:
                UNROLL_CALL_RAW(2, cb12);
                break;
            case 3:
                cb11(2);
            case 2:
                UNROLL_CALL_RAW(1, cb12);
                break;
            case 1:
                cb11(0);
            default:
                break;
        }
    }
};

template <int ow_remain, typename Op, typename T>
struct StoreOCxOWx<2, ow_remain, Op, T> {
    static MEGDNN_ALWAYS_INLINE void impl(int32x4_t res[][8], const Op& op,
                                          T* dst_ptr, const int ld_dst_oc) {
        switch (ow_remain) {
            case 8:
                UNROLL_CALL_RAW(4, cb22);
                break;
            case 7:
                cb21(6);
            case 6:
                UNROLL_CALL_RAW(3, cb22);
                break;
            case 5:
                cb21(4);
            case 4:
                UNROLL_CALL_RAW(2, cb22);
                break;
            case 3:
                cb21(2);
            case 2:
                UNROLL_CALL_RAW(1, cb22);
                break;
            case 1:
                cb21(0);
            default:
                break;
        }
    }
};

template <int ow_remain, typename Op, typename T>
struct StoreOCxOWx<3, ow_remain, Op, T> {
    static MEGDNN_ALWAYS_INLINE void impl(int32x4_t res[][8], const Op& op,
                                          T* dst_ptr, const int ld_dst_oc) {
        switch (ow_remain) {
            case 8:
                UNROLL_CALL_RAW(4, cb32);
                break;
            case 7:
                cb31(6);
            case 6:
                UNROLL_CALL_RAW(3, cb32);
                break;
            case 5:
                cb31(4);
            case 4:
                UNROLL_CALL_RAW(2, cb32);
                break;
            case 3:
                cb31(2);
            case 2:
                UNROLL_CALL_RAW(1, cb32);
                break;
            case 1:
                cb31(0);
            default:
                break;
        }
    }
};

#undef cb11
#undef cb21
#undef cb31
#undef cb12
#undef cb22
#undef cb32

template <int row, int ow_remain, typename Op, typename T>
MEGDNN_ALWAYS_INLINE void store_ocx_owx_remain_static(int32x4_t res[][8],
                                                      const Op& op, T* dst_ptr,
                                                      const int ld_dst_oc) {
    StoreOCxOWx<row, ow_remain, Op, T>::impl(res, op, dst_ptr, ld_dst_oc);
}

template <int res_row, int src_row, int src_start_idx, int weight_idx,
          typename T, typename T2, typename T3>
struct ShiftCalHelper {
    static MEGDNN_ALWAYS_INLINE void impl(T& res, T2& src, T3& weight) {
#define cb(step)                                                      \
    res[res_row][step] =                                              \
            vdotq_laneq_s32(res[res_row][step], weight[weight_idx],   \
                            src[src_row][(src_start_idx + step) / 4], \
                            (src_start_idx + step) % 4);
        UNROLL_CALL_RAW(8, cb);
#undef cb
    }
};

template <int res_row, int src_row, int src_start_idx, int weight_idx,
          typename T, typename T2, typename T3>
MEGDNN_ALWAYS_INLINE void cal_helper(T& res, T2& src, T3& weight) {
    ShiftCalHelper<res_row, src_row, src_start_idx, weight_idx, T, T2,
                   T3>::impl(res, src, weight);
};

/**
 *  oc12_owx(m = 12, n = x) and oc8_owx(m = 8, n = x) and oc4_owx(m = 4, n = x)
 * gemm like kernel
 * */
template <typename dst_type, int stride, BiasMode bias_mode, typename Op,
          int ow_remain, int filter_size, int oc_interval, int ow_interval>
struct KernNeonSdotNCHW44 {
    static void impl(dst_type* dst, const int dst_step, const int8_t* src,
                     const int ih, const int iw, const int8_t* filter,
                     const int32_t* bias, const int ic, const Op& op);
};

}  // namespace direct_dotprod_nchw44
}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen