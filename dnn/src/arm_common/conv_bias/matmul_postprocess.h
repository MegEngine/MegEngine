/**
 * \file dnn/src/arm_common/conv_bias/matmul_postprocess.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {

#define SAVE(C, vres, n, idx)                                  \
    switch (n) {                                               \
        case 4:                                                \
            vst1_lane_s32(reinterpret_cast<int32_t*>(C),       \
                          vreinterpret_s32_s8(vres), idx / 4); \
            break;                                             \
        case 3:                                                \
            vst1_lane_s8(C + 2, vres, idx + 2); MEGDNN_FALLTHRU\
        case 2:                                                \
            vst1_lane_s8(C + 1, vres, idx + 1); MEGDNN_FALLTHRU\
        case 1:                                                \
            vst1_lane_s8(C + 0, vres, idx + 0);                \
            break;                                             \
        default:                                               \
            megdnn_assert(0);                                  \
    }

#define SAVEU(C, vres, n, idx)                                 \
    switch (n) {                                               \
        case 4:                                                \
            vst1_lane_s32(reinterpret_cast<int32_t*>(C),       \
                          vreinterpret_s32_u8(vres), idx / 4); \
            break;                                             \
        case 3:                                                \
            vst1_lane_u8(C + 2, vres, idx + 2); MEGDNN_FALLTHRU\
        case 2:                                                \
            vst1_lane_u8(C + 1, vres, idx + 1); MEGDNN_FALLTHRU\
        case 1:                                                \
            vst1_lane_u8(C + 0, vres, idx + 0);                \
            break;                                             \
        default:                                               \
            megdnn_assert(0);                                  \
    }

template <typename Op, typename dst_type, typename dst_neon_type,
          typename enable = void>
struct Process;

template <typename Op, typename dst_type, typename dst_neon_type>
struct Process<Op, dst_type, dst_neon_type,
               std::enable_if_t<std::is_base_of<
                       UnaryOpBase<dt_qint32, dst_type>, Op>::value>> {
    static dst_neon_type run(const int32x4x2_t& wp, const int32x4x2_t,
                             const Op& op) {
        return op(wp);
    }
};

template <typename Op, typename dst_type, typename dst_neon_type>
struct Process<Op, dst_type, dst_neon_type,
               std::enable_if_t<std::is_base_of<
                       BinaryOpBase<dt_qint32, dst_type>, Op>::value>> {
    static dst_neon_type run(const int32x4x2_t& wp, const int32x4x2_t bias,
                             const Op& op) {
        return op(wp, bias);
    }
};

template <BiasMode bmode, typename Op, typename dst_ctype, int block_m,
          int block_n, int m, int n>
struct ConvBiasMatmul {
    static void postprocess(const dt_int32* bias, const dt_int32* workspace,
                            dst_ctype* C, size_t LDC, Op op);
};

template <BiasMode bmode, typename Op, int block_m, int m>
struct ConvBiasMatmul<bmode, Op, dt_int8, block_m, 12, m, 12> {
    static void postprocess(const dt_int32* bias, const dt_int32* workspace,
                            dt_int8* C, size_t LDC, const Op& op) {
        static_assert(m > 0 && m <= block_m, "invalid m or n");
        int32x4_t vbias0, vwp0, vwp1, vwp2;
        if (bmode != BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias0 = QConverterBase::vzero();
        }
        for (int i = 0; i < m; i++) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vld1q_s32(workspace);
            vwp1 = vld1q_s32(workspace + 4);
            vwp2 = vld1q_s32(workspace + 8);

            int8x8_t vres;
            vres = Process<Op, dt_qint8, int8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias0}}, op);
            vst1_s8(C, vres);

            vres = Process<Op, dt_qint8, int8x8_t>::run({{vwp1, vwp2}},
                                                        {{vbias0, vbias0}}, op);
            //! save the high half
            vst1_lane_s32(reinterpret_cast<int32_t*>(C + 8),
                          vreinterpret_s32_s8(vres), 1);

            bias++;
            C += LDC;
            workspace += 12;
        }
    }
};


template <BiasMode bmode, typename Op, int block_m, int m, int n>
struct ConvBiasMatmul<bmode, Op, dt_int8, block_m, 4, m, n> {
    static void postprocess(const dt_int32* bias, const dt_int32* workspace,
                            dt_int8* C, size_t LDC, const Op& op) {
        static_assert(m > 0 && m <= block_m && n > 0 && n <= 4,
                      "invalid m or n");
        int i = 0;
        int32x4_t vbias0, vbias1, vwp0, vwp1;
        if (bmode != BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias0 = QConverterBase::vzero();
            vbias1 = QConverterBase::vzero();
        }
        for (; i + 1 < m; i += 2) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vld1q_s32(workspace);
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias++;
                vbias1 = vdupq_n_s32(*bias);
            }
            workspace += 4;
            vwp1 = vld1q_s32(workspace);

            int8x8_t vres;
            vres = Process<Op, dt_qint8, int8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias1}}, op);
            SAVE(C, vres, n, 0);
            C += LDC;
            SAVE(C, vres, n, 4);

            bias++;
            C += LDC;
            workspace += 4;
        }

        if (i < m) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vld1q_s32(workspace);
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias1 = QConverterBase::vzero();
            }
            vwp1 = QConverterBase::vzero();

            int8x8_t vres;
            vres = Process<Op, dt_qint8, int8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias1}}, op);
            SAVE(C, vres, n, 0);
            C += LDC;
        }
    }
};

template <BiasMode bmode, typename Op, int block_m, int m, int n>
struct ConvBiasMatmul<bmode, Op, dt_int8, block_m, 2, m, n> {
    static void postprocess(const dt_int32* bias, const dt_int32* workspace,
                            dt_int8* C, size_t LDC, const Op& op) {
        static_assert(m > 0 && m <= block_m, "invalid m or n");
        int i = 0;
        int32x4_t vbias0, vbias1, vwp0, vwp1;
        if (bmode != BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias0 = QConverterBase::vzero();
            vbias1 = QConverterBase::vzero();
        }
        for (; i + 1 < m; i += 2) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vcombine_s32(vld1_s32(workspace), vdup_n_s32(0));
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias++;
                vbias1 = vdupq_n_s32(*bias);
            }
            workspace += 2;
            vwp1 = vcombine_s32(vld1_s32(workspace), vdup_n_s32(0));

            int8x8_t vres;
            vres = Process<Op, dt_qint8, int8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias1}}, op);
            SAVE(C, vres, n, 0);
            C += LDC;
            SAVE(C, vres, n, 4);

            bias++;
            C += LDC;
            workspace += 2;
        }

        if (i < m) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vcombine_s32(vld1_s32(workspace), vdup_n_s32(0));
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias1 = QConverterBase::vzero();
            }
            vwp1 = QConverterBase::vzero();

            int8x8_t vres;
            vres = Process<Op, dt_qint8, int8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias1}}, op);
            SAVE(C, vres, n, 0);
            C += LDC;
        }
    }
};

template <BiasMode bmode, typename Op, int block_m, int m>
struct ConvBiasMatmul<bmode, Op, dt_uint8, block_m, 8, m, 8> {
    static void postprocess(const dt_int32* bias, const dt_int32* workspace,
                            dt_uint8* C, size_t LDC, const Op& op) {
        static_assert(m > 0 && m <= block_m, "invalid m or n");
        int32x4_t vbias0, vwp0, vwp1;
        if (bmode != BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias0 = QConverterBase::vzero();
        }
        for (int i = 0; i < m; i++) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vld1q_s32(workspace);
            vwp1 = vld1q_s32(workspace + 4);

            uint8x8_t vres;
            vres = Process<Op, dt_quint8, uint8x8_t>::run(
                    {{vwp0, vwp1}}, {{vbias0, vbias0}}, op);
            vst1_u8(C, vres);

            bias++;
            C += LDC;
            workspace += 8;
        }
    }
};


template <BiasMode bmode, typename Op, int block_m, int m, int n>
struct ConvBiasMatmul<bmode, Op, dt_uint8, block_m, 4, m, n> {
    static void postprocess(const dt_int32* bias, const dt_int32* workspace,
                            dt_uint8* C, size_t LDC, const Op& op) {
        static_assert(m > 0 && m <= block_m && n > 0 && n <= 4,
                      "invalid m or n");
        int i = 0;
        int32x4_t vbias0, vbias1, vwp0, vwp1;
        if (bmode != BiasMode::BROADCAST_CHANNEL_BIAS) {
            vbias0 = QConverterBase::vzero();
            vbias1 = QConverterBase::vzero();
        }
        for (; i + 1 < m; i += 2) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vld1q_s32(workspace);
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias++;
                vbias1 = vdupq_n_s32(*bias);
            }
            workspace += 4;
            vwp1 = vld1q_s32(workspace);

            uint8x8_t vres;
            vres = Process<Op, dt_quint8, uint8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias1}}, op);
            SAVEU(C, vres, n, 0);
            C += LDC;
            SAVEU(C, vres, n, 4);

            bias++;
            C += LDC;
            workspace += 4;
        }

        if (i < m) {
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias0 = vdupq_n_s32(*bias);
            }
            vwp0 = vld1q_s32(workspace);
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                vbias1 = QConverterBase::vzero();
            }
            vwp1 = QConverterBase::vzero();

            uint8x8_t vres;
            vres = Process<Op, dt_quint8, uint8x8_t>::run({{vwp0, vwp1}},
                                                        {{vbias0, vbias1}}, op);
            SAVEU(C, vres, n, 0);
            C += LDC;
        }
    }
};


#define DISPATCH_M(cb, _m, _n, ...)               \
    switch (_m) {                                 \
        case 4: {                                 \
            DISPATCH_N(cb, 4, _n, ##__VA_ARGS__); \
            break;                                \
        }                                         \
        case 3: {                                 \
            DISPATCH_N(cb, 3, _n, ##__VA_ARGS__); \
            break;                                \
        }                                         \
        case 2: {                                 \
            DISPATCH_N(cb, 2, _n, ##__VA_ARGS__); \
            break;                                \
        }                                         \
        case 1: {                                 \
            DISPATCH_N(cb, 1, _n, ##__VA_ARGS__); \
            break;                                \
        }                                         \
        default:                                  \
            megdnn_assert(0);                     \
    }

#define DISPATCH_N(cb, _m, _n, ...)   \
    switch (_n) {                     \
        case 4: {                     \
            cb(_m, 4, ##__VA_ARGS__); \
            break;                    \
        }                             \
        case 3: {                     \
            cb(_m, 3, ##__VA_ARGS__); \
            break;                    \
        }                             \
        case 2: {                     \
            cb(_m, 2, ##__VA_ARGS__); \
            break;                    \
        }                             \
        case 1: {                     \
            cb(_m, 1, ##__VA_ARGS__); \
            break;                    \
        }                             \
        default:                      \
            megdnn_assert(0);         \
    }

//! _n should be a compiler time constant
#define DISPATCH_M_N(cb, _m, _n, ...)             \
    switch (_m) {                                 \
        case 4: {                                 \
            cb(4, _n, ##__VA_ARGS__);             \
            break;                                \
        }                                         \
        case 3: {                                 \
            cb(3, _n, ##__VA_ARGS__);             \
            break;                                \
        }                                         \
        case 2: {                                 \
            cb(2, _n, ##__VA_ARGS__);             \
            break;                                \
        }                                         \
        case 1: {                                 \
            cb(1, _n, ##__VA_ARGS__);             \
            break;                                \
        }                                         \
        default:                                  \
            megdnn_assert(0);                     \
    }

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
