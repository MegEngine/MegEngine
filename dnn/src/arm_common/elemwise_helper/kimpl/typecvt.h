/**
 * \file dnn/src/arm_common/elemwise_helper/kimpl/typecvt.h
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

#include "src/arm_common/elemwise_helper/kimpl/op_base.h"

namespace megdnn {
namespace arm_common {

template <typename src_ctype, typename dst_ctype = src_ctype>
struct TypeCvtOp;

template <>
struct TypeCvtOp<dt_qint32, dt_qint8> : UnaryOpBase<dt_qint32, dt_qint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_qint8* dst) const {
        vst1_s8(reinterpret_cast<int8_t*>(dst), operator()(vsrc));
    }
    void operator()(const int32x4_t& vsrc, dt_qint8* dst) const {
        vst1_lane_s32(reinterpret_cast<int32_t*>(dst),
                      (int32x2_t)(operator()(vsrc)), 0);
    }
    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }
    dt_qint8 operator()(const dt_qint32& src) const {
        float fsrc = src.as_int32() * this->scale;
        return QConverter::convert<dt_qint8, float>(fsrc);
    }

    int8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale);

        return QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
    }
    int8x8_t operator()(const int32x4_t& src) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(src), this->vscale);
        return QConverter::convert<int8x8_t, float32x4_t>(vitem0);
    }
    int8x8_t operator()(const float32x4_t& src) const {
        auto vitem0 = vmulq_f32(src, this->vscale);
        return QConverter::convert<int8x8_t, float32x4_t>(vitem0);
    }
};

template <>
struct TypeCvtOp<dt_qint32, dt_quint8> : UnaryOpBase<dt_qint32, dt_quint8> {
    using UnaryOpBase::UnaryOpBase;
    constexpr static size_t SIMD_WIDTH = 4;

    void operator()(const int32x4x2_t& vsrc, dt_quint8* dst) const {
        vst1_u8(reinterpret_cast<uint8_t*>(dst), operator()(vsrc));
    }

    void operator()(const src_ctype& src, dst_ctype* dst) const {
        *dst = operator()(src);
    }

    dt_quint8 operator()(const src_ctype& src) const {
        return QConverter::convert<dt_quint8, float>(
                src.as_int32() * this->scale, this->zp);
    }
    uint8x8_t operator()(const int32x4x2_t& vsrc) const {
        auto vitem0 = vmulq_f32(vcvtq_f32_s32(vsrc.val[0]), this->vscale);
        auto vitem1 = vmulq_f32(vcvtq_f32_s32(vsrc.val[1]), this->vscale);

        return QConverter::convert<uint8x8_t, float32x4x2_t>({{vitem0, vitem1}},
                                                             this->vzp);
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
