/**
 * \file dnn/src/arm_common/quantized_converter.h
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

#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "src/common/utils.h"
#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"

namespace megdnn {
namespace fallback {

struct QConverterBase {
    inline static GI_INT32_t vzero() { return GiBroadcastInt32(0); }

    inline static GI_FLOAT32_t vfzero() { return GiBroadcastFloat32(0.f); }

    inline static GI_FLOAT32_t vfhalf() { return GiBroadcastFloat32(0.5f); }

    inline static GI_FLOAT32_t vfneg_half() { return GiBroadcastFloat32(-0.5f); }
};

struct QConverter {
    template <typename dst_type, typename... src_type>
    static inline dst_type convert(const src_type&... src);

    template <typename dst_type, typename... src_type>
    static inline dst_type round(const src_type&... src);
};

template <>
inline dt_qint8 QConverter::convert(const float& src) {
    return dt_qint8(saturate<int8_t, float>(std::round(src), -128, 127));
}

template <>
inline dt_quint8 QConverter::convert(const float& src, const uint8_t& zp) {
    return dt_quint8(saturate<uint8_t, float>(std::round(src) + zp, 0, 255));
}

template <>
inline dt_qint32 QConverter::convert(const float& src) {
    return dt_qint32(saturate<int32_t, float>(
            std::round(src), static_cast<float>(std::numeric_limits<int32_t>::min()),
            static_cast<float>(std::numeric_limits<int32_t>::max())));
}

template <>
inline GI_FLOAT32_V2_t QConverter::convert(const GI_INT16_t& vsrc) {
    GI_INT32_t vhi = GiMoveHighLongInt16(vsrc);
    GI_INT32_t vlo = GiMoveLowLongInt16(vsrc);
    return {{GiCastToFloat32(vlo), GiCastToFloat32(vhi)}};
}

template <>
inline GI_INT8_t QConverter::convert(const GI_FLOAT32_V2_t& vsrc) {
    return GiCvtFromFloat32V2ToInt8(vsrc);
}
template <>
inline GI_INT8_t QConverter::convert(const GI_FLOAT32_t& src) {
    return GiCvtFromFloat32ToInt8(src);
}

template <>
inline GI_INT32_t QConverter::round(const GI_FLOAT32_t& vsrc) {
    return GiRoundAsInt32(vsrc);
}
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
