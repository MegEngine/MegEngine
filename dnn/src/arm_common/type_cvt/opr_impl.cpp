/**
 * \file dnn/src/arm_common/type_cvt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/type_cvt/opr_impl.h"

#include <cstring>
#include <deque>
#include "midout.h"
#include "src/arm_common/quantized_converter.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

MIDOUT_DECL(megdnn_arm_typecvt_fix2float)
MIDOUT_DECL(megdnn_arm_typecvt_quan2float)
MIDOUT_DECL(megdnn_arm_typecvt_quantized)
MIDOUT_DECL(megdnn_arm_typecvt_float)

using namespace megdnn;
using namespace arm_common;

namespace {

template <typename ctype, typename dtype>
struct QuantizedTypeCvter;

template <>
struct QuantizedTypeCvter<int32_t, int8_t> {
    using stype = int32_t;
    using dst_type = int8_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const int32_t* src, int8_t* dst) {
        float32x4_t vitem0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src)), vscale);
        float32x4_t vitem1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + 4)), vscale);

        auto vres = QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
        vst1_s8(dst, vres);
    }

    void cvt_remain(const int32_t* src, int8_t* dst) {
        *dst = saturate<int8_t, float>(std::round(*src * scale), -128, 127);
    }
};

template <>
struct QuantizedTypeCvter<int8_t, int32_t> {
    using stype = int8_t;
    using dst_type = int32_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale;
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const int8_t* src, int32_t* dst) {
        int16x8_t vitem = vmovl_s8(vld1_s8(src));
        auto vret0 = QConverter::convert<int32x4_t, float32x4_t>(
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vitem))), vscale));
        auto vret1 = QConverter::convert<int32x4_t, float32x4_t>(
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vitem))), vscale));
        vst1q_s32(dst, vret0);
        vst1q_s32(dst + 4, vret1);
    }

    void cvt_remain(const int8_t* src, int32_t* dst) {
        *dst = saturate<int32_t, float>(
                std::round(*src * scale), -2147483648, 2147483647);
    }
};

template <>
struct QuantizedTypeCvter<float, int8_t> {
    using stype = float;
    using dst_type = int8_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        float src_scale = 1;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const float* src, int8_t* dst) {
        float32x4_t vitem0 = vmulq_f32(vld1q_f32(src), vscale);
        float32x4_t vitem1 = vmulq_f32(vld1q_f32(src + 4), vscale);

        auto vres = QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
        vst1_s8(dst, vres);
    }

    void cvt_remain(const float* src, int8_t* dst) {
        *dst = saturate<int8_t, float>(std::round(*src * scale), -128, 127);
    }
};

template <>
struct QuantizedTypeCvter<int32_t, int32_t> {
    using stype = int32_t;
    using dst_type = int32_t;
    static constexpr size_t SIMD_WIDTH = 4;
    float scale;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale;
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const int32_t* src, int32_t* dst) {
        float32x4_t vitem = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src)), vscale);

        auto vres = QConverter::convert<int32x4_t, float32x4_t>(vitem);
        vst1q_s32(dst, vres);
    }

    void cvt_remain(const int32_t* src, int32_t* dst) {
        *dst = saturate<int32_t, float>(
                std::round(*src * scale), -2147483648, 2147483647);
    }
};

template <>
struct QuantizedTypeCvter<int8_t, int8_t> {
    using stype = int8_t;
    using dst_type = int8_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        scale = src_scale / dst_scale;
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const int8_t* src, int8_t* dst) {
        int16x8_t vdata = vmovl_s8(vld1_s8(src));
        float32x4_t vitem0 =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vdata))), vscale);
        float32x4_t vitem1 =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vdata))), vscale);

        auto vres = QConverter::convert<int8x8_t, float32x4x2_t>({{vitem0, vitem1}});
        vst1_s8(dst, vres);
    }

    void cvt_remain(const int8_t* src, int8_t* dst) {
        *dst = saturate<int8_t, float>(std::round(*src * scale), -128, 127);
    }
};

template <>
struct QuantizedTypeCvter<float, uint8_t> {
    using stype = float;
    using dst_type = uint8_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    uint8_t zp;
    int32x4_t vzp, vzero;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        float src_scale = 1;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        scale = src_scale / dst_scale;
        zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        vzp = vdupq_n_s32(static_cast<int>(zp));
        vzero = vdupq_n_s32(0);
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const float* src, uint8_t* dst) {
        float32x4_t vitem0 = vmulq_f32(vld1q_f32(src), vscale);
        float32x4_t vitem1 = vmulq_f32(vld1q_f32(src + 4), vscale);

        auto vres = QConverter::convert<uint8x8_t, float32x4x2_t>(
                {{vitem0, vitem1}}, this->vzp);
        vst1_u8(dst, vres);
    }

    void cvt_remain(const float* src, uint8_t* dst) {
        *dst = saturate<uint8_t, float>(std::round(*src * scale) + zp, 0, 255);
    }
};

template <>
struct QuantizedTypeCvter<int32_t, uint8_t> {
    using stype = int32_t;
    using dst_type = uint8_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    uint8_t zp;
    int32x4_t vzp, vzero;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        scale = src_scale / dst_scale;
        zp = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        vzp = vdupq_n_s32(static_cast<int>(zp));
        vzero = vdupq_n_s32(0);
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const int32_t* src, uint8_t* dst) {
        float32x4_t vitem0 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src)), vscale);
        float32x4_t vitem1 = vmulq_f32(vcvtq_f32_s32(vld1q_s32(src + 4)), vscale);
        auto vres = QConverter::convert<uint8x8_t, float32x4x2_t>(
                {{vitem0, vitem1}}, this->vzp);
        vst1_u8(dst, vres);
    }

    void cvt_remain(const int32_t* src, uint8_t* dst) {
        *dst = saturate<uint8_t, float>(std::round(*src * scale) + zp, 0, 255);
    }
};

template <>
struct QuantizedTypeCvter<uint8_t, uint8_t> {
    using stype = uint8_t;
    using dst_type = uint8_t;
    static constexpr size_t SIMD_WIDTH = 8;
    float scale;
    uint8_t zp_dst, zp_src;
    int32x4_t vzp_dst, vzero;
    int16x8_t vzp_src;
    float32x4_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::Quantized8Asymm>().scale;
        float dst_scale = dst_dtype.param<dtype::Quantized8Asymm>().scale;
        scale = src_scale / dst_scale;
        zp_dst = dst_dtype.param<dtype::Quantized8Asymm>().zero_point;
        zp_src = src_dtype.param<dtype::Quantized8Asymm>().zero_point;
        vzp_dst = vdupq_n_s32(static_cast<int>(zp_dst));
        vzp_src = vdupq_n_s16(static_cast<int16_t>(zp_src));
        vzero = vdupq_n_s32(0);
        vscale = vdupq_n_f32(scale);
    }

    void cvt(const uint8_t* src, uint8_t* dst) {
        int16x8_t vdata = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src)));
        vdata = vsubq_s16(vdata, vzp_src);
        float32x4_t vitem0 =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vdata))), vscale);
        float32x4_t vitem1 =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(vdata))), vscale);

        auto vres = QConverter::convert<uint8x8_t, float32x4x2_t>(
                {{vitem0, vitem1}}, this->vzp_dst);
        vst1_u8(dst, vres);
    }

    void cvt_remain(const uint8_t* src, uint8_t* dst) {
        *dst = saturate<uint8_t, float>(
                std::round((*src - zp_src) * scale) + zp_dst, 0, 255);
    }
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <typename ctype, typename dtype>
struct FloatTypeCvter;

template <>
struct FloatTypeCvter<__fp16, float> {
    using stype = __fp16;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 8;
    FloatTypeCvter(DType src_dtype, DType dst_dtype) {}

    void cvt(const __fp16* src, float* dst) {
        float16x8_t vdata = vld1q_f16(src);
        float32x4_t vdata_low = vcvt_f32_f16(vget_low_f16(vdata));
        float32x4_t vdata_high = vcvt_f32_f16(vget_high_f16(vdata));
        vst1q_f32(dst, vdata_low);
        vst1q_f32(dst + 4, vdata_high);
    }

    void cvt_remain(const __fp16* src, float* dst) { *dst = *src; }
};

template <>
struct FloatTypeCvter<float, __fp16> {
    using stype = float;
    using dst_type = __fp16;
    static constexpr size_t SIMD_WIDTH = 8;
    FloatTypeCvter(DType src_dtype, DType dst_dtype) {}

    void cvt(const float* src, __fp16* dst) {
        float32x4_t vdata0 = vld1q_f32(src);
        float32x4_t vdata1 = vld1q_f32(src + 4);
        float16x8_t vdata = vcombine_f16(vcvt_f16_f32(vdata0), vcvt_f16_f32(vdata1));
        vst1q_f16(dst, vdata);
    }

    void cvt_remain(const float* src, __fp16* dst) { *dst = *src; }
};
#endif

template <typename TypeCvter>
void do_typecvt(
        const typename TypeCvter::stype* src, typename TypeCvter::dst_type* dst,
        DType src_dtype, DType dst_dtype, size_t nr_elems) {
    TypeCvter typecvt(src_dtype, dst_dtype);
    size_t i = 0;
    for (; i + TypeCvter::SIMD_WIDTH <= nr_elems; i += TypeCvter::SIMD_WIDTH) {
        typecvt.cvt(src, dst);
        src += TypeCvter::SIMD_WIDTH;
        dst += TypeCvter::SIMD_WIDTH;
    }
#if MEGDNN_FIX_AARCH32_BUG
// FIXME: as llvm may cause cannot select error if enable vectorize
#pragma clang loop vectorize(disable)
#endif
    for (; i < nr_elems; i++) {
        typecvt.cvt_remain(src, dst);
        src++;
        dst++;
    }
}

template <typename ctype, typename dtype>
struct Fix2FloatTypeCvter;

template <typename ctype, typename dtype>
struct Quan2FloatTypeCvter;

template <>
struct Fix2FloatTypeCvter<int16_t, float> {
    using stype = int16_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 8;

    Fix2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const int16_t* src, float* dst) {
        int16x8_t vitem = vld1q_s16(src);
        auto vres = QConverter::convert<float32x4x2_t, int16x8_t>(vitem);
        vst1q_f32_x2(dst, vres);
    }

    void cvt_remain(const int16_t* src, float* dst) { *dst = *src; }
};

template <>
struct Fix2FloatTypeCvter<uint16_t, float> {
    using stype = uint16_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 8;

    Fix2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const uint16_t* src, float* dst) {
        uint16x8_t vitem = vld1q_u16(src);
        auto vres = QConverter::convert<float32x4x2_t, uint16x8_t>(vitem);
        vst1q_f32_x2(dst, vres);
    }

    void cvt_remain(const uint16_t* src, float* dst) { *dst = *src; }
};

template <>
struct Fix2FloatTypeCvter<int8_t, float> {
    using stype = int8_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 16;

    Fix2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const int8_t* src, float* dst) {
        int8x16_t vitem = vld1q_s8(src);
        int16x8_t vtrans_high = vmovl_s8(vget_high_s8(vitem));
        int16x8_t vtrans_low = vmovl_s8(vget_low_s8(vitem));
        auto vres_high = QConverter::convert<float32x4x2_t, int16x8_t>(vtrans_high);
        auto vres_low = QConverter::convert<float32x4x2_t, int16x8_t>(vtrans_low);
        vst1q_f32_x2(dst, vres_low);
        vst1q_f32_x2(dst + 8, vres_high);
    }

    void cvt_remain(const int8_t* src, float* dst) { *dst = *src; }
};

template <>
struct Fix2FloatTypeCvter<uint8_t, float> {
    using stype = uint8_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 16;

    Fix2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const uint8_t* src, float* dst) {
        uint8x16_t vitem = vld1q_u8(src);
        uint16x8_t vtrans_high = vmovl_u8(vget_high_u8(vitem));
        uint16x8_t vtrans_low = vmovl_u8(vget_low_u8(vitem));
        auto vres_high = QConverter::convert<float32x4x2_t, uint16x8_t>(vtrans_high);
        auto vres_low = QConverter::convert<float32x4x2_t, uint16x8_t>(vtrans_low);
        vst1q_f32_x2(dst, vres_low);
        vst1q_f32_x2(dst + 8, vres_high);
    }

    void cvt_remain(const uint8_t* src, float* dst) { *dst = *src; }
};

template <>
struct Quan2FloatTypeCvter<int8_t, float> {
    using stype = int8_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 16;
    float _scale = 0.0f;
    float32x4_t _vscale;

    Quan2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        _scale = src_dtype.param<dtype::QuantizedS8>().scale;
        _vscale = vdupq_n_f32(_scale);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const int8_t* src, float* dst) {
        int8x16_t vitem = vld1q_s8(src);
        int16x8_t vtrans_high = vmovl_s8(vget_high_s8(vitem));
        int16x8_t vtrans_low = vmovl_s8(vget_low_s8(vitem));
        auto vres_high = QConverter::convert<float32x4x2_t, int16x8_t>(vtrans_high);
        auto vres_low = QConverter::convert<float32x4x2_t, int16x8_t>(vtrans_low);
        vst1q_f32(dst, vmulq_f32(vres_low.val[0], _vscale));
        vst1q_f32(dst + 4, vmulq_f32(vres_low.val[1], _vscale));
        vst1q_f32(dst + 8, vmulq_f32(vres_high.val[0], _vscale));
        vst1q_f32(dst + 12, vmulq_f32(vres_high.val[1], _vscale));
    }

    void cvt_remain(const int8_t* src, float* dst) { *dst = *src * _scale; }
};

#if defined(__ARM_FEATURE_FMA)
#define Vfmaq_f32(d, n, m) vfmaq_f32(d, n, m)
#else
#define Vfmaq_f32(d, n, m) vmlaq_f32(d, n, m)
#endif

template <>
struct Quan2FloatTypeCvter<uint8_t, float> {
    using stype = uint8_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = 16;
    float _scale = 0.0f;
    float32x4_t _vscale;
    uint8_t _zp = 0;
    float32x4_t _vbias;

    Quan2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        _scale = src_dtype.param<dtype::Quantized8Asymm>().scale;
        _vscale = vdupq_n_f32(_scale);
        _zp = src_dtype.param<dtype::Quantized8Asymm>().zero_point;
        float bias = -_zp * 1.0f * _scale;
        _vbias = vdupq_n_f32(bias);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const uint8_t* src, float* dst) {
        uint8x16_t vitem = vld1q_u8(src);
        uint16x8_t vtrans_high = vmovl_u8(vget_high_u8(vitem));
        uint16x8_t vtrans_low = vmovl_u8(vget_low_u8(vitem));
        auto vres_high = QConverter::convert<float32x4x2_t, uint16x8_t>(vtrans_high);
        auto vres_low = QConverter::convert<float32x4x2_t, uint16x8_t>(vtrans_low);
        vst1q_f32(dst, Vfmaq_f32(_vbias, vres_low.val[0], _vscale));
        vst1q_f32(dst + 4, Vfmaq_f32(_vbias, vres_low.val[1], _vscale));
        vst1q_f32(dst + 8, Vfmaq_f32(_vbias, vres_high.val[0], _vscale));
        vst1q_f32(dst + 12, Vfmaq_f32(_vbias, vres_high.val[1], _vscale));
    }

    void cvt_remain(const uint8_t* src, float* dst) { *dst = (*src - _zp) * _scale; }
};

#undef Vfmaq_f32

template <typename stype, typename dtype>
struct TypeCvtTask {
    const stype* src;
    dtype* dst;
    size_t dim;
    size_t nr_elems;

    explicit TypeCvtTask(const stype* s, dtype* d, size_t n, size_t tot)
            : src(s), dst(d), dim(n), nr_elems(tot) {}
};

template <typename TypeCvter>
void do_typecvt(
        const typename TypeCvter::stype* src, typename TypeCvter::dst_type* dst,
        DType src_dtype, DType dst_dtype, const TensorLayout& src_layout) {
    TypeCvter typecvt(src_dtype, dst_dtype);
    auto src_collapse_layout = src_layout.collapse_contiguous();

    using TypeCvtTaskWithType =
            TypeCvtTask<typename TypeCvter::stype, typename TypeCvter::dst_type>;
    std::deque<TypeCvtTaskWithType> task_queue;
    task_queue.emplace_back(src, dst, 0, src_collapse_layout.total_nr_elems());

    while (!task_queue.empty()) {
        auto&& task = task_queue.front();
        const typename TypeCvter::stype* psrc = task.src;
        typename TypeCvter::dst_type* pdst = task.dst;
        size_t dim = task.dim;
        size_t nr_elems = task.nr_elems;

        //! calc according to stride information
        if (src_collapse_layout.stride[dim] == 1) {
            size_t i = 0;
            for (; i + TypeCvter::SIMD_WIDTH < nr_elems; i += TypeCvter::SIMD_WIDTH) {
                typecvt.cvt(psrc, pdst);
                psrc += TypeCvter::SIMD_WIDTH;
                pdst += TypeCvter::SIMD_WIDTH;
            }
#if MEGDNN_FIX_AARCH32_BUG
// FIXME: as llvm may cause cannot select error if enable vectorize
#pragma clang loop vectorize(disable)
#endif
            for (; i < nr_elems; i++) {
                typecvt.cvt_remain(psrc, pdst);
                psrc++;
                pdst++;
            }
        } else {
            size_t calc_num = src_collapse_layout.shape[dim];
            size_t src_stride = src_collapse_layout.stride[dim];
            size_t dst_stride = nr_elems / calc_num;
            for (size_t i = 0; i < calc_num; ++i) {
                task_queue.emplace_back(psrc, pdst, dim + 1, dst_stride);
                psrc += src_stride;
                pdst += dst_stride;
            }
        }

        task_queue.pop_front();
    }
}

}  // anonymous namespace

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    DType src_dtype = src.layout.dtype;
    DType dst_dtype = dst.layout.dtype;
    bool execed = false;
    auto src_collapse_layout = src.layout.collapse_contiguous();

    if (src.layout.is_contiguous()) {
        using namespace dtype;
        size_t nr_elems = src.layout.total_nr_elems();
#define DISPATCH_QUANTIZED(_stype_enumv, _stype, _dtype_enumv, _dtype, _midout_iv) \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&                    \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {                    \
        MIDOUT_BEGIN(megdnn_arm_typecvt_quantized, midout_iv(_midout_iv)) {        \
            using _TypeCvter = QuantizedTypeCvter<_stype, _dtype>;                 \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_typecvt<_TypeCvter>(                   \
                    src.compatible_ptr<_stype>(), dst.compatible_ptr<_dtype>(),    \
                    src_dtype, dst_dtype, nr_elems));                              \
            execed = true;                                                         \
        }                                                                          \
        MIDOUT_END();                                                              \
    }
        DISPATCH_QUANTIZED(QuantizedS32, int32_t, Quantized8Asymm, uint8_t, 0);
        DISPATCH_QUANTIZED(QuantizedS32, int32_t, QuantizedS8, int8_t, 1);
        DISPATCH_QUANTIZED(QuantizedS8, int8_t, QuantizedS32, int32_t, 2);
        DISPATCH_QUANTIZED(QuantizedS8, int8_t, QuantizedS8, int8_t, 3);
        DISPATCH_QUANTIZED(Quantized8Asymm, uint8_t, Quantized8Asymm, uint8_t, 4);
        DISPATCH_QUANTIZED(QuantizedS32, int32_t, QuantizedS32, int32_t, 5);
        DISPATCH_QUANTIZED(float, float, QuantizedS8, int8_t, 6);
        DISPATCH_QUANTIZED(float, float, Quantized8Asymm, uint8_t, 7);
#undef DISPATCH_QUANTIZED

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define DISPATCH_FLOAT(_stype_enumv, _stype, _dtype_enumv, _dtype, _midout_iv)      \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&                     \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {                     \
        MIDOUT_BEGIN(megdnn_arm_typecvt_float, midout_iv(_midout_iv)) {             \
            using _TypeCvter = FloatTypeCvter<_stype, _dtype>;                      \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_typecvt<_TypeCvter>(                    \
                    reinterpret_cast<_stype*>(src.raw_ptr()),                       \
                    reinterpret_cast<_dtype*>(dst.raw_ptr()), src_dtype, dst_dtype, \
                    nr_elems));                                                     \
            execed = true;                                                          \
        }                                                                           \
        MIDOUT_END();                                                               \
    }
        DISPATCH_FLOAT(dt_float16, __fp16, float, float, 0);
        DISPATCH_FLOAT(float, float, dt_float16, __fp16, 1);
#undef DISPATCH_FLOAT
#endif
    }

    size_t last_stride = src_collapse_layout.stride[src_collapse_layout.ndim - 1];
    if (!execed && last_stride == 1 && dst.layout.is_contiguous()) {
        using namespace dtype;
#define DISPATCH_FIX2FLOAT(_stype_enumv, _stype, _dtype_enumv, _dtype, _midout_iv) \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&                    \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {                    \
        MIDOUT_BEGIN(megdnn_arm_typecvt_fix2float, midout_iv(_midout_iv)) {        \
            using _TypeCvter = Fix2FloatTypeCvter<_stype, _dtype>;                 \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_typecvt<_TypeCvter>(                   \
                    src.compatible_ptr<_stype>(), dst.compatible_ptr<_dtype>(),    \
                    src_dtype, dst_dtype, src.layout));                            \
            execed = true;                                                         \
        }                                                                          \
        MIDOUT_END();                                                              \
    }
        DISPATCH_FIX2FLOAT(Int16, int16_t, Float32, float, 0);
        DISPATCH_FIX2FLOAT(Uint16, uint16_t, Float32, float, 1);
        DISPATCH_FIX2FLOAT(Int8, int8_t, Float32, float, 2);
        DISPATCH_FIX2FLOAT(Uint8, uint8_t, Float32, float, 3);
#undef DISPATCH_FIX2FLOAT

#define DISPATCH_QUAN2FLOAT(_stype_enumv, _stype, _dtype_enumv, _dtype, _midout_iv) \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&                     \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {                     \
        MIDOUT_BEGIN(megdnn_arm_typecvt_quan2float, midout_iv(_midout_iv)) {        \
            using _TypeCvter = Quan2FloatTypeCvter<_stype, _dtype>;                 \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_typecvt<_TypeCvter>(                    \
                    src.compatible_ptr<_stype>(), dst.compatible_ptr<_dtype>(),     \
                    src_dtype, dst_dtype, src.layout));                             \
            execed = true;                                                          \
        }                                                                           \
        MIDOUT_END();                                                               \
    }
        DISPATCH_QUAN2FLOAT(QuantizedS8, int8_t, Float32, float, 0);
        DISPATCH_QUAN2FLOAT(Quantized8Asymm, uint8_t, Float32, float, 1);
#undef DISPATCH_QUAN2FLOAT

#define DISPATCH_QUANTIZED(_stype_enumv, _stype, _dtype_enumv, _dtype, _midout_iv) \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&                    \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {                    \
        MIDOUT_BEGIN(megdnn_arm_typecvt_quantized, midout_iv(_midout_iv)) {        \
            using _TypeCvter = QuantizedTypeCvter<_stype, _dtype>;                 \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_typecvt<_TypeCvter>(                   \
                    src.compatible_ptr<_stype>(), dst.compatible_ptr<_dtype>(),    \
                    src_dtype, dst_dtype, src.layout));                            \
            execed = true;                                                         \
        }                                                                          \
        MIDOUT_END();                                                              \
    }
        DISPATCH_QUANTIZED(QuantizedS32, int32_t, Quantized8Asymm, uint8_t, 0);
        DISPATCH_QUANTIZED(QuantizedS32, int32_t, QuantizedS8, int8_t, 1);
        DISPATCH_QUANTIZED(QuantizedS8, int8_t, QuantizedS32, int32_t, 2);
        DISPATCH_QUANTIZED(QuantizedS8, int8_t, QuantizedS8, int8_t, 3);
        DISPATCH_QUANTIZED(Quantized8Asymm, uint8_t, Quantized8Asymm, uint8_t, 4);
        DISPATCH_QUANTIZED(QuantizedS32, int32_t, QuantizedS32, int32_t, 5);
        DISPATCH_QUANTIZED(float, float, QuantizedS8, int8_t, 6);
        DISPATCH_QUANTIZED(float, float, Quantized8Asymm, uint8_t, 7);
#undef DISPATCH_QUANTIZED

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define DISPATCH_FLOAT(_stype_enumv, _stype, _dtype_enumv, _dtype, _midout_iv)      \
    if (src_dtype.enumv() == DTypeTrait<_stype_enumv>::enumv &&                     \
        dst_dtype.enumv() == DTypeTrait<_dtype_enumv>::enumv) {                     \
        MIDOUT_BEGIN(megdnn_arm_typecvt_float, midout_iv(_midout_iv)) {             \
            using _TypeCvter = FloatTypeCvter<_stype, _dtype>;                      \
            MEGDNN_DISPATCH_CPU_KERN_OPR(do_typecvt<_TypeCvter>(                    \
                    reinterpret_cast<_stype*>(src.raw_ptr()),                       \
                    reinterpret_cast<_dtype*>(dst.raw_ptr()), src_dtype, dst_dtype, \
                    src.layout));                                                   \
            execed = true;                                                          \
        }                                                                           \
        MIDOUT_END();                                                               \
    }
        DISPATCH_FLOAT(dt_float16, __fp16, float, float, 0);
        DISPATCH_FLOAT(float, float, dt_float16, __fp16, 1);
#undef DISPATCH_FLOAT
#endif
    }

    if (!execed) {
        fallback::TypeCvtImpl::exec(src, dst);
    }
}

// vim: syntax=cpp.doxygen
