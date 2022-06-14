/**
 * \file dnn/src/fallback/type_cvt/typecvt_helper.h
 */
#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"
#include "src/fallback/quantized_converter.h"

namespace megdnn {
namespace fallback {

template <typename ctype, typename dtype>
struct QuantizedTypeCvter;

template <>
struct QuantizedTypeCvter<int32_t, int8_t> {
    using stype = int32_t;
    using dst_type = int8_t;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int32_t) * 2;
    static constexpr size_t SIMD_STEP = GI_SIMD_LEN_BYTE / sizeof(int32_t);
    float scale;
    GI_FLOAT32_FIXLEN_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        scale = src_scale / dst_scale;
        vscale = GiFloat32Type2FixLenType(GiBroadcastFloat32(scale));
    }

    void cvt(const int32_t* src, int8_t* dst) {
        GI_FLOAT32_t t;
        t = GiFixLenType2GiFloat32Type(vscale);
        GI_FLOAT32_t vitem0 = GiMultiplyFloat32(GiCastToFloat32(GiLoadInt32(src)), t);
        GI_FLOAT32_t vitem1 =
                GiMultiplyFloat32(GiCastToFloat32(GiLoadInt32(src + SIMD_STEP)), t);

        GI_FLOAT32_V2_t v2;
        GiSetSubVectorFloat32V2(v2, 0, vitem0);
        GiSetSubVectorFloat32V2(v2, 1, vitem1);
        auto vres = QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(v2);
        GiStoreLowInt8(dst, vres);
    }

    void cvt_remain(const int32_t* src, int8_t* dst) {
        *dst = saturate<int8_t, float>(std::round(*src * scale), -128.f, 127.f);
    }
};

template <>
struct QuantizedTypeCvter<int8_t, int32_t> {
    using stype = int8_t;
    using dst_type = int32_t;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    float scale;
    GI_FLOAT32_FIXLEN_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale;
        scale = src_scale / dst_scale;
        vscale = GiFloat32Type2FixLenType(GiBroadcastFloat32(scale));
    }

    void cvt(const int8_t* src, int32_t* dst) {
        GI_FLOAT32_t t;
        t = GiFixLenType2GiFloat32Type(vscale);
        GI_INT8_t data = GiLoadInt8(src);
        GI_INT16_t vitem0 = GiMoveLowLongInt8(data);
        GI_INT16_t vitem1 = GiMoveHighLongInt8(data);
        auto vret0 = QConverter::round<GI_INT32_t, GI_FLOAT32_t>(
                GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vitem0)), t));
        auto vret1 = QConverter::round<GI_INT32_t, GI_FLOAT32_t>(
                GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vitem0)), t));
        auto vret2 = QConverter::round<GI_INT32_t, GI_FLOAT32_t>(
                GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vitem1)), t));
        auto vret3 = QConverter::round<GI_INT32_t, GI_FLOAT32_t>(
                GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vitem1)), t));

        constexpr size_t step = GI_SIMD_LEN_BYTE / sizeof(int32_t);
        GiStoreInt32(dst, vret0);
        GiStoreInt32(dst + step, vret1);
        GiStoreInt32(dst + 2 * step, vret2);
        GiStoreInt32(dst + 3 * step, vret3);
    }

    void cvt_remain(const int8_t* src, int32_t* dst) {
        *dst = saturate<int32_t, float>(
                std::round(*src * scale), -2147483648.f, 2147483647.f);
    }
};

template <>
struct QuantizedTypeCvter<float, int8_t> {
    using stype = float;
    using dst_type = int8_t;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(float) * 2;
    static constexpr size_t SIMD_STEP = GI_SIMD_LEN_BYTE / sizeof(float);
    float scale;
    GI_FLOAT32_FIXLEN_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        float src_scale = 1;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        scale = src_scale / dst_scale;
        vscale = GiFloat32Type2FixLenType(GiBroadcastFloat32(scale));
    }

    void cvt(const float* src, int8_t* dst) {
        GI_FLOAT32_t t;
        t = GiFixLenType2GiFloat32Type(vscale);
        GI_FLOAT32_t vitem0 = GiMultiplyFloat32(GiLoadFloat32(src), t);
        GI_FLOAT32_t vitem1 = GiMultiplyFloat32(GiLoadFloat32(src + SIMD_STEP), t);

        GI_FLOAT32_V2_t v2;
        GiSetSubVectorFloat32V2(v2, 0, vitem0);
        GiSetSubVectorFloat32V2(v2, 1, vitem1);
        auto vres = QConverter::convert<GI_INT8_t, GI_FLOAT32_V2_t>(v2);
        GiStoreLowInt8(dst, vres);
    }

    void cvt_remain(const float* src, int8_t* dst) {
        *dst = saturate<int8_t, float>(std::round(*src * scale), -128.f, 127.f);
    }
};

template <>
struct QuantizedTypeCvter<int32_t, int32_t> {
    using stype = int32_t;
    using dst_type = int32_t;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int32_t);
    float scale;
    GI_FLOAT32_FIXLEN_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS32>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS32>().scale;
        scale = src_scale / dst_scale;
        vscale = GiFloat32Type2FixLenType(GiBroadcastFloat32(scale));
    }

    void cvt(const int32_t* src, int32_t* dst) {
        GI_FLOAT32_t t;
        t = GiFixLenType2GiFloat32Type(vscale);
        GI_FLOAT32_t vitem = GiMultiplyFloat32(GiCastToFloat32(GiLoadInt32(src)), t);

        auto vres = QConverter::round<GI_INT32_t, GI_FLOAT32_t>(vitem);
        GiStoreInt32(dst, vres);
    }

    void cvt_remain(const int32_t* src, int32_t* dst) {
        *dst = saturate<int32_t, float>(
                std::round(*src * scale), -2147483648.f, 2147483647.f);
    }
};

template <>
struct QuantizedTypeCvter<int8_t, int8_t> {
    using stype = int8_t;
    using dst_type = int8_t;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    float scale;
    GI_FLOAT32_FIXLEN_t vscale;

    QuantizedTypeCvter(DType src_dtype, DType dst_dtype) {
        float src_scale = src_dtype.param<dtype::QuantizedS8>().scale;
        float dst_scale = dst_dtype.param<dtype::QuantizedS8>().scale;
        scale = src_scale / dst_scale;
        vscale = GiFloat32Type2FixLenType(GiBroadcastFloat32(scale));
    }

    void cvt(const int8_t* src, int8_t* dst) {
        GI_FLOAT32_t t;
        t = GiFixLenType2GiFloat32Type(vscale);
        GI_INT8_t data = GiLoadInt8(src);
        GI_INT16_t vitem0 = GiMoveLowLongInt8(data);
        GI_INT16_t vitem1 = GiMoveHighLongInt8(data);
        auto vret0 = GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vitem0)), t);
        auto vret1 = GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vitem0)), t);
        auto vret2 = GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vitem1)), t);
        auto vret3 = GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vitem1)), t);

        GI_FLOAT32_V4_t v4;
        GiSetSubVectorFloat32V4(v4, 0, vret0);
        GiSetSubVectorFloat32V4(v4, 1, vret1);
        GiSetSubVectorFloat32V4(v4, 2, vret2);
        GiSetSubVectorFloat32V4(v4, 3, vret3);
        auto vres = QConverter::convert<GI_INT8_t, GI_FLOAT32_V4_t>(v4);
        GiStoreInt8(dst, vres);
    }

    void cvt_remain(const int8_t* src, int8_t* dst) {
        *dst = saturate<int8_t, float>(std::round(*src * scale), -128.f, 127.f);
    }
};

template <typename ctype, typename dtype>
struct Fix2FloatTypeCvter;

template <typename ctype, typename dtype>
struct Quan2FloatTypeCvter;

template <>
struct Fix2FloatTypeCvter<int16_t, float> {
    using stype = int16_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int16_t);
    static constexpr size_t SIMD_STEP = GI_SIMD_LEN_BYTE / sizeof(float);

    Fix2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const int16_t* src, float* dst) {
        GI_INT16_t vitem = GiLoadInt16(src);
        auto vret0 = GiCastToFloat32(GiMoveLowLongInt16(vitem));
        auto vret1 = GiCastToFloat32(GiMoveHighLongInt16(vitem));
        GiStoreFloat32(dst, vret0);
        GiStoreFloat32(dst + SIMD_STEP, vret1);
    }

    void cvt_remain(const int16_t* src, float* dst) { *dst = *src; }
};

template <>
struct Fix2FloatTypeCvter<int8_t, float> {
    using stype = int8_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    static constexpr size_t SIMD_STEP = GI_SIMD_LEN_BYTE / sizeof(float);

    Fix2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        MEGDNN_MARK_USED_VAR(src_dtype);
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const int8_t* src, float* dst) {
        GI_INT8_t data = GiLoadInt8(src);
        GI_INT16_t vitem0 = GiMoveLowLongInt8(data);
        GI_INT16_t vitem1 = GiMoveHighLongInt8(data);
        auto vret0 = GiCastToFloat32(GiMoveLowLongInt16(vitem0));
        auto vret1 = GiCastToFloat32(GiMoveHighLongInt16(vitem0));
        auto vret2 = GiCastToFloat32(GiMoveLowLongInt16(vitem1));
        auto vret3 = GiCastToFloat32(GiMoveHighLongInt16(vitem1));
        GiStoreFloat32(dst, vret0);
        GiStoreFloat32(dst + SIMD_STEP, vret1);
        GiStoreFloat32(dst + 2 * SIMD_STEP, vret2);
        GiStoreFloat32(dst + 3 * SIMD_STEP, vret3);
    }

    void cvt_remain(const int8_t* src, float* dst) { *dst = *src; }
};

template <>
struct Quan2FloatTypeCvter<int8_t, float> {
    using stype = int8_t;
    using dst_type = float;
    static constexpr size_t SIMD_WIDTH = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    static constexpr size_t SIMD_STEP = GI_SIMD_LEN_BYTE / sizeof(float);
    float _scale = 0.0f;
    GI_FLOAT32_FIXLEN_t vscale;

    Quan2FloatTypeCvter(DType src_dtype, DType dst_dtype) {
        _scale = src_dtype.param<dtype::QuantizedS8>().scale;
        vscale = GiFloat32Type2FixLenType(GiBroadcastFloat32(_scale));
        MEGDNN_MARK_USED_VAR(dst_dtype);
    }

    void cvt(const int8_t* src, float* dst) {
        GI_FLOAT32_t t;
        t = GiFixLenType2GiFloat32Type(vscale);
        GI_INT8_t data = GiLoadInt8(src);
        GI_INT16_t vitem0 = GiMoveLowLongInt8(data);
        GI_INT16_t vitem1 = GiMoveHighLongInt8(data);
        auto vret0 = GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vitem0)), t);
        auto vret1 = GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vitem0)), t);
        auto vret2 = GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vitem1)), t);
        auto vret3 = GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vitem1)), t);

        GiStoreFloat32(dst, vret0);
        GiStoreFloat32(dst + SIMD_STEP, vret1);
        GiStoreFloat32(dst + 2 * SIMD_STEP, vret2);
        GiStoreFloat32(dst + 3 * SIMD_STEP, vret3);
    }
    void cvt_remain(const int8_t* src, float* dst) { *dst = *src * _scale; }
};

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

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
