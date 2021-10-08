/**
 * \file dnn/src/fallback/type_cvt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/type_cvt/opr_impl.h"

#include "midout.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

// MIDOUT_DECL(megdnn_fb_typecvt_src)
MIDOUT_DECL(megdnn_fb_typecvt_dst_dtype)
MIDOUT_DECL(megdnn_fb_typecvt_src_dtype)

namespace {

using namespace megdnn;

template <typename stype, typename dtype>
struct TypeCvt {
    static void do_cvt(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
        using sctype = typename DTypeTrait<stype>::ctype;
        using dctype = typename DTypeTrait<dtype>::ctype;
        auto n = src.layout.total_nr_elems();
        const sctype* __restrict sptr = src.ptr<sctype>();
        dctype* __restrict dptr = dst.ptr<dctype>();
        for (size_t i = 0; i < n; ++i) {
            dptr[i] = static_cast<dctype>(sptr[i]);
        }
    }
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

//! As aarch32 __fp16 vectorize may cause llvm error, so if macro \c
//! MEGDNN_FIX_AARCH32_BUG defined, we use dt_float16, otherwise __fp16
#if MEGDNN_FIX_AARCH32_BUG
#define FLOAT16 dt_float16
#else
#define FLOAT16 __fp16
#endif
template <typename stype>
struct TypeCvt<stype, dtype::Float16> {
    static void do_cvt(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
        using sctype = typename DTypeTrait<stype>::ctype;
        auto n = src.layout.total_nr_elems();
        const sctype* __restrict sptr = src.ptr<sctype>();
        FLOAT16* __restrict dptr = static_cast<FLOAT16*>(dst.raw_ptr);
        for (size_t i = 0; i < n; ++i) {
            dptr[i] = static_cast<FLOAT16>(sptr[i]);
        }
    }
};

template <typename dst_type>
struct TypeCvt<dtype::Float16, dst_type> {
    static void do_cvt(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
        auto n = src.layout.total_nr_elems();
        using dctype = typename DTypeTrait<dst_type>::ctype;
        const FLOAT16* __restrict sptr = static_cast<FLOAT16*>(src.raw_ptr);
        dctype* __restrict dptr = dst.ptr<dctype>();
        for (size_t i = 0; i < n; ++i) {
            dptr[i] = static_cast<FLOAT16>(sptr[i]);
        }
    }
};

template <>
struct TypeCvt<dtype::Float16, dtype::Float16> {
    static void do_cvt(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
        auto n = src.layout.total_nr_elems();
        const FLOAT16* __restrict sptr = static_cast<FLOAT16*>(src.raw_ptr);
        FLOAT16* __restrict dptr = static_cast<FLOAT16*>(dst.raw_ptr);
        for (size_t i = 0; i < n; ++i) {
            dptr[i] = static_cast<FLOAT16>(sptr[i]);
        }
    }
};

#undef FLOAT16

#endif

template <typename stype>
void do_cvt_normal_s8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    using sctype = typename DTypeTrait<stype>::ctype;
    auto n = src.layout.total_nr_elems();
    const sctype* __restrict sptr = src.ptr<sctype>();
    int8_t* __restrict dptr = static_cast<int8_t*>(dst.raw_ptr);
    float scale = dst.layout.dtype.param<dtype::QuantizedS8>().scale;
    float dscale = 1.f / scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int8_t, float>(std::round(sptr[i] * dscale), -128, 127);
    }
}

template <typename stype>
void do_cvt_normal_s32(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    using sctype = typename DTypeTrait<stype>::ctype;
    auto n = src.layout.total_nr_elems();
    const sctype* __restrict sptr = src.ptr<sctype>();
    int32_t* __restrict dptr = static_cast<int32_t*>(dst.raw_ptr);
    float scale = dst.layout.dtype.param<dtype::QuantizedS32>().scale;
    float dscale = 1.f / scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int32_t, float>(
                std::round(sptr[i] * dscale),
                static_cast<float>(std::numeric_limits<int32_t>::min()),
                static_cast<float>(std::numeric_limits<int32_t>::max()));
    }
}

template <typename stype>
void do_cvt_normal_asymm8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    using sctype = typename DTypeTrait<stype>::ctype;
    auto n = src.layout.total_nr_elems();
    const sctype* __restrict sptr = src.ptr<sctype>();
    uint8_t* __restrict dptr = static_cast<uint8_t*>(dst.raw_ptr);
    float scale = dst.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t zp = dst.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float dscale = 1.f / scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<uint8_t, float>(std::round(sptr[i] * dscale) + zp, 0, 255);
    }
}

template <typename type>
void do_cvt_s8_normal(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    using dctype = typename DTypeTrait<type>::ctype;
    auto n = src.layout.total_nr_elems();
    const int8_t* __restrict sptr = static_cast<int8_t*>(src.raw_ptr);
    dctype* __restrict dptr = dst.ptr<dctype>();
    float scale = src.layout.dtype.param<dtype::QuantizedS8>().scale;
    for (size_t i = 0; i < n; ++i) {
        auto val = sptr[i] * scale;
        dptr[i] = static_cast<dctype>(val);
    }
}

template <typename type>
void do_cvt_s32_normal(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    using dctype = typename DTypeTrait<type>::ctype;
    auto n = src.layout.total_nr_elems();
    const int32_t* __restrict sptr = static_cast<int32_t*>(src.raw_ptr);
    dctype* __restrict dptr = dst.ptr<dctype>();
    float scale = src.layout.dtype.param<dtype::QuantizedS32>().scale;
    for (size_t i = 0; i < n; ++i) {
        auto val = sptr[i] * scale;
        dptr[i] = static_cast<dctype>(val);
    }
}

template <typename type>
void do_cvt_asymm8_normal(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    using dctype = typename DTypeTrait<type>::ctype;
    auto n = src.layout.total_nr_elems();
    const uint8_t* __restrict sptr = static_cast<uint8_t*>(src.raw_ptr);
    dctype* __restrict dptr = dst.ptr<dctype>();
    float scale = src.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t zp = src.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    for (size_t i = 0; i < n; ++i) {
        auto val = (sptr[i] - zp) * scale;
        dptr[i] = static_cast<dctype>(val);
    }
}

void do_cvt_s8_s8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const int8_t* __restrict sptr = static_cast<int8_t*>(src.raw_ptr);
    int8_t* __restrict dptr = static_cast<int8_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::QuantizedS8>().scale;
    float dst_scale = dst.layout.dtype.param<dtype::QuantizedS8>().scale;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int8_t, float>(std::round(sptr[i] * scale), -128, 127);
    }
}

void do_cvt_s32_s8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const int32_t* __restrict sptr = static_cast<int32_t*>(src.raw_ptr);
    int8_t* __restrict dptr = static_cast<int8_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::QuantizedS32>().scale;
    float dst_scale = dst.layout.dtype.param<dtype::QuantizedS8>().scale;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int8_t, float>(std::round(sptr[i] * scale), -128, 127);
    }
}

void do_cvt_asymm8_s8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const uint8_t* __restrict sptr = static_cast<uint8_t*>(src.raw_ptr);
    int8_t* __restrict dptr = static_cast<int8_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t src_zp = src.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float dst_scale = dst.layout.dtype.param<dtype::QuantizedS8>().scale;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int8_t, float>(
                std::round((sptr[i] - src_zp) * scale), -128, 127);
    }
}

void do_cvt_s8_s32(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const int8_t* __restrict sptr = static_cast<int8_t*>(src.raw_ptr);
    int32_t* __restrict dptr = static_cast<int32_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::QuantizedS8>().scale;
    float dst_scale = dst.layout.dtype.param<dtype::QuantizedS32>().scale;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int32_t, float>(
                std::round(sptr[i] * scale),
                static_cast<float>(std::numeric_limits<int32_t>::min()),
                static_cast<float>(std::numeric_limits<int32_t>::max()));
    }
}

void do_cvt_s32_s32(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const int32_t* __restrict sptr = static_cast<int32_t*>(src.raw_ptr);
    int32_t* __restrict dptr = static_cast<int32_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::QuantizedS32>().scale;
    float dst_scale = dst.layout.dtype.param<dtype::QuantizedS32>().scale;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int32_t, float>(
                std::round(sptr[i] * scale),
                static_cast<float>(std::numeric_limits<int32_t>::min()),
                static_cast<float>(std::numeric_limits<int32_t>::max()));
    }
}

void do_cvt_asymm8_s32(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const uint8_t* __restrict sptr = static_cast<uint8_t*>(src.raw_ptr);
    int32_t* __restrict dptr = static_cast<int32_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t src_zp = src.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float dst_scale = dst.layout.dtype.param<dtype::QuantizedS32>().scale;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<int32_t, float>(
                std::round((sptr[i] - src_zp) * scale),
                static_cast<float>(std::numeric_limits<int32_t>::min()),
                static_cast<float>(std::numeric_limits<int32_t>::max()));
    }
}

void do_cvt_s8_asymm8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const int8_t* __restrict sptr = static_cast<int8_t*>(src.raw_ptr);
    uint8_t* __restrict dptr = static_cast<uint8_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::QuantizedS8>().scale;
    float dst_scale = dst.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t dst_zp = dst.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] =
                saturate<uint8_t, float>(std::round(sptr[i] * scale) + dst_zp, 0, 255);
    }
}

void do_cvt_s32_asymm8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const int32_t* __restrict sptr = static_cast<int32_t*>(src.raw_ptr);
    uint8_t* __restrict dptr = static_cast<uint8_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::QuantizedS32>().scale;
    float dst_scale = dst.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t dst_zp = dst.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] =
                saturate<uint8_t, float>(std::round(sptr[i] * scale) + dst_zp, 0, 255);
    }
}

void do_cvt_asymm8_asymm8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    auto n = src.layout.total_nr_elems();
    const uint8_t* __restrict sptr = static_cast<uint8_t*>(src.raw_ptr);
    int8_t* __restrict dptr = static_cast<int8_t*>(dst.raw_ptr);
    float src_scale = src.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t src_zp = src.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float dst_scale = dst.layout.dtype.param<dtype::Quantized8Asymm>().scale;
    uint8_t dst_zp = dst.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
    float scale = src_scale / dst_scale;
    for (size_t i = 0; i < n; ++i) {
        dptr[i] = saturate<uint8_t, float>(
                std::round((sptr[i] - src_zp) * scale) + dst_zp, 0, 255);
    }
}

template <typename dtype>
void on_dest_ctype(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                                        \
    case DTypeTrait<_dt>::enumv: {                                                     \
        MIDOUT_BEGIN(megdnn_fb_typecvt_src_dtype, midout_iv(DTypeTrait<_dt>::enumv)) { \
            TypeCvt<_dt, dtype>::do_cvt(src, dst);                                     \
        }                                                                              \
        MIDOUT_END();                                                                  \
        break;                                                                         \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
                cb(::megdnn::dtype::Uint16) case DTypeEnum::QuantizedS8
                : MIDOUT_BEGIN(
                          megdnn_fb_typecvt_src_dtype,
                          midout_iv(DTypeEnum::QuantizedS8)) {
            do_cvt_s8_normal<dtype>(src, dst);
        }
        MIDOUT_END();
        break;
        case DTypeEnum::QuantizedS32:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS32)) {
                do_cvt_s32_normal<dtype>(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::Quantized8Asymm:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype,
                    midout_iv(DTypeEnum::Quantized8Asymm)) {
                do_cvt_asymm8_normal<dtype>(src, dst);
            }
            MIDOUT_END();
            break;
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void on_dest_s8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                                        \
    case DTypeTrait<_dt>::enumv: {                                                     \
        MIDOUT_BEGIN(megdnn_fb_typecvt_src_dtype, midout_iv(DTypeTrait<_dt>::enumv)) { \
            do_cvt_normal_s8<_dt>(src, dst);                                           \
        }                                                                              \
        MIDOUT_END();                                                                  \
        break;                                                                         \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        case DTypeEnum::QuantizedS8:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS8)) {
                do_cvt_s8_s8(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::QuantizedS32:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS32)) {
                do_cvt_s32_s8(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::Quantized8Asymm:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype,
                    midout_iv(DTypeEnum::Quantized8Asymm)) {
                do_cvt_asymm8_s8(src, dst);
            }
            MIDOUT_END();
            break;
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void on_dest_s32(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                                        \
    case DTypeTrait<_dt>::enumv: {                                                     \
        MIDOUT_BEGIN(megdnn_fb_typecvt_src_dtype, midout_iv(DTypeTrait<_dt>::enumv)) { \
            do_cvt_normal_s32<_dt>(src, dst);                                          \
        }                                                                              \
        MIDOUT_END();                                                                  \
        break;                                                                         \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        case DTypeEnum::QuantizedS8:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS8)) {
                do_cvt_s8_s32(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::QuantizedS32:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS32)) {
                do_cvt_s32_s32(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::Quantized8Asymm:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype,
                    midout_iv(DTypeEnum::Quantized8Asymm)) {
                do_cvt_asymm8_s32(src, dst);
            }
            MIDOUT_END();
            break;
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void on_dest_asymm8(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                                        \
    case DTypeTrait<_dt>::enumv: {                                                     \
        MIDOUT_BEGIN(megdnn_fb_typecvt_src_dtype, midout_iv(DTypeTrait<_dt>::enumv)) { \
            do_cvt_normal_asymm8<_dt>(src, dst);                                       \
        }                                                                              \
        MIDOUT_END();                                                                  \
        break;                                                                         \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        case DTypeEnum::QuantizedS8:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS8)) {
                do_cvt_s8_asymm8(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::QuantizedS32:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype, midout_iv(DTypeEnum::QuantizedS32)) {
                do_cvt_s32_asymm8(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::Quantized8Asymm:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_src_dtype,
                    midout_iv(DTypeEnum::Quantized8Asymm)) {
                do_cvt_asymm8_asymm8(src, dst);
            }
            MIDOUT_END();
            break;
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

void run_contiguous(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                                                        \
    case DTypeTrait<_dt>::enumv: {                                                     \
        MIDOUT_BEGIN(megdnn_fb_typecvt_dst_dtype, midout_iv(DTypeTrait<_dt>::enumv)) { \
            on_dest_ctype<_dt>(src, dst);                                              \
        }                                                                              \
        MIDOUT_END();                                                                  \
        break;                                                                         \
    }

        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
                cb(::megdnn::dtype::Uint16) case DTypeEnum::QuantizedS8
                : MIDOUT_BEGIN(
                          megdnn_fb_typecvt_dst_dtype,
                          midout_iv(DTypeEnum::QuantizedS8)) {
            on_dest_s8(src, dst);
        }
        MIDOUT_END();
        break;
        case DTypeEnum::QuantizedS32:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_dst_dtype, midout_iv(DTypeEnum::QuantizedS32)) {
                on_dest_s32(src, dst);
            }
            MIDOUT_END();
            break;
        case DTypeEnum::Quantized8Asymm:
            MIDOUT_BEGIN(
                    megdnn_fb_typecvt_dst_dtype,
                    midout_iv(DTypeEnum::Quantized8Asymm)) {
                on_dest_asymm8(src, dst);
            }
            MIDOUT_END();
            break;
#undef cb
        default:
            megdnn_throw("bad dtype");
    }
}

}  // anonymous namespace

namespace megdnn {
namespace fallback {

void TypeCvtImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst) {
    check_exec(src.layout, dst.layout);
    auto is_quantize_lowbit = [](const DType& dt) {
        return dt.category() == DTypeCategory::QUANTIZED && dt.is_low_bit();
    };
    if (src.layout.is_contiguous() && dst.layout.is_contiguous() &&
        !is_quantize_lowbit(src.layout.dtype) &&
        !is_quantize_lowbit(dst.layout.dtype)) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(run_contiguous(src, dst));
    } else {
        naive::TypeCvtImpl::exec(src, dst);
    }
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
