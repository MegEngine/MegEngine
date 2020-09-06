/**
 * \file src/core/impl/dtype.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"
#include "megbrain/dtype.h"
#include "megbrain/exception.h"
#include "megbrain/utils/metahelper.h"
#include "megbrain/utils/arith_helper.h"
#include "megdnn/dtype.h"

#include <cmath>
#include <cstring>

using namespace mgb;

namespace {

    template<bool integral_diff>
    struct SafeCastFloatCheck;

    template<>
    struct SafeCastFloatCheck<false> {
        template<typename U>
        static void check(U val) {
            MGB_MARK_USED_VAR(val);
        }
    };

    template<>
    struct SafeCastFloatCheck<true> {
        static void check(float val) {
            mgb_throw_if(fabs(val) > 16777216 || ceilf(val) != val,
                    ConversionError,
                    "can not convert float value %g to int "
                    "without precession loss", val);
        }

        static void check(int val) {
            mgb_throw_if(abs(val) > 16777216,
                    ConversionError,
                    "can not convert int value %d to float "
                    "without precession loss", val);
        }
    };

    template<class T, class U>
    T static_cast_safe(U from) {
        constexpr bool integral_diff =
            (std::is_integral<T>::value ^ std::is_integral<U>::value) &&
            !(std::is_same<T, bool>::value);
        SafeCastFloatCheck<integral_diff>::check(from);
        return static_cast<T>(from);
    }

    template <typename T>
    using QuantizedCType = std::enable_if_t<
            DTypeTrait<T>::category == DTypeCategory::QUANTIZED, T>;

    template <typename T, typename U>
    void batched_static_cast(T* dest, const U* src, size_t nr,
                             DType src_dtype) {
        for (size_t i = 0; i < nr; ++i)
            dest[i] = static_cast<T>(src[i]);
    }

    template <typename T, typename U>
    void batched_static_cast(T* dest, const QuantizedCType<U>* src, size_t nr,
                             DType src_dtype) {
        const auto& param = src_dtype.param<typename DTypeTrait<U>::dtype>();
        for (size_t i = 0; i < nr; ++i) {
            dest[i] = static_cast<T>(param.dequantize(src[i]));
        }
    }

#define cb(_name, _bits)                                                    \
    template <typename T>                                                   \
    void batched_static_cast(T* dest, const megdnn::dt_##_name##_bits* src, \
                             size_t nr, DType src_dtype) {                  \
        std::unique_ptr<int8_t[]> unpacked_byte(new int8_t[nr]);            \
        lowbit_memcpy_compact2byte(megdnn::dtype::_name##_bits(),           \
                                   unpacked_byte.get(), src, nr);           \
        for (size_t i = 0; i < nr; ++i)                                     \
            dest[i] = static_cast<T>(unpacked_byte[i]);                     \
    }
    MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb

    template <typename T, typename U>
    void batched_static_cast_safe(T* dest, const U* src, size_t nr,
                                  DType src_dtype) {
        for (size_t i = 0; i < nr; ++i)
            dest[i] = static_cast_safe<T>(src[i]);
    }

    template <typename T, typename U>
    void batched_static_cast_safe(T* dest, const QuantizedCType<U>* src,
                                  size_t nr, DType src_dtype) {
        const auto& param = src_dtype.param<typename DTypeTrait<U>::dtype>();
        for (size_t i = 0; i < nr; ++i) {
            dest[i] = static_cast_safe<T>(param.dequantize(src[i]));
        }
    }

#define cb(_name, _bits)                                                \
    template <typename T>                                               \
    void batched_static_cast_safe(T* dest,                              \
                                  const megdnn::dt_##_name##_bits* src, \
                                  size_t nr, DType src_dtype) {         \
        std::unique_ptr<int8_t[]> unpacked_byte(new int8_t[nr]);        \
        lowbit_memcpy_compact2byte(megdnn::dtype::_name##_bits(),       \
                                   unpacked_byte.get(), src, nr);       \
        for (size_t i = 0; i < nr; ++i)                                 \
            dest[i] = static_cast_safe<T>(unpacked_byte[i]);            \
    }
    MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb

} // anonymous namespace

template <typename T>
void mgb::static_cast_dtype(T* dest, DType src_type, const void* storage,
                            size_t nr_elem) {
    switch (src_type.enumv()) {
#define cb(_dt)                                                            \
    case DTypeTrait<_dt>::enumv:                                           \
        return batched_static_cast<T, DTypeTrait<_dt>::ctype>(             \
                dest, static_cast<const DTypeTrait<_dt>::ctype*>(storage), \
                nr_elem, src_type);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
#define cb(_name, _bits)                                                    \
    case DTypeTrait<dtype::_name##_bits>::enumv:                            \
        return batched_static_cast(                                         \
                dest,                                                       \
                static_cast<const DTypeTrait<dtype::_name##_bits>::ctype*>( \
                        storage),                                           \
                nr_elem, src_type);
        MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb

        default:
            mgb_throw(ConversionError, "can not convert from dtype %s",
                      src_type.name());
    }
}

template <typename T>
void mgb::static_cast_dtype_safe(T* dest, DType src_type, const void* storage,
                                 size_t nr_elem) {
    switch (src_type.enumv()) {
#define cb(_dt)                                                            \
    case DTypeTrait<_dt>::enumv:                                           \
        return batched_static_cast_safe<T, DTypeTrait<_dt>::ctype>(        \
                dest, static_cast<const DTypeTrait<_dt>::ctype*>(storage), \
                nr_elem, src_type);
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
#define cb(_name, _bits)                                                    \
    case DTypeTrait<dtype::_name##_bits>::enumv:                            \
        return batched_static_cast_safe(                                    \
                dest,                                                       \
                static_cast<const DTypeTrait<dtype::_name##_bits>::ctype*>( \
                        storage),                                           \
                nr_elem, src_type);
        MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb

        default:
            mgb_throw(ConversionError, "can not convert from dtype %s",
                      src_type.name());
    }
}

namespace mgb {

#define INST(t)                                                         \
    template void static_cast_dtype<t>(t*, DType, const void*, size_t); \
    template void static_cast_dtype_safe<t>(t*, DType, const void*, size_t)
INST(bool);
INST(unsigned);
INST(int);
INST(unsigned long);
INST(long);
INST(float);
INST(double);
INST(long long);
INST(unsigned long long);
#undef INST

template<typename ctype>
typename ctype_enable_if<ctype>::type DTypeScalar::set_retain_dtype(ctype val) {
    switch (m_dtype.enumv()) {
#define cb(_dt) \
        case DTypeTrait<_dt>::enumv: { \
            using mct = DTypeTrait<_dt>::ctype; \
            static_assert(sizeof(mct) <= sizeof(m_storage), "large ctype"); \
            visit<mct>() = static_cast<mct>(val); \
            return; \
        }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
        default:
            mgb_throw(ConversionError,
                    "can not assign to dtype %s", m_dtype.name());
    }
}

#define INST(t) template void DTypeScalar::set_retain_dtype<t>(t);
INST(int);
INST(float);
#undef INST

}

DTypeScalar& DTypeScalar::set_raw(DType dtype, const void* storage) {
    mgb_assert(dtype.valid() && dtype.size(1) <= sizeof(m_storage));
    m_dtype = dtype;
    memcpy(&m_storage, storage, dtype.size(1));
    return *this;
}

DType mgb::dtype_promotion(DType t0, DType t1) {
    mgb_assert(t0 != dtype::Byte() && t1 != dtype::Byte());
    if (t0 == t1)
        return t0;

    // Now t0 != t1.
    if (t0.category() == DTypeCategory::QUANTIZED &&
        t1.category() == DTypeCategory::QUANTIZED) {
        mgb_assert(t0.enumv() == t1.enumv(),
                   "promoting unexpected quantized DType: %s and %s", t0.name(),
                   t1.name());
        if (t0.enumv() == DTypeEnum::Quantized8Asymm) {
            auto& param0 = t0.param<dtype::Quantized8Asymm>();
            auto& param1 = t1.param<dtype::Quantized8Asymm>();
            mgb_assert(param0.zero_point == param1.zero_point &&
                           fabs(param0.scale - param1.scale) < 1e-6,
                   "trying to promote two Quantized8Asymm with different scale "
                   "or zero_point, this usually does not make sense: (%f, %u) "
                   "vs (%f, %u)",
                   param0.scale, param0.zero_point, param1.scale,
                   param1.zero_point);
            return t0;
        } else if (t0.enumv() == DTypeEnum::QuantizedS8) {
            auto& param0 = t0.param<dtype::QuantizedS8>();
            auto& param1 = t1.param<dtype::QuantizedS8>();
            mgb_assert(fabs(param0.scale - param1.scale) < 1e-6,
                       "trying to promote two QuantizedS8 with different "
                       "scale, this usually does not make sense: %f vs %f",
                       param0.scale, param1.scale);
            return t0;
        } else {
            mgb_assert(t0.enumv() == DTypeEnum::QuantizedS32,
                       "promoting unsupported quantized DType: %s", t0.name());
            auto& param0 = t0.param<dtype::QuantizedS32>();
            auto& param1 = t1.param<dtype::QuantizedS32>();
            mgb_assert(fabs(param0.scale - param1.scale) < 1e-6,
                       "trying to promote two QuantizedS32 with different "
                       "scale, this usually does not make sense: %f vs %f",
                       param0.scale, param1.scale);
            return t0;
        }
    } else if (t0.category() == DTypeCategory::QUANTIZED) {
        return t0;
    } else if (t1.category() == DTypeCategory::QUANTIZED) {
        return t1;
    }

#if !MEGDNN_DISABLE_FLOAT16
    if (t0 == dtype::Float16())
        t0 = dtype::Float32();

    if (t1 == dtype::Float16())
        t1 = dtype::Float32();
#endif

    if (t0.category() != t1.category()) {
        return dtype::Float32();
    }

    mgb_throw_if(t0.signedness() != t1.signedness(),
            ConversionError,
            "dtype promotion rule between different signedness is undefined: "
            "%s %s", t0.name(), t1.name());

    if (t0.size() > t1.size())
        return t0;
    return t1;
}

/* ================== lowbit memcpy ================== */

namespace {

template<int bits, bool div_byte = 8 % bits == 0>
struct LowbitMemcpy;

template<int bits>
struct LowbitTrait;

template<>
struct LowbitTrait<1> {
    // intb1: -1, 1
    static constexpr int8_t SHIFT = 1, STEP = 2;
};

template<>
struct LowbitTrait<2> {
    // intb2: -3, -1, 1, 3
    static constexpr int8_t SHIFT = 3, STEP = 2;
};

template<>
struct LowbitTrait<4> {
    // intb2: -15 to 15
    static constexpr int8_t SHIFT = 15, STEP = 2;
};

template<int bits>
struct LowbitMemcpy<bits, true> {
    // cast with bits that 8 % bits == 0

    static constexpr uint8_t MASK = (1 << bits) - 1;
    using Trait = LowbitTrait<bits>;

    static void byte2compact(
            void *dest_raw, const void *src_raw, size_t n) {
        auto dest = static_cast<uint8_t*>(dest_raw);
        auto src = static_cast<const int8_t*>(src_raw);
        memset(dest, 0, divup<size_t>(n * bits, 8));
        for (size_t i = 0; i < n; ++ i) {
            int8_t val = src[i];
            mgb_assert(val + Trait::SHIFT >= 0 &&
                    ((val + Trait::SHIFT) % Trait::STEP) == 0);
            val = (val + Trait::SHIFT) / Trait::STEP;
            mgb_assert(val >= 0 && val < (1 << bits));
            dest[i * bits / 8] |= val << (i * bits % 8);
        }
    }
    static void compact2byte(
            void *dest_raw, const void *src_raw, size_t n) {
        auto dest = static_cast<int8_t*>(dest_raw);
        auto src = static_cast<const uint8_t*>(src_raw);
        for (size_t i = 0; i < n; ++ i) {
            int8_t val = ((src[i * bits / 8] >> (i * bits % 8)) & MASK);
            dest[i] = val * Trait::STEP - Trait::SHIFT;
        }
    }
};

template<typename DT>
struct QuantizedLowbitTrait;

template<>
struct QuantizedLowbitTrait<dtype::Quantized4Asymm> {
    static constexpr int8_t SHIFT = 0;
};

template<>
struct QuantizedLowbitTrait<dtype::QuantizedS4> {
    static constexpr int8_t SHIFT = 8;
};

template <typename DT, bool div_byte = (DTypeTrait<DT>::category ==
                                        DTypeCategory::QUANTIZED) &&
                                       (8 % DTypeTrait<DT>::low_bit == 0)>
struct QuantizedLowbitMemcpy;

template <typename DT>
struct QuantizedLowbitMemcpy<DT, true> {
    // cast with bits that 8 % bits == 0
    static constexpr uint16_t bits = DTypeTrait<DT>::low_bit;
    static constexpr uint8_t MASK = (1 << bits) - 1;
    using Trait = QuantizedLowbitTrait<DT>;

    static void byte2compact(void* dest_raw, const void* src_raw, size_t n) {
        auto dest = static_cast<uint8_t*>(dest_raw);
        auto src = static_cast<const int8_t*>(src_raw);
        memset(dest, 0, divup<size_t>(n * bits, 8));
        for (size_t i = 0; i < n; ++i) {
            int8_t val = src[i] + Trait::SHIFT;
            mgb_assert(val >= 0 && val < (1 << bits));
            dest[i * bits / 8] |= val << (i * bits % 8);
        }
    }
    static void compact2byte(void* dest_raw, const void* src_raw, size_t n) {
        auto dest = static_cast<int8_t*>(dest_raw);
        auto src = static_cast<const uint8_t*>(src_raw);
        for (size_t i = 0; i < n; ++i) {
            int8_t val = ((src[i * bits / 8] >> (i * bits % 8)) & MASK);
            dest[i] = val - Trait::SHIFT;
        }
    }
};

} // anonymous namespace

void mgb::lowbit_memcpy_byte2compact(
        DType dtype, void *dest, const void *src, size_t n) {
#define cb(name, bits) \
    if (dtype == mgb::dtype::name##bits()) \
        return LowbitMemcpy<bits>::byte2compact(dest, src, n);
    MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb
#define cb(dt) \
    if (dtype.enumv() == DTypeTrait<dt>::enumv) \
        return QuantizedLowbitMemcpy<dt>::byte2compact(dest, src, n);
    MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb
    mgb_throw(MegBrainError, "bad dtype for lowbit: %s", dtype.name());
}

void mgb::lowbit_memcpy_compact2byte(
        DType dtype, void *dest, const void *src, size_t n) {
#define cb(name, bits) \
    if (dtype == mgb::dtype::name##bits()) \
        return LowbitMemcpy<bits>::compact2byte(dest, src, n);
    MEGDNN_FOREACH_LOWBIT_DTYPE(cb)
#undef cb
#define cb(dt) \
    if (dtype.enumv() == DTypeTrait<dt>::enumv) \
        return QuantizedLowbitMemcpy<dt>::compact2byte(dest, src, n);
    MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb)
#undef cb
    mgb_throw(MegBrainError, "bad dtype for lowbit: %s", dtype.name());
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
