/**
 * \file dnn/include/megdnn/dtype.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megdnn/arch.h"

#include <stdint.h>
#include <cfloat>
#include <cstddef>
#include <limits>

#ifdef MEGDNN_CC_HOST
#include <cmath>
#include <utility>
#endif

#include "megdnn/internal/visibility_prologue.h"

#if MEGDNN_DISABLE_FLOAT16
#define MEGDNN_INC_FLOAT16(_x)
#define MEGDNN_FLOAT16_SELECT(_x, _y)   _y
#else
#include "megdnn/dtype/half.hpp"
#define MEGDNN_INC_FLOAT16(_x) _x
#define MEGDNN_FLOAT16_SELECT(_x, _y)   _x
#endif

namespace megdnn {

/*!
 * \brief iterate through each dtype name
 */
#define MEGDNN_FOREACH_DTYPE_NAME(cb) \
    cb(Float32) \
    cb(Uint8) \
    cb(Int8) \
    cb(Int16) \
    cb(Int32) \
    cb(IntB1) \
    cb(IntB2) \
    cb(IntB4) \
    cb(Byte) \
    MEGDNN_INC_FLOAT16(cb(Float16)) \
    cb(UintB4) \

/*!
 * \brief iterate through each full byte dtype
 */
#define MEGDNN_FOREACH_FULL_BYTE_DTYPE(cb) \
    cb(Float32) \
    cb(Uint8) \
    cb(Int8) \
    cb(Int16) \
    cb(Int32) \
    cb(Byte) \
    MEGDNN_INC_FLOAT16(cb(Float16)) \

/*!
 * \brief iterate through each fractional byte dtype
 */
#define MEGDNN_FOREACH_LOWBIT_DTYPE(cb) \
    cb(IntB, 1)\
    cb(IntB, 2)\
    cb(IntB, 4)\
    cb(UintB, 4)\

// This is used to make enum definition possible.
#define MEGDNN_FOREACH_PARAMETERIZED_DTYPE_FIRST(cb) \
    cb(Quantized8Asymm)

#define MEGDNN_FOREACH_PARAMETERIZED_DTYPE_OTHERS(cb) \
    cb(QuantizedS32) \
    cb(QuantizedS8) \
    cb(Quantized4Asymm) \
    cb(QuantizedS4) \
    cb(QuantizedS16)

#define MEGDNN_FOREACH_PARAMETERIZED_DTYPE_2(cb_first, cb_others) \
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE_FIRST(cb_first) \
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE_OTHERS(cb_others)

/*!
 * \brief iterate through each parameterized dtype
 */
#define MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb) \
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE_FIRST(cb) \
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE_OTHERS(cb)

/*!
 * \brief iterate through each dtype object that can be involved in float
 *      numeric computing
 */
#define MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb) \
    cb(::megdnn::dtype::Float32) \
    MEGDNN_INC_FLOAT16(cb(::megdnn::dtype::Float16)) \

/*!
 * \brief iterate through each dtype object that can be involved in integer
 *      numeric computing
 */
#define MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb) \
    cb(::megdnn::dtype::Int32) \
    cb(::megdnn::dtype::Int16) \
    cb(::megdnn::dtype::Int8) \
    cb(::megdnn::dtype::Uint8) \

/*!
 * \brief iterate through each dtype object that can be involved in numeric
 *      computing (i.e. dtypes except Byte)
 */
#define MEGDNN_FOREACH_COMPUTING_DTYPE(cb) \
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb) \
    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)

//! In order to avoid an unnecessary increase in binary size, we just
//! use QuantizedS16 dtype in winograd_filter_preprocess now. So I didn't add
//! this data type here.
#define MEGDNN_FOREACH_QUANTIZED_DTYPE(cb) \
    cb(::megdnn::dtype::Quantized8Asymm) \
    cb(::megdnn::dtype::QuantizedS32) \
    cb(::megdnn::dtype::QuantizedS8) \

#define MEGDNN_FOREACH_QUANTIZED_LOWBIT_DTYPE(cb) \
    cb(::megdnn::dtype::Quantized4Asymm) \
    cb(::megdnn::dtype::QuantizedS4)

#define MEGDNN_FOREACH_QUANTIZED_DTYPE_SYMM(cb) \
    cb(::megdnn::dtype::QuantizedS32) \
    cb(::megdnn::dtype::QuantizedS8) \
    cb(::megdnn::dtype::QuantizedS4)

#define MEGDNN_FOREACH_QUANTIZED_DTYPE_ASYMM(cb) \
    cb(::megdnn::dtype::Quantized8Asymm) \
    cb(::megdnn::dtype::Quantized4Asymm)

/*!
 * \brief a POD representation of a single byte
 *
 * Byte is used as storage of unspecific raw data, and should not be involved in
 * any computing.
 */
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
class dt_byte {
    unsigned char _;

    public:

        //! convert to given type
        template<typename T>
        T* as() {
            return reinterpret_cast<T*>(this);
        }

        //! convert to given type
        template<typename T>
        const T* as() const {
            return reinterpret_cast<const T*>(this);
        }
} MEGDNN_PACKED;

#define DEFINE_LOWBIT(_name, b) \
    class dt_##_name##b {\
        unsigned char _;\
    } MEGDNN_PACKED;
MEGDNN_FOREACH_LOWBIT_DTYPE(DEFINE_LOWBIT)
#undef DEFINE_LOWBIT

class dt_quint8 {
    uint8_t _;

    public:
        //! Convert to normal uint8_t
        MEGDNN_DEVICE uint8_t as_uint8() const {
            return _;
        }

        MEGDNN_HOST MEGDNN_DEVICE explicit dt_quint8(uint8_t val):_(val) {}
#ifdef MEGDNN_CC_HOST
        explicit operator uint8_t() { return _; }
#endif
        bool operator<(const dt_quint8& b) const { return _ < b._; }
        bool operator>(const dt_quint8& b) const { return _ > b._; }
} MEGDNN_PACKED;

class dt_qint32 {
    int32_t _;

    public:
        //! Convert to normal uint32_t
        MEGDNN_DEVICE int32_t as_int32() const {
            return _;
        }

        MEGDNN_HOST MEGDNN_DEVICE explicit dt_qint32(int32_t val):_(val) {}
#ifdef MEGDNN_CC_HOST
        explicit operator int32_t() { return _; }
#endif
        dt_qint32 operator*(const dt_qint32& b) const {
            return dt_qint32(_ * b._);
        }
        dt_qint32 operator+(const dt_qint32& b) const {
            return dt_qint32(_ + b._);
        }
        dt_qint32 operator-(const dt_qint32& b) const {
            return dt_qint32(_ - b._);
        }
#ifdef MEGDNN_CC_HOST
        dt_qint32 operator/(int b) const {
            return dt_qint32(std::round(_ / static_cast<float>(b)));
        }
        dt_qint32 operator/(const dt_qint32& b) const {
            return dt_qint32(std::round(_ / static_cast<float>(b._)));
        }
#endif
        dt_qint32 operator+=(const dt_qint32& b) {
            _ += b._;
            return *this;
        }
        bool operator<(const dt_qint32& b) const { return _ < b._; }
        bool operator>(const dt_qint32& b) const { return _ > b._; }
} MEGDNN_PACKED;

class dt_qint8 {
    int8_t _;

    public:
        MEGDNN_DEVICE int8_t as_int8() const {
            return _;
        }

        MEGDNN_HOST MEGDNN_DEVICE explicit dt_qint8(int8_t val):_(val) {}
#ifdef MEGDNN_CC_HOST
        explicit operator int8_t() { return _; }
#endif
        bool operator<(const dt_qint8& b) const { return _ < b._; }
        bool operator>(const dt_qint8& b) const { return _ > b._; }
} MEGDNN_PACKED;

class dt_qint16 {
    int16_t _;

    public:
        //! Convert to normal int16_t
        MEGDNN_DEVICE int16_t as_int16() const {
            return _;
        }

        MEGDNN_HOST MEGDNN_DEVICE explicit dt_qint16(int16_t val):_(val) {}
#ifdef MEGDNN_CC_HOST
        explicit operator int16_t() { return _; }
#endif
        dt_qint16 operator*(const dt_qint16& b) const {
            return dt_qint16(_ * b._);
        }
        dt_qint16 operator+(const dt_qint16& b) const {
            return dt_qint16(_ + b._);
        }
        dt_qint16 operator-(const dt_qint16& b) const {
            return dt_qint16(_ - b._);
        }
#ifdef MEGDNN_CC_HOST
        dt_qint16 operator/(int b) const {
            return dt_qint16(std::round(_ / static_cast<float>(b)));
        }
        dt_qint16 operator/(const dt_qint16& b) const {
            return dt_qint16(std::round(_ / static_cast<float>(b._)));
        }
#endif
        dt_qint16 operator+=(const dt_qint16& b) {
            _ += b._;
            return *this;
        }
        bool operator<(const dt_qint16& b) const { return _ < b._; }
        bool operator>(const dt_qint16& b) const { return _ > b._; }
} MEGDNN_PACKED;

template <uint8_t BITS>
class dt_qulowbit {
    uint8_t _;
    public:
        //! Convert to normal uint8_t
        MEGDNN_DEVICE uint8_t as_uint8() const {
            return _;
        }

        MEGDNN_HOST MEGDNN_DEVICE explicit dt_qulowbit(uint8_t val):_(val) {}
#ifdef MEGDNN_CC_HOST
        explicit operator uint8_t() { return _; }
#endif
        bool operator<(const dt_qulowbit<BITS>& b) const { return _ < b._; }
        bool operator>(const dt_qulowbit<BITS>& b) const { return _ > b._; }

        dt_qulowbit& operator=(const uint8_t val) {
            _ = val;
            return *this;
        }
};
using dt_quint4 = dt_qulowbit<4>;

template <uint8_t BITS>
class dt_qlowbit {
    int8_t _;

    public:
        //! Convert to normal int8_t
        MEGDNN_DEVICE int8_t as_int8() const {
            return _;
        }

        MEGDNN_HOST MEGDNN_DEVICE explicit dt_qlowbit(int8_t val):_(val) {}
#ifdef MEGDNN_CC_HOST
        explicit operator int8_t() { return _; }
#endif
        bool operator<(const dt_qlowbit<BITS>& b) const { return _ < b._; }
        bool operator>(const dt_qlowbit<BITS>& b) const { return _ > b._; }

        dt_qlowbit& operator=(const int8_t val) {
            _ = val;
            return *this;
        }
};
using dt_qint4 = dt_qlowbit<4>;

#ifdef __clang__
#pragma clang diagnostic pop
#endif
MEGDNN_STATIC_ASSERT(sizeof(dt_byte) == 1, "bad dt_byte size");
MEGDNN_STATIC_ASSERT(sizeof(dt_quint8) == 1, "bad dt_quint8 size");
MEGDNN_STATIC_ASSERT(sizeof(dt_qint16) == 2, "bad dt_qint16 size");
MEGDNN_STATIC_ASSERT(sizeof(dt_qint32) == 4, "bad dt_qint32 size");
typedef float dt_float32;
typedef int32_t dt_int32;
typedef int16_t dt_int16;
typedef int8_t dt_int8;
typedef uint8_t dt_uint8;
MEGDNN_INC_FLOAT16(typedef half_float::half dt_float16;)

#define MEGDNN_PARAMETERIZED_DTYPE_ENUM_BASE 100000
#if MEGDNN_CC_HOST
    //! enumeration of dtypes; useful for hash or being used in switch-case
    enum class DTypeEnum: uint32_t {
#else
    struct DTypeEnum {
        enum Ev {
#endif
            Float32,
            Uint8,
            Int8,
            Int16,
            Int32,
            IntB1,
            IntB2,
            IntB4,
            Byte,
#if !MEGDNN_DISABLE_FLOAT16
            Float16,
#endif
            UintB4 = 10,

            #define FST(_name) _name = MEGDNN_PARAMETERIZED_DTYPE_ENUM_BASE,
            #define D(_name) _name,
            MEGDNN_FOREACH_PARAMETERIZED_DTYPE_2(FST, D)
            #undef D
            #undef FST
#if !MEGDNN_CC_HOST
        };
        uint32_t ev;
        DTypeEnum(): ev(0) {}
        DTypeEnum(uint32_t e): ev(e) {}
#endif
    };

#if MEGDNN_CC_HOST
    //! dtype numeric category fo
    enum class DTypeCategory: int {
        OTHER, FLOAT, INT, LOWBIT, QUANTIZED
    };
    //! dtype signedness
    enum class DTypeSignedness: int {
        OTHER, UNSIGNED, SIGNED
    };
#else
    struct DTypeCategory {
        enum Ev {
            OTHER, FLOAT, INT, LOWBIT, QUANTIZED
        };
        int ev;
    };
    struct DTypeSignedness {
        enum Ev {
            OTHER, UNSIGNED, SIGNED
        };
        int ev;
    };
#endif

/*!
 * \brief information about a data type that can be accessed at compile time
 * \tparam DTypeImpl either an implementation class (e.g. dtype::Int32), or a
 *      plain c type (e.g. int or dt_int32)
 */
template <class DTypeImpl>
struct DTypeTrait;

// This can be specialized to define custom param structures for each
// parameterized DType, it should implement `std::size_t hash()` and
// `bool operator==(rhs).`
template <typename Type>
struct DTypeParamImpl;

template <typename DType>
using DTypeParam = DTypeParamImpl<typename DTypeTrait<DType>::ctype>;

/*!
 * \brief Information about a data type that can be accessed at runtime
 */
class DType {
    private:
        MEGDNN_NORETURN void on_request_lowbit_size() const;
    // HACK: This is required in ParameterizedDType::downcast_from
    public:
        MEGDNN_NORETURN void on_assert_is_failed(const char *rname) const;
    protected:
        struct Trait {
            const char *const name;
            const uint16_t size_log;    //!< log2 of sizeof(dt) for non-lowbit
            const uint16_t low_bit; //!< 0 for non-lowbit; otherwise num bits
            DTypeEnum enumv;
            DTypeCategory category;
            DTypeSignedness signedness;
            const bool has_param;
        };
        Trait *m_trait;

        explicit DType(Trait *t):
            m_trait(t)
        {}

    public:
        DType():
            m_trait(nullptr)
        {}

        bool valid() const {
            return m_trait != nullptr;
        }

        /*!
         * \brief name of this data type
         */
        const char *name() const {
            return m_trait ? m_trait->name : "invalid";
        }

        /*!
         * \brief size of elem_num this data type, if fraction form return ceil
         */
        size_t size(size_t elem_num) const {
            if (m_trait->low_bit != 0)
                return static_cast<size_t>( (m_trait->low_bit*elem_num + 7)/8 );
            return elem_num << m_trait->size_log;
        }

        /*!
         * \brief max number of elements within representation
         *
         * The total size of the tensor (in bytes) should not exceed size_t range.
         */
        size_t max_elements() const {
            if (m_trait->low_bit != 0)
                return std::numeric_limits<size_t>::max();

            return std::numeric_limits<size_t>::max() >> m_trait->size_log;
        }

        bool is_low_bit() const {
            return m_trait->low_bit != 0;
        }

        /*!
         * \brief size of this data type, in bytes
         */
        size_t size() const {
            if (m_trait->low_bit == 0)
                return 1 << m_trait->size_log;
            on_request_lowbit_size();
        }

        //! size() in log2
        size_t size_log() const {
            if (m_trait->low_bit == 0)
                return m_trait->size_log;
            on_request_lowbit_size();
        }

        //! assert this dtype is given type; throw exception on failure
        void assert_is(const DType &rhs) const {
            if (m_trait != rhs.m_trait)
                on_assert_is_failed(rhs.name());
        }

        template<typename T>
        inline void assert_is_ctype() const;

        template<typename T>
        inline void assert_is_compatible_ctype() const;

        //! get corresponding enum value for this dtype
        DTypeEnum enumv() const {
            return m_trait->enumv;
        }

        //! get category of this data type
        DTypeCategory category() const {
            return m_trait->category;
        }

        //! get signedness of this data type
        DTypeSignedness signedness() const {
            return m_trait->signedness;
        }

        bool has_param() const {
            return m_trait->has_param;
        }

        bool operator == (const DType &rhs) const {
            return m_trait == rhs.m_trait;
        }

        bool operator != (const DType &rhs) const {
            return m_trait != rhs.m_trait;
        }

        //! get dtype object from enum
        static DType from_enum(DTypeEnum ev);

        //! get a handle of the dtype that could be used for equivalence check
        const void* handle() const {
            return m_trait;
        }

        template <typename T>
        T as() const {
            return T::downcast_from(*this);
        }

        template <typename T>
        const DTypeParam<T>& param() const {
            return as<typename DTypeTrait<T>::dtype>().param();
        }
};

#ifdef MEGDNN_CC_HOST

/*!
 * \brief class template for parameterized DTypes
 *
 * You should not change this template in order to add new parameterized
 * DType, instead you should add new entry to
 * MEGDNN_FOREACH_PARAMETERIZED_DTYPE_OTHERS, follow the compile error, then add
 * new specialization of DTypeParam at the end of this file.
 */
template <DTypeEnum type_enum>
class ParameterizedDType MEGDNN_FINAL : public DType {
    using SelfType = ParameterizedDType<type_enum>;

    struct Trait : DType::Trait {
        DTypeParam<SelfType> param;

        Trait(const DType::Trait& static_trait,
              const DTypeParam<SelfType>& param)
                : DType::Trait(static_trait), param(param) {}
    };

    // static part of the trait
    static DType::Trait sm_trait;

    static Trait* make_from_param(const DTypeParam<SelfType>& param);
    explicit ParameterizedDType(DType dtype) : DType(dtype) {}

public:
    template <class... Args>
    explicit ParameterizedDType(Args&&... args)
            : DType(make_from_param({std::forward<Args>(args)...})) {}

/**
 * static member \c sm_trait is been used, the compiler wil trigger
 * warnings if it hasn't an explicit instantiation declaration with include dir
 * using \c -I; while build by bazel, include dir is traited as system headers,
 * using \c -isystem, and the warnings is supressed.
 *
 * Here we just supressed the warning, as it will explicit instantiation in
 * \c dtype.cpp.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wundefined-var-template"
    static SelfType downcast_from(DType dtype) {
        if (dtype.enumv() != type_enum) {
            dtype.on_assert_is_failed(sm_trait.name);
        }
        return ParameterizedDType(dtype);
    }
#pragma GCC diagnostic pop

    const DTypeParam<SelfType>& param() {
        return static_cast<Trait*>(m_trait)->param;
    }
};

#endif  // MEGDNN_CC_HOST

//! dtype implementation classes
namespace dtype {

#define IMPL(_name) \
    class _name MEGDNN_FINAL: public DType { \
        static Trait sm_trait; \
        public: \
            _name(): DType(&sm_trait) {} \
    };

MEGDNN_FOREACH_DTYPE_NAME(IMPL)
#undef IMPL

#ifdef MEGDNN_CC_HOST
#define cb(_name) using _name = ParameterizedDType<DTypeEnum::_name>;
#else
#define cb(_name) \
    class _name MEGDNN_FINAL : public DType {};
#endif
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb

//! log function used in DTypeTrait
template<uint16_t n> struct log {
    static MEGDNN_CONSTEXPR size_t value = log<(n>>1)>::value + 1;
#if MEGDNN_CC_HOST
    MEGDNN_STATIC_ASSERT( (n&(n-1)) == 0, "only full power number can have log");
#endif
};
template<> struct log<1> {static MEGDNN_CONSTEXPR size_t value = 0;};

} // namespace dtype

// begin define DTypeTrait impls {

#if MEGDNN_CC_HOST
#define MEGDNN_DEF_DT_BASIC_FIELDS(_name, _ctype, _cat, _sign, _bits, \
                                   _has_param) \
    static MEGDNN_CONSTEXPR const char *name = #_name; \
    using ctype = _ctype; \
    using dtype = ::megdnn::dtype::_name; \
    static MEGDNN_CONSTEXPR DTypeCategory category = DTypeCategory::_cat; \
    static MEGDNN_CONSTEXPR DTypeSignedness \
        signedness = DTypeSignedness::_sign; \
    static MEGDNN_CONSTEXPR uint16_t size_log = \
        ::megdnn::dtype::log<sizeof(ctype)>::value; \
    static MEGDNN_CONSTEXPR DTypeEnum enumv = DTypeEnum::_name;\
    static MEGDNN_CONSTEXPR uint16_t low_bit = _bits;\
    static MEGDNN_CONSTEXPR bool has_param = _has_param
#else
#define MEGDNN_DEF_DT_BASIC_FIELDS(_name, _ctype, _cat, _sign, _bits, \
                                   _has_param) \
    typedef _ctype ctype; \
    typedef ::megdnn::dtype::_name dtype; \
    static const uint16_t size_log = \
        ::megdnn::dtype::log<sizeof(ctype)>::value; \
    static MEGDNN_CONSTEXPR int enumv = DTypeEnum::_name;\
    static MEGDNN_CONSTEXPR uint16_t low_bit = _bits
#endif // MEGDNN_CC_HOST

#define MEGDNN_DEF_DT(_name, _ctype, _cat, _sign, _minval, _maxval) \
    template <> \
    struct DTypeTrait <dtype::_name> { \
        MEGDNN_DEF_DT_BASIC_FIELDS(_name, _ctype, _cat, _sign, 0, false); \
        MEGDNN_HOST MEGDNN_DEVICE static ctype min() { \
            return _minval; \
        } \
        MEGDNN_HOST MEGDNN_DEVICE static ctype max() { \
            return _maxval; \
        } \
    }

MEGDNN_DEF_DT(Float32, dt_float32, FLOAT, SIGNED, -FLT_MAX, FLT_MAX);
MEGDNN_DEF_DT(Int32, dt_int32, INT, SIGNED, INT32_MIN, INT32_MAX);
MEGDNN_DEF_DT(Int16, dt_int16, INT, SIGNED, INT16_MIN, INT16_MAX);
MEGDNN_DEF_DT(Int8, dt_int8, INT, SIGNED, INT8_MIN, INT8_MAX);
MEGDNN_DEF_DT(Uint8, dt_uint8, INT, UNSIGNED, 0, UINT8_MAX);
MEGDNN_INC_FLOAT16(MEGDNN_DEF_DT(Float16, dt_float16, FLOAT, SIGNED,
            std::numeric_limits<dt_float16>::lowest(),
            std::numeric_limits<dt_float16>::max()));

template <>
struct DTypeTrait<dtype::Byte> {
    MEGDNN_DEF_DT_BASIC_FIELDS(Byte, dt_byte, OTHER, OTHER, 0, false);
};

#define MEGDNN_DEF_FRACTION_DT(_name, b)\
    template <> \
    struct DTypeTrait<dtype::_name##b> {\
        MEGDNN_DEF_DT_BASIC_FIELDS(_name##b, dt_##_name##b, LOWBIT, OTHER, b, \
                                   false); \
    };
MEGDNN_FOREACH_LOWBIT_DTYPE(MEGDNN_DEF_FRACTION_DT)
#undef MEGDNN_DEF_FRACTION_DT

#define MEGDNN_DEF_PARAMETERIZED_DT(_name, _ctype, _itype, _cat, _sign,      \
                                    _minval, _maxval, _bits)                 \
    template <>                                                              \
    struct DTypeTrait<dtype::_name> {                                        \
        MEGDNN_DEF_DT_BASIC_FIELDS(_name, _ctype, _cat, _sign, _bits, true); \
        MEGDNN_HOST MEGDNN_DEVICE static _itype min() {                      \
            return static_cast<_itype>(_minval);                             \
        }                                                                    \
        MEGDNN_HOST MEGDNN_DEVICE static _itype max() {                      \
            return static_cast<_itype>(_maxval);                             \
        }                                                                    \
    };

MEGDNN_DEF_PARAMETERIZED_DT(Quantized4Asymm, dt_quint4, uint8_t, QUANTIZED,
                            SIGNED, 0, 15, 4);
MEGDNN_DEF_PARAMETERIZED_DT(QuantizedS4, dt_qint4, int8_t, QUANTIZED,
                            SIGNED, -8, 7, 4);
MEGDNN_DEF_PARAMETERIZED_DT(Quantized8Asymm, dt_quint8, dt_quint8, QUANTIZED,
                            SIGNED, 0, 255, 0);
MEGDNN_DEF_PARAMETERIZED_DT(QuantizedS8, dt_qint8, dt_qint8, QUANTIZED, SIGNED,
                            INT8_MIN, INT8_MAX, 0);
MEGDNN_DEF_PARAMETERIZED_DT(QuantizedS16, dt_qint16, dt_qint16, QUANTIZED,
                            SIGNED, INT16_MIN, INT16_MAX, 0);
MEGDNN_DEF_PARAMETERIZED_DT(QuantizedS32, dt_qint32, dt_qint32, QUANTIZED,
                            SIGNED, INT32_MIN, INT32_MAX, 0);
#undef MEGDNN_DEF_PARAMETERIZED_DT

#undef MEGDNN_DEF_DT
#undef MEGDNN_DEF_DT_BASIC_FIELDS
// end define DTypeTrait impls }


// alias DTypeTrait for ctypes
#define IMPL(_obj) \
template <> \
struct DTypeTrait<DTypeTrait<dtype::_obj>::ctype>: \
public DTypeTrait<dtype::_obj> { };

MEGDNN_FOREACH_DTYPE_NAME(IMPL)
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(IMPL)
#undef IMPL


template<typename T>
inline void DType::assert_is_ctype() const {
    return assert_is(typename DTypeTrait<T>::dtype());
}

#ifdef MEGDNN_CC_HOST

#define INST(_dt)                                                       \
    template <>                                                         \
    inline void DType::assert_is_ctype<DTypeTrait<dtype::_dt>::ctype>() \
            const {                                                     \
        if (enumv() != DTypeTrait<dtype::_dt>::enumv) {                 \
            on_assert_is_failed(DTypeTrait<dtype::_dt>::name);          \
        }                                                               \
    }
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(INST)
#undef INST


template <typename T>
inline void DType::assert_is_compatible_ctype() const {
    if (enumv() != DTypeTrait<T>::enumv) {
        on_assert_is_failed(DTypeTrait<T>::name);
    }
}

#define INST(_dt, _dtype)                                                      \
    template <>                                                                \
    inline void                                                                \
    DType::assert_is_compatible_ctype<DTypeTrait<dtype::_dt>::ctype>() const { \
        if (enumv() != DTypeTrait<dtype::_dt>::enumv &&                        \
            enumv() != DTypeTrait<dtype::_dtype>::enumv) {                     \
            on_assert_is_failed(DTypeTrait<dtype::_dt>::name);                 \
        }                                                                      \
    }

INST(Int8, QuantizedS8)
INST(Uint8, Quantized8Asymm)
INST(Int16, QuantizedS16)
INST(Int32, QuantizedS32)
#undef INST

#else

#define INST(_dt)                                                       \
    template <>                                                         \
    inline void DType::assert_is_ctype<DTypeTrait<dtype::_dt>::ctype>() \
            const {                                                     \
        if (enumv().ev != DTypeTrait<dtype::_dt>::enumv) {              \
            on_assert_is_failed(dtype::_dt().name());                   \
        }                                                               \
    }
MEGDNN_FOREACH_PARAMETERIZED_DTYPE(INST)
#undef INST

#endif  // MEGDNN_CC_HOST


// begin Specialization of DTypeParamImpl for each parameterzied DType {
template <>
struct DTypeParamImpl<dt_quint8> {
    float scale;
    uint8_t zero_point;

    DTypeParamImpl<dt_quint8>() = default;
    DTypeParamImpl<dt_quint8>(float scale, uint8_t zero_point);

#ifdef MEGDNN_CC_HOST
    std::size_t hash() const;
#endif
    bool operator==(const DTypeParam<dt_quint8>& rhs) const;

    MEGDNN_DEVICE dt_quint8 quantize(float in) const {
        float v = in / scale;
        v = roundf(v);
        v = v + zero_point;
        v = fmin(fmax(0.f, v), 255.f);
        return static_cast<dt_quint8>(v);
    }
    MEGDNN_DEVICE float dequantize(dt_quint8 in) const {
        return (in.as_uint8() - zero_point) * scale;
    }
};

template <>
struct DTypeParamImpl<dt_qint8> {
    float scale;

    DTypeParamImpl<dt_qint8>() = default;
    DTypeParamImpl<dt_qint8>(float scale);
#ifdef MEGDNN_CC_HOST
    std::size_t hash() const;
#endif
    bool operator==(const DTypeParam<dt_qint8>& rhs) const;
    MEGDNN_DEVICE dt_qint8 quantize(float in) const {
        float v = in / scale;
        //! roundf(nan) -> nan
        v = roundf(v);
        //! \warning As fmax(nan, a) = a, this should match the process
        //! in function saturate(), otherwise may cause precision error.
        v = fmin(fmax(-128.f, v), 127.f);
        return static_cast<dt_qint8>(v);
    }
    MEGDNN_DEVICE float dequantize(dt_qint8 in) const {
        return in.as_int8() * scale;
    }
};

template <>
struct DTypeParamImpl<dt_qint16> {
    float scale;

    DTypeParamImpl<dt_qint16>() = default;
    DTypeParamImpl<dt_qint16>(float scale);
#ifdef MEGDNN_CC_HOST
    std::size_t hash() const;
#endif  // MEGDNN_CC_HOST
    bool operator==(const DTypeParam<dt_qint16>& rhs) const;
    MEGDNN_DEVICE dt_qint16 quantize(float in) const {
        float v = in / scale;
        v = roundf(v);
        //! \warning As fmax(nan, a) = a, this should match the process
        //! in function saturate(), otherwise may cause precision error.
        v = fmin(fmax(-32768.f, v), 32767.f);
        return static_cast<dt_qint16>(v);
    }
    MEGDNN_DEVICE float dequantize(dt_qint16 in) const {
        return in.as_int16() * scale;
    }
};

template <>
struct DTypeParamImpl<dt_qint32> {
    float scale;

    DTypeParamImpl<dt_qint32>() = default;
    DTypeParamImpl<dt_qint32>(float scale);
#ifdef MEGDNN_CC_HOST
    std::size_t hash() const;
#endif  // MEGDNN_CC_HOST
    bool operator==(const DTypeParam<dt_qint32>& rhs) const;
    MEGDNN_DEVICE dt_qint32 quantize(float in) const {
        float v = in / scale;
        v = roundf(v);
        /*! \note: the maximal signed integer that can be correctly represented
         * as a single precision floating point number is 2147483520
         */
        v = fmin(fmax(-2147483648.f, v), 2147483520.f);
        return static_cast<dt_qint32>(v);
    }
    MEGDNN_DEVICE float dequantize(dt_qint32 in) const {
        return in.as_int32() * scale;
    }
};

template <>
struct DTypeParamImpl<dt_quint4> {
    float scale;
    uint8_t zero_point;

    DTypeParamImpl<dt_quint4>() = default;
    DTypeParamImpl<dt_quint4>(float scale, uint8_t zero_point);
#ifdef MEGDNN_CC_HOST
    std::size_t hash() const;
#endif
    bool operator==(const DTypeParam<dt_quint4>& rhs) const;
    MEGDNN_DEVICE dt_quint4 quantize(float in) const {
        float v = in / scale;
        v = roundf(v);
        v = v + zero_point;
        v = fmin(fmax(0.f, v), 15.f);
        return static_cast<dt_quint4>(v);
    }
    MEGDNN_DEVICE float dequantize(uint8_t in) const {
        return (in - zero_point) * scale;
    }
    MEGDNN_DEVICE float dequantize(dt_quint4 in) const {
        return (in.as_uint8() - zero_point) * scale;
    }
};

template <>
struct DTypeParamImpl<dt_qint4> {
    float scale;

    DTypeParamImpl<dt_qint4>() = default;
    DTypeParamImpl<dt_qint4>(float scale);
#ifdef MEGDNN_CC_HOST
    std::size_t hash() const;
#endif
    bool operator==(const DTypeParam<dt_qint4>& rhs) const;
    MEGDNN_DEVICE dt_qint4 quantize(float in) const {
        float v = in / scale;
        v = roundf(v);
        v = fmin(fmax(-8.f, v), 7.f);
        return static_cast<dt_qint4>(v);
    }
    MEGDNN_DEVICE float dequantize(int8_t in) const {
        return in * scale;
    }
    MEGDNN_DEVICE float dequantize(dt_qint4 in) const {
        return in.as_int8() * scale;
    }
};

// end Specialization of DTypeParamImpl for each parameterzied DType }

} // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
