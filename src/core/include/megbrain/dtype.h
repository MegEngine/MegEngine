/**
 * \file src/core/include/megbrain/dtype.h
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
#include "megbrain/common.h"

namespace mgb {

using ::megdnn::dt_byte;
MEGDNN_INC_FLOAT16(using ::megdnn::dt_float16;)
MEGDNN_INC_FLOAT16(using ::megdnn::dt_bfloat16;)
using ::megdnn::dt_float32;
using ::megdnn::dt_int8;
using ::megdnn::dt_uint8;
using ::megdnn::dt_int16;
using ::megdnn::dt_int32;
using ::megdnn::dt_quint8;
using ::megdnn::dt_qint8;
using ::megdnn::dt_qint32;
using ::megdnn::dt_bool;
using ::megdnn::DType;
using ::megdnn::DTypeEnum;
using ::megdnn::DTypeTrait;
using ::megdnn::DTypeCategory;
using dt_max_float = dt_float32;

namespace dtype = ::megdnn::dtype;

/*!
 * \brief static cast from any dtype to given ctype
 *
 * Batched interface to improve speed (so dtype dispatch would not take much
 * time)
 *
 * \param nr_elem number of elements to write in *dest*
 */
template<typename T>
void static_cast_dtype(T* dest,
        DType src_type, const void *storage, size_t nr_elem = 1);

/*!
 * \brief similar to static_cast_dtype, but throws exception if precision loss
 *
 * Note: no exception would be thrown when casting an out-of-range value.
 */
template<typename T>
void static_cast_dtype_safe(T *dest,
        DType src_type, const void *storage, size_t nr_elem = 1);

/*!
 * \brief a template to test whether a ctype is supported; for supported ctype,
 *      it would have a member type named *type* defined as *T*
 */
template<typename ctype, typename T = void>
struct ctype_enable_if;

#define cb(_dt)  \
template<typename T> \
struct ctype_enable_if<DTypeTrait<_dt>::ctype, T> { \
    using type = T; \
};
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb


/*!
 * \brief a scalar value with associated dtype
 */
class DTypeScalar {
    public:
        using max_ctype = size_t;

        DTypeScalar() {
            init_storage();
        }

        //! DTypeScalar with given dtype and zero initialization
        DTypeScalar(DType dtype):
            m_dtype{dtype}
        {
            init_storage();
        }

        template<typename ctype,
            typename = typename ctype_enable_if<ctype>::type>
        DTypeScalar(ctype val)
        {
            set(val);
        }

        static DTypeScalar make_from_raw(DType dtype, const void *storage) {
            return DTypeScalar{}.set_raw(dtype, storage);
        }

        /*!
         * \brief set to given value by raw storage
         */
        DTypeScalar& set_raw(DType dtype, const void *storage);

        /*!
         * \brief set to given value, with dtype corresponding to ctype
         */
        template<typename ctype>
        typename ctype_enable_if<ctype>::type set(ctype val) {
            static_assert(sizeof(ctype) <= sizeof(m_storage),
                    "bad converted ctype");
            init_storage();
            m_dtype = typename DTypeTrait<ctype>::dtype();
            visit<ctype>() = val;
        }

        /*!
         * \brief set to given value, but use current dtype and cast value to it
         */
        template<typename ctype>
        typename ctype_enable_if<ctype>::type set_retain_dtype(ctype val);

        /*!
         * \brief get underlying value, which must be exactly given type
         * \tparam T expected scalar type
         */
        template<typename T>
        T& get() {
            m_dtype.assert_is_ctype<T>();
            return visit<T>();
        }

        template<typename T>
        T get() const {
            return const_cast<DTypeScalar*>(this)->get<T>();
        }

        /*!
         * \brief get underlying value and static_cast to given type
         */
        template<typename T>
        T get_cast() const {
            T v;
            static_cast_dtype(&v, m_dtype, storage());
            return v;
        }

        DType dtype() const {
            return m_dtype;
        }

        //! get underlying raw storage
        const void* storage() const { return &m_storage; }

        bool operator == (const DTypeScalar &rhs) const {
            return m_dtype == rhs.m_dtype &&
                visit<max_ctype>() == rhs.visit<max_ctype>();
        }

        bool operator!=(const DTypeScalar& rhs) const {
            return !this->operator==(rhs);
        }

    private:
        std::aligned_storage_t<sizeof(max_ctype), alignof(max_ctype)> m_storage;
        DType m_dtype;

        template <typename T>
        T& visit() {
            return reinterpret_cast<T&>(m_storage);
        }

        template <typename T>
        T visit() const {
            return reinterpret_cast<const T&>(m_storage);
        }

        void init_storage() { visit<max_ctype>() = 0; }
};
static_assert(sizeof(DTypeScalar) == sizeof(DTypeScalar::max_ctype) +
        sizeof(DType), "bad DTypeScalar size");

DType dtype_promotion(DType t0, DType t1);

/*!
 * \brief copy from byte representation to compact representation for lowbit
 *      types
 */
void lowbit_memcpy_byte2compact(
        DType dtype, void *dest, const void *src, size_t n);

/*!
 * \brief copy from compact representation to byte representation for lowbit
 *      types
 */
void lowbit_memcpy_compact2byte(
        DType dtype, void *dest, const void *src, size_t n);


} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

