/**
 * \file dnn/include/megdnn/tensor_iter.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"
#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {

template <typename T>
class TypeRef {
public:
    using dtype = T&;
    static T& get(T* _ptr, size_t _offset) {
        T& ret = _ptr[_offset];
        return ret;
    }
};

template <>
class TypeRef<dt_quint4> {
private:
    uint8_t* ptr = nullptr;
    size_t offset = 0;

public:
    using dtype = TypeRef<dt_quint4>;
    dt_quint4 val = dt_quint4(0);

    TypeRef(dt_quint4* _ptr, size_t _offset);

    void operator=(const uint8_t _);
    void operator=(const dt_quint4& _) { *this = _.as_uint8(); }
    void operator=(const TypeRef<dt_quint4>& _) { *this = _.val.as_uint8(); }
    operator dt_quint4() const { return val; }
    operator uint8_t() const { return val.as_uint8(); }

    static TypeRef<dt_quint4> get(dt_quint4* _ptr, size_t _offset) {
        return TypeRef<dt_quint4>(_ptr, _offset);
    }
};

template <>
class TypeRef<dt_qint4> {
private:
    int8_t* ptr = nullptr;
    size_t offset = 0;

public:
    using dtype = TypeRef<dt_qint4>;
    dt_qint4 val = dt_qint4(0);
    TypeRef(dt_qint4* _ptr, size_t _offset);

    void operator=(const int8_t _);
    void operator=(const dt_qint4& _) { *this = _.as_int8(); }
    void operator=(const TypeRef<dt_qint4>& _) { *this = _.val.as_int8(); }
    operator dt_qint4() const { return val; }
    operator int8_t() const { return val.as_int8(); }

    static TypeRef<dt_qint4> get(dt_qint4* _ptr, size_t _offset) {
        return TypeRef<dt_qint4>(_ptr, _offset);
    }
};

/*!
 * \brief helper for iterating on a tensor with arbitrary layout
 * \tparam ctype tensor element plain data type
 * \tparam valonly whether only value is needed (so logical index does not need
 *      to be maintained)
 */
template <typename ctype, bool valonly>
class TensorIter {
    TensorND m_tensor;

public:
    class Iter {
        MEGDNN_NORETURN void on_access_idx_valonly_true() const;

        ctype* m_ptr = nullptr;

        TensorLayout m_layout;

        ptrdiff_t m_axis_reset_stride[TensorShape::MAX_NDIM],
                m_offset = 0;  //!< physical offset in buffer

        //! offset in each axis
        size_t m_axis_offset[TensorShape::MAX_NDIM],
                m_logical_offset = 0,  //!< contiguous logical offset
                m_tot_nr_elems = 0;    //!< tot elems (max logical offset)

    public:
        Iter() {
            memset(m_axis_reset_stride, 0, sizeof(m_axis_reset_stride));
            memset(m_axis_offset, 0, sizeof(m_axis_offset));
        }

        /*!
         * \brief create an iterator
         */
        static Iter make(ctype* ptr, const TensorLayout& layout, size_t offset);

        static Iter make(TensorND& t, size_t offset) {
            return make(t.ptr<ctype>(), t.layout, offset);
        }

        //! access element without boundary check
        typename TypeRef<ctype>::dtype operator*() {
            return TypeRef<ctype>::get(m_ptr, m_offset);
        };

        Iter& operator++() {
            if ((++m_logical_offset) == m_tot_nr_elems)
                return *this;
            auto mem_offset = m_offset;
            for (int axis = m_layout.ndim - 1;; axis--) {
                size_t& ax_offset = ++m_axis_offset[axis];
                if (ax_offset < m_layout.shape[axis]) {
                    mem_offset += m_layout.stride[axis];
                    break;
                } else {
                    ax_offset = 0;
                    mem_offset -= m_axis_reset_stride[axis];
                }
            }
            m_offset = mem_offset;
            return *this;
        }

        //! whether current value valid
        bool valid() const { return m_logical_offset < m_tot_nr_elems; }

        //! whether current pos is at end of buffer
        bool at_end() const { return m_logical_offset == m_tot_nr_elems; }

        //! get logical index; valonly must be false
        const size_t* idx() const {
            if (valonly)
                on_access_idx_valonly_true();
            return m_axis_offset;
        }

        /*!
         * \brief memory address offset, measured in number of elements
         */
        size_t offset() const { return m_offset; }

        /*!
         * \brief number of elements from first element
         */
        size_t logical_offset() const { return m_logical_offset; }

        bool operator!=(const Iter& rhs) const {
            return m_logical_offset != rhs.m_logical_offset;
        }
    };
    TensorIter() = default;

    TensorIter(const TensorND& tensor) : m_tensor(tensor) {}

    Iter begin() const {
        return Iter::make(const_cast<TensorND&>(m_tensor), 0);
    }

    Iter end() const {
        return Iter::make(const_cast<TensorND&>(m_tensor),
                          m_tensor.layout.total_nr_elems());
    }
};
/*!
 * \brief iterate over elements of a tensor; only access tensor value
 */
template <typename ctype>
TensorIter<ctype, true> tensor_iter_valonly(const TensorND& t) {
    return {t};
}

/*!
 * \brief iterate over elements of a tensor, retaining logical index
 */
template <typename ctype>
TensorIter<ctype, false> tensor_iter(const TensorND& t) {
    return {t};
}

}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
