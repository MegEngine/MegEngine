/**
 * \file src/core/include/megbrain/utils/hashable.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/hash.h"
#include "megbrain/utils/metahelper.h"

#include "megbrain/common.h"

#include <cstddef>
#include <cstring>

#include <type_traits>

namespace mgb {

/*!
 * \brief Hashable object, which supports hashing and equality test
 */
class Hashable: public DynTypeObj {
    public:
        /*!
         * \brief calculate the hash value
         */
        virtual size_t hash() const = 0;

        /*!
         * \brief Equality test
         */
        bool is_same(const Hashable &rhs) const {
            return dyn_typeinfo() == rhs.dyn_typeinfo() && is_same_st(rhs);
        }

    protected:
        ~Hashable() = default;

        /*!
         * \brief check whether two objects is the same; rhs guaranteed to be
         *      the same type of this
         */
        virtual bool is_same_st(const Hashable &rhs) const = 0;
};

/*!
 * \brief Hashable with virtual destructor
 */
class HashableVD: public Hashable {
    public:
        virtual ~HashableVD() = default;
};

/*!
 * \brief Fixed-size container for small hashable object, to avoid excessive
 *      heap allocation
 */
class alignas(std::max_align_t) HashableContainer: public NonCopyableObj {
    static constexpr size_t MAX_SIZE = 28;
    uint8_t m_raw_data[MAX_SIZE];
    int32_t m_base_offset = -1; // -1 for uninitialized

    HashableVD& obj() {
        return *aliased_ptr<HashableVD>(m_raw_data + m_base_offset);
    }

    const HashableVD& obj() const {
        return const_cast<HashableContainer*>(this)->obj();
    }

    public:
        HashableContainer() = default;

        HashableContainer(HashableContainer &&rhs) noexcept {
            this->operator=(std::move(rhs));
        }

        ~HashableContainer() noexcept {
            release();
        }

        HashableContainer& operator = (HashableContainer &&rhs) noexcept {
            if (this == &rhs)
                return *this;
            release();
            memcpy(m_raw_data, rhs.m_raw_data, MAX_SIZE);
            m_base_offset = rhs.m_base_offset;
            rhs.m_base_offset = -1;
            return *this;
        }

        /*!
         * \brief factory method; see init<T> for more details
         */
        template<typename T, typename ...Args>
        static HashableContainer create(Args &&...args) {
            HashableContainer v;
            v.init<T, Args...>(std::forward<Args>(args)...);
            return v;
        }

        void release() noexcept {
            if (m_base_offset >= 0) {
                obj().~HashableVD();
                m_base_offset = -1;
            } else
                mgb_assert(m_base_offset == -1);
        }

        /*!
         * \brief initialize using placement new; note that T should have no
         * pointer reference to itself, so it could be copied by memcpy
         */
        template<typename T, typename ...Args>
        void init(Args &&...args) {
            static_assert(std::is_base_of<HashableVD, T>::value,
                    "must be HashableVD objects");
            static_assert(alignof(T) <= alignof(HashableContainer) &&
                    sizeof(T) <= MAX_SIZE, "could not be put into container");
            release();
            mgb_assert(reinterpret_cast<ptrdiff_t>(m_raw_data)
                    % alignof(T) == 0, "could not be aligned");
            T *ptr = new (m_raw_data) T(std::forward<Args>(args)...);
            m_base_offset =
                reinterpret_cast<uint8_t*>(static_cast<HashableVD*>(ptr)) -
                m_raw_data;
        }

        size_t hash() const {
            return obj().hash();
        }

        bool is_same(const Hashable &rhs) const {
            return obj().is_same(rhs);
        }

        bool is_same(const HashableContainer &rhs) const {
            return obj().is_same(rhs.obj());
        }
};

/*!
 * \brief hash of scalar types
 */
template<typename T>
class ScalarHash final: public HashableVD {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    union U {
        T t;
        size_t v;
        U() {}
    };
    U m_val;

    static_assert(std::is_scalar<T>::value &&
            sizeof(T) <= sizeof(size_t) && !(alignof(size_t) % alignof(T)),
            "bad type");

    bool is_same_st(const Hashable &rhs) const override {
        return m_val.v == static_cast<const ScalarHash&>(rhs).m_val.v;
    }

    public:
        ScalarHash(T val)
        {
            m_val.v = 0;    // fill padding bytes
            m_val.t = val;
        }

        size_t hash() const override {
            return m_val.v;
        }
};
#undef _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL template<typename T>
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ScalarHash<T>);
#undef _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL

/*!
 * \brief Hash for data of non-scalar POD types
 */
template<typename T>
class PODHash final: public HashableVD {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    static_assert(is_location_invariant<T>::value,
            "key must be location invariant");

    const T *m_ptr;
    size_t m_nr_elem;

    bool is_same_st(const Hashable &rhs) const override {
        auto p = static_cast<const PODHash*>(&rhs);
        return !memcmp(m_ptr, p->m_ptr, m_nr_elem * sizeof(T));
    }

    public:
        /*!
         * \brief note that the object would not be copied, so its lifespan must
         * contain lifespan of this PODHash object
         */
        PODHash(const T *ptr, size_t nr_elem = 1):
            m_ptr(ptr), m_nr_elem(nr_elem)
        {
        }

        static size_t perform(const T *ptr, size_t nr_elem) {
            XXHash xh;
            xh.reset();
            xh.update(ptr, nr_elem * sizeof(T));
            return xh.digest();
        }

        size_t hash() const override {
            return perform(m_ptr, m_nr_elem);
        }

};
#undef _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL template<typename T>
MGB_DYN_TYPE_OBJ_FINAL_IMPL(PODHash<T>);
#undef _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL

/*!
 * \brief wraps around a raw pointer to Hashable object
 */
class HashableObjPtrWrapper final: public HashableVD {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;

    const Hashable *m_ptr;

    bool is_same_st(const Hashable &rhs) const override {
        return m_ptr->is_same(
                *static_cast<const HashableObjPtrWrapper&>(rhs).m_ptr);
    }

    public:
        HashableObjPtrWrapper(const Hashable *ptr):
            m_ptr(ptr)
        {}

        size_t hash() const override {
            return m_ptr->hash();
        }
};

template<>
struct HashTrait<HashableContainer> {
    static size_t eval(const HashableContainer &val) {
        return val.hash();
    }
};

}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
