/**
 * \file src/core/include/megbrain/utils/metahelper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/exception.h"
#include "megbrain/utils/hash.h"
#include "megbrain/utils/thin/function.h"
#include "megbrain/utils/thin/hash_table.h"

#include <algorithm>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace mgb {

/*!
 * \brief an object to represent a type
 *
 * MegBrain has a lightweight RTTI system. Each type is represented by the
 * address of a Typeinfo object, which is stored in the .bss segment.
 *
 * MGB_TYPEINFO_OBJ_DECL should be placed into the definition of classes that
 * need compile-time type support.
 *
 * For classes that need RTTI, they should be derived from DynTypeObj and
 * include MGB_DYN_TYPE_OBJ_FINAL_DECL in the definition.
 */
struct Typeinfo {
    //! name of the corresponding type; nullptr if MGB_VERBOSE_TYPEINFO_NAME==0
    const char* const name;

    /*!
     * \brief whether this is the type of given object
     * \tparam T a class with static typeinfo() method
     */
    template<typename T>
    bool is() const {
        return T::typeinfo() == this;
    }
};

/*!
 * \brief base class to emulate RTTI without compiler support
 */
class DynTypeObj {
    public:
        virtual Typeinfo* dyn_typeinfo() const = 0;

        //! cast this to a final object (no type check is performed)
        template<class T>
        T& cast_final() {
            return *static_cast<T*>(this);
        }

        template<class T>
        const T& cast_final() const {
            return const_cast<DynTypeObj*>(this)->cast_final<T>();
        }

        //! cast this to a final object with type check
        template<class T>
        T& cast_final_safe() {
            mgb_assert(T::typeinfo() == dyn_typeinfo(),
                    "can not convert type %s to %s",
                    dyn_typeinfo()->name, T::typeinfo()->name);
            return cast_final<T>();
        }

        template<class T>
        const T& cast_final_safe() const {
            return const_cast<DynTypeObj*>(this)->cast_final_safe<T>();
        }

        //! cast this to a final object if type matches; return nullptr if not
        template <class T>
        T* try_cast_final() {
            return T::typeinfo() == dyn_typeinfo() ? static_cast<T*>(this)
                                                   : nullptr;
        }

        template <class T>
        const T* try_cast_final() const {
            return const_cast<DynTypeObj*>(this)->try_cast_final<T>();
        }

        //! check whether this is same to given type
        template<class T>
        bool same_type() const {
            return dyn_typeinfo() == T::typeinfo();
        }

    protected:
        ~DynTypeObj() = default;
};

//! define to template param so MGB_DYN_TYPE_OBJ_FINAL_IMPL for templates can
//! work
#define _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL

//! put in the declaration of a class that only needs static typeinfo()
#define MGB_TYPEINFO_OBJ_DECL \
    public: \
        static inline ::mgb::Typeinfo* typeinfo() { \
            return &sm_typeinfo; \
        } \
    private: \
        static ::mgb::Typeinfo sm_typeinfo \


#if MGB_VERBOSE_TYPEINFO_NAME
//! get class name from class object
#define _MGB_TYPEINFO_CLASS_NAME(_cls) #_cls
#else
#define _MGB_TYPEINFO_CLASS_NAME(_cls) nullptr
#endif

//! put in the impl file of a class that needs static typeinfo()
#define MGB_TYPEINFO_OBJ_IMPL(_cls) \
    _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL \
    ::mgb::Typeinfo _cls::sm_typeinfo{_MGB_TYPEINFO_CLASS_NAME(_cls)}


//! put in the declaration of a final class inherited from DynTypeObj
#define MGB_DYN_TYPE_OBJ_FINAL_DECL \
    public: \
        ::mgb::Typeinfo* dyn_typeinfo() const override final; \
    MGB_TYPEINFO_OBJ_DECL


//! put in the impl file of a final class inherited from DynTypeObj
#define MGB_DYN_TYPE_OBJ_FINAL_IMPL(_cls) \
    _MGB_DYN_TYPE_OBJ_FINAL_IMPL_TPL \
    ::mgb::Typeinfo* _cls::dyn_typeinfo() const { return &sm_typeinfo; } \
    MGB_TYPEINFO_OBJ_IMPL(_cls)


/*!
 * \brief base class for non-copyable objects
 */
class NonCopyableObj {
    NonCopyableObj(const NonCopyableObj&) = delete;
    NonCopyableObj& operator = (const NonCopyableObj&) = delete;

    public:

        NonCopyableObj() = default;
};

template<typename T>
class ReverseAdaptor {
    T &m_t;

    public:
        ReverseAdaptor(T &t):
            m_t(t)
        {}

        typename T::reverse_iterator begin() {
            return m_t.rbegin();
        }

        typename T::reverse_iterator end() {
            return m_t.rend();
        }
};

template<typename T>
class ConstReverseAdaptor {
    const T &m_t;

    public:
        ConstReverseAdaptor(const T &t):
            m_t(t)
        {}

        typename T::const_reverse_iterator begin() {
            return m_t.crbegin();
        }

        typename T::const_reverse_iterator end() {
            return m_t.crend();
        }
};

template<typename T>
ReverseAdaptor<T> reverse_adaptor(T &t) {
    return {t};
}

template<typename T>
ConstReverseAdaptor<T> reverse_adaptor(const T &t) {
    return {t};
}

/*!
 * \brief insertion sort for small arrays, to reduce code size
 * \tparam Iter iterator type, which must support ==, ++, --
 * \tparam Cmp comparator for strict less-than
 */
template<typename Iter,
    class Cmp = std::less<typename std::iterator_traits<Iter>::value_type>>
void small_sort(Iter begin, Iter end, const Cmp &cmp = {}) {
    if (begin == end)
        return;
    Iter i = begin;
    ++ i;
    for (; !(i == end); ++ i) {
        auto pivot = std::move(*i);
        Iter j = i;
        for (; ; ) {
            if (begin == j)
                break;
            Iter jnext = j;
            -- j;
            if (cmp(pivot, *j)) {
                *jnext = std::move(*j);
            } else {
                j = jnext;
                break;
            }
        }
        *j = std::move(pivot);
    }
}

/*!
 * \brief find key in container with out-of-boundary check
 */
template<class Container, class Key>
typename Container::iterator safe_find(Container &container, const Key &key) {
    typename Container::iterator iter = container.find(key);
    mgb_assert(iter != container.end());
    return iter;
}

/*!
 * \brief find key in container with out-of-boundary check
 */
template<class Container, class Key>
typename Container::const_iterator safe_find(
        const Container &container, const Key &key) {
    typename Container::const_iterator iter = container.find(key);
    mgb_assert(iter != container.end());
    return iter;
}

/*!
 * \brief find key in vector with out-of-boundary check
 */
template<class T, class Key>
typename SmallVector<T>::iterator safe_find(SmallVector<T>& container,
                                            const Key& key) {
    typename SmallVector<T>::iterator iter =
            std::find(container.begin(), container.end(), key);
    mgb_assert(iter != container.end());
    return iter;
}

/*!
 * \brief find key in container with out-of-boundary check
 */
template <class T, class Key>
typename SmallVector<T>::const_iterator safe_find(
        const SmallVector<T>& container, const Key& key) {
    typename SmallVector<T>::const_iterator iter =
            std::find(container.begin(), container.end(), key);
    mgb_assert(iter != container.end());
    return iter;
}

/*!
 * \brief find in vector
 */
template<class Key>
typename std::vector<Key>::iterator find(
        std::vector<Key> &vec, const Key &key) {
    return std::find(vec.begin(), vec.end(), key);
}

/*!
 * \brief find in vector
 */
template<class Key>
typename std::vector<Key>::const_iterator find(
        const std::vector<Key> &vec, const Key &key) {
    return std::find(vec.begin(), vec.end(), key);
}


/*!
 * \brief explicit hash specification for std::vector
 */
template<class Key>
struct HashTrait<std::vector<Key>> {
    static size_t eval(const std::vector<Key> &val) {
        size_t rst = hash(val.size());
        for (auto &&i: val)
            rst = hash_pair_combine(rst, ::mgb::hash(i));
        return rst;
    }
};

//! like python dict.get(key, default)
template<class Map>
const typename Map::value_type::second_type& get_map_with_default(
        const Map &map,
        const typename Map::value_type::first_type &key,
        const typename Map::value_type::second_type &default_ = {}) {
    auto iter = map.find(key);
    return iter == map.end() ? default_ : iter->second;
}

/*!
 * \brief raw memory storage for incomplete type Obj; Obj only needs to be
 *      complete in ctor and dtor
 */
template<class Obj, size_t SIZE, size_t ALIGN>
class alignas(ALIGN) IncompleteObjStorage {
    uint8_t m_mem[SIZE];

    public:
        IncompleteObjStorage() {
            static_assert(sizeof(Obj) <= SIZE && !(ALIGN % alignof(Obj)),
                          "SIZE and ALIGN do not match Obj");
            new (m_mem) Obj;
        }

        IncompleteObjStorage(const IncompleteObjStorage &rhs) {
            new (m_mem) Obj(rhs.get());
        }
        IncompleteObjStorage(IncompleteObjStorage &&rhs) noexcept {
            new (m_mem) Obj(std::move(rhs.get()));
        }

        IncompleteObjStorage& operator = (const IncompleteObjStorage &rhs) {
            get() = rhs.get();
            return *this;
        }

        IncompleteObjStorage& operator = (IncompleteObjStorage &&rhs) noexcept {
            get() = std::move(rhs.get());
            return *this;
        }

        ~IncompleteObjStorage() noexcept {
            get().~Obj();
        }

        Obj& get() {
            return *aliased_ptr<Obj>(m_mem);
        }

        const Obj& get() const {
            return const_cast<IncompleteObjStorage*>(this)->get();
        }
};

//! use size and align of another object
template<class Obj, class Mock>
using IncompleteObjStorageMock = IncompleteObjStorage<
    Obj, sizeof(Mock), alignof(Mock)>;

/*!
 * \brief container for arbitrary objects
 *
 * This container keeps a reference to added objects and allows retriving by
 * type. Objects of the same type form a stack, and supports add/pop.
 *
 * NOTE: This object is not thread-safe.
 */
class UserDataContainer {
    public:
        /*!
         * \brief base class for all user data
         *
         * Note that the impls must provide static typeinfo() (i.e. use
         * MGB_TYPEINFO_OBJ_DECL and MGB_TYPEINFO_OBJ_IMPL)
         */
        class UserData {
            public:
                virtual ~UserData() = default;
        };

        ~UserDataContainer() noexcept;

        /*!
         * \brief register new user data
         */
        template<typename T>
        T* add_user_data(std::shared_ptr<T> data) {
            static_assert(std::is_base_of<UserData, T>::value,
                    "must be derived from UserData");
            auto ptr = data.get();
            do_add(T::typeinfo(), std::move(data));
            return ptr;
        }

        /*!
         * \brief remove most recently added user data of a specific type
         * \return number of items removed
         */
        template<typename T>
        int pop_user_data() {
            static_assert(std::is_base_of<UserData, T>::value,
                    "must be derived from UserData");
            return do_pop(T::typeinfo());
        }

        /*!
         * \brief get user data
         * \return pair of (data object array ptr, number of data objects)
         */
        template<typename T>
        std::pair<T* const *, size_t> get_user_data() const {
            static_assert(std::is_base_of<UserData, T>::value,
                    "must be derived from UserData");
            auto ret = do_get(T::typeinfo());
            return {reinterpret_cast<T* const *>(ret.first), ret.second};
        }

        /*!
         * \brief get user data or create a new one; the registry for this user
         *      data type must contain only one instance
         */
        template<typename T, typename Maker>
        T* get_user_data_or_create(Maker &&maker) {
            static_assert(std::is_base_of<UserData, T>::value,
                    "must be derived from UserData");
            auto type = T::typeinfo();
            if (!m_storage.count(type)) {
                do_add(type, maker());
            }
            return static_cast<T*>(do_get_one(type));
        }

        //! get_user_data_or_create(), with std::make_shared as maker
        template<typename T>
        T* get_user_data_or_create() {
            return get_user_data_or_create<T>(std::make_shared<T>);
        }

        void clear_all_user_data();

        void swap(UserDataContainer& other) {
            m_refkeeper.swap(other.m_refkeeper);
            m_storage.swap(other.m_storage);
        }

    private:
        void do_add(Typeinfo *type, std::shared_ptr<UserData> ptr);
        std::pair<void* const*, size_t> do_get(Typeinfo *type) const;
        void* do_get_one(Typeinfo *type) const;
        int do_pop(Typeinfo *type);

        //! use a set to help erase
        std::unordered_set<std::shared_ptr<UserData>> m_refkeeper;
        ThinHashMap<Typeinfo*, SmallVector<void*, 1>> m_storage;
};

/*!
 * \brief continuation context, usually used for an async function
 * \tparam Args args to be passed to Next
 */
template<typename ...Args>
class ContinuationCtx {
    public:
        using Next = thin_function<void(Args...)>;
        using Err = thin_function<void(std::exception&)>;

        ContinuationCtx(const Next& next = {}, const Err& err = {}):
            m_next{next}, m_err{err}
        {}

        template<class ...T>
        void next(T &&... args) const {
            if (m_next)
                m_next(std::forward<T>(args)...);
        }

        void err(std::exception &exc) const {
            if (m_err)
                m_err(exc);
        }
    private:
        Next m_next;
        Err m_err;
};

//! a class that invokes given callbacks in the destructor
class CleanupCallback {
public:
    using Callback = thin_function<void()>;
    void add(Callback callback);

    ~CleanupCallback() noexcept(false);

private:
    SmallVector<Callback> m_callbacks;
};

}  // namespace mgb

#define _MGB_DEFINE_CLS_WITH_SUPER_IMPL(_tpl, _name, _base, ...)  \
class _name: public _base ,##__VA_ARGS__ { \
    public: \
        using Super = _tpl _base; \
    private:

/*!
 * \brief define a class which has Super defined to base
 */
#define MGB_DEFINE_CLS_WITH_SUPER(_name, _base, ...)  \
        _MGB_DEFINE_CLS_WITH_SUPER_IMPL(, _name, _base ,##__VA_ARGS__)

/*!
 * \brief define a class which has Super defined to base
 *
 * Used when this class is a template and base class has template
 */
#define MGB_DEFINE_CLS_WITH_SUPER_TPL(_name, _base, ...)  \
        _MGB_DEFINE_CLS_WITH_SUPER_IMPL(typename, _name, _base ,##__VA_ARGS__)
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
