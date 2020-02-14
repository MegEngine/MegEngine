/**
 * \file src/core/include/megbrain/utils/thin/hash_table.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/common.h"
#include "megbrain/utils/mempool.h"
#include "megbrain/utils/metahelper_basic.h"

#include <unordered_map>
#include <type_traits>
#include <cstdint>

namespace mgb {
namespace thin_hash_table {

    //! wrapper for std::unordered_map that addes MGB_NOINLINE to some methods
    template<class Key, class Val, class Hash, class Eq>
    class NoinlineHashMap {
        public:
            using Impl = std::unordered_map<Key, Val, Hash, Eq>;
            using impl_iter = typename Impl::iterator;
            using impl_citer = typename Impl::const_iterator;
            using value_type = typename Impl::value_type;

            NoinlineHashMap() = default;

            MGB_NOINLINE
            NoinlineHashMap(const NoinlineHashMap &) = default;

#ifndef _MSC_VER
            NoinlineHashMap(NoinlineHashMap &&) noexcept = default;
#else
            // Visual C++ needs exception
            NoinlineHashMap(NoinlineHashMap &&) = default;
#endif
            MGB_NOINLINE
            ~NoinlineHashMap() noexcept = default;

            MGB_NOINLINE
            void operator = (const NoinlineHashMap &rhs) noexcept {
                m_impl = rhs.m_impl;
            }

            MGB_NOINLINE
            void operator = (NoinlineHashMap &&rhs) noexcept {
                m_impl = std::move(rhs.m_impl);
            }

            MGB_NOINLINE
            std::pair<typename Impl::iterator, bool> insert(Key key) {
                return m_impl.emplace(key, Val());
            }

            MGB_NOINLINE
            typename Impl::iterator erase(typename Impl::const_iterator pos) {
                return m_impl.erase(pos);
            }

            MGB_NOINLINE
            impl_iter find(Key key) {
                return m_impl.find(key);
            }

            impl_citer find(Key key) const {
                return const_cast<NoinlineHashMap*>(this)->find(key);
            }

            MGB_NOINLINE
            const Val& at(Key key) const {
                return m_impl.at(key);
            }

            MGB_NOINLINE
            void clear() {
                m_impl.clear();
            }

            MGB_NOINLINE
            void swap(NoinlineHashMap &other) {
                m_impl.swap(other.m_impl);
            }

            MGB_NOINLINE
            void reserve(size_t size) {
                m_impl.reserve(size);
            }

            Impl& impl() {
                return m_impl;
            }

            const Impl& impl() const {
                return m_impl;
            }

        private:
            Impl m_impl;
    };

    template<int Size>
    struct PODHashEq;

    template<class EquivKey>
    struct PODHashEqImpl {
        template<class T>
        size_t operator() (const T &t) const {
            return std::hash<EquivKey>{}(raw_cast<EquivKey>(t));
        }

        template<class T>
        bool operator() (const T &a, const T &b) const {
            return raw_cast<EquivKey>(a) == raw_cast<EquivKey>(b);
        }
    };

    template<> struct PODHashEq<1>: public PODHashEqImpl<uint8_t > { };
    template<> struct PODHashEq<2>: public PODHashEqImpl<uint16_t> { };
    template<> struct PODHashEq<4>: public PODHashEqImpl<uint32_t> { };
    template<> struct PODHashEq<8>: public PODHashEqImpl<uint64_t> { };

    struct tag_thin_hash_check_complete;

    //! trait of value to support incomplete types
    template <typename T,
              bool is_complete = is_complete_v<T, tag_thin_hash_check_complete>>
    struct ValueTrait {
        static constexpr bool can_embed_in_pair =
                ((sizeof(T) - 1) / alignof(T) + 1) * alignof(T) <=
                sizeof(void*);
        static constexpr bool trivial_dtor =
                std::is_trivially_destructible<T>::value;
    };
    template <typename T>
    struct ValueTrait<T, false> {
        static constexpr bool can_embed_in_pair = false;
        static constexpr bool trivial_dtor = false;
    };

    template <class Val,
              bool can_embed_in_pair = ValueTrait<Val>::can_embed_in_pair>
    class ThinHashMapMemValueStorage;
    template<class Val_>
    class ThinHashMapMemValueStorage<Val_, true> {
        public:
            using Val = Val_;

            static Val* visit(void **ptr) {
                // we need to cast ptr to void* because Val may be const T*
                return reinterpret_cast<Val*>(reinterpret_cast<void*>(ptr));
            }

            template<typename ...InitArgs>
            Val* alloc(void **ptr, InitArgs &&... init) {
                auto pv = visit(ptr);
                new(pv) Val(std::forward<InitArgs>(init)...);
                return pv;
            }

            void record_free(Val*) {
            }

            void swap(ThinHashMapMemValueStorage &) {
            }

            void clear() {
            }
    };
    template<class Val_>
    class ThinHashMapMemValueStorage<Val_, false> {
        MemPoolStorage m_storage;
        public:
            using Val = Val_;

            static Val* visit(void **ptr) {
                return static_cast<Val*>(*ptr);
            }

            template<typename ...InitArgs>
            Val* alloc(void **ptr, InitArgs &&... init) {
                auto pv = static_cast<Val*>(m_storage.alloc(sizeof(Val)));
                new(pv) Val(std::forward<InitArgs>(init)...);
                *ptr = pv;
                return pv;
            }

            void record_free(Val *ptr) {
                m_storage.free(ptr);
            }

            void swap(ThinHashMapMemValueStorage &other) {
                m_storage.swap(other.m_storage);
            }

            void clear() {
                m_storage.clear();
            }
    };


    template<class ValStorage,
        bool trivial_dtor = ValueTrait<typename ValStorage::Val>::trivial_dtor>
    struct ThinHashMapDtorAux;
    template<class ValStorage>
    struct ThinHashMapDtorAux<ValStorage, true> {
        using Val = typename ValStorage::Val;

        template <typename Map>
        static void foreach_map_item(Map&) {}

        static void single(Val&) {}
    };
    template<class ValStorage>
    struct ThinHashMapDtorAux<ValStorage, false> {
        using Val = typename ValStorage::Val;

        template <typename Map>
        MGB_NOINLINE static void foreach_map_item(Map& map) {
            // check triviality again to handle incomplete types
            if (!std::is_trivially_destructible<Val>::value) {
                for (auto&& i : map) {
                    ValStorage::visit(&i.second)->~Val();
                }
            }
        }

        static void single(Val& v) {
            if (!std::is_trivially_destructible<Val>::value) {
                v.~Val();
            }
        }
    };

    /*!
     * \brief a container like std::unordered_map, with small binary size after
     *      compile
     *
     * Key must be location-invariant POD without padding, and Key types with
     * the same size/alignment would use the same map implementation
     *
     * Value type \p Val is allowed to be incomplete at the time of this map
     * definition.
     */
    template<class Key, class Val>
    class ThinHashMap {
        static_assert(is_location_invariant<Key>::value,
                "key must be location invariant");
        using ImplKey = std::aligned_storage_t<sizeof(Key), alignof(Key)>;
        using ImplHashEq = PODHashEq<sizeof(Key)>;
        using Impl = NoinlineHashMap<ImplKey, void*, ImplHashEq, ImplHashEq>;
        using ValStorage = ThinHashMapMemValueStorage<Val>;
        using ValDtorAux = ThinHashMapDtorAux<ValStorage>;

        Impl m_impl;
        ValStorage m_val_storage;

        static Key make_key(const ImplKey &key) {
            return raw_cast<Key>(key);
        }

        static ImplKey make_impl_key(const Key &key) {
            return raw_cast<ImplKey>(key);
        }

        void clear_values() {
            ValDtorAux::foreach_map_item(m_impl.impl());
            m_val_storage.clear();
        }

        typename Impl::Impl::iterator do_erase(
                typename Impl::Impl::const_iterator pos) {
            auto ptr = ValStorage::visit(const_cast<void**>(&pos->second));
            ValDtorAux::single(*ptr);
            m_val_storage.record_free(ptr);
            return m_impl.erase(pos);
        }

        public:
            using value_type = std::pair<const Key, Val>;

            ThinHashMap() = default;

#ifndef _MSC_VER
            ThinHashMap(ThinHashMap &&rhs) noexcept = default;
#else
            // Visual C++ needs exception
            ThinHashMap(ThinHashMap &&rhs) = default;
#endif

            ThinHashMap(std::initializer_list<value_type> init) {
                for (auto &&i: init) {
                    this->insert(i);
                }
            }

            ThinHashMap& operator = (ThinHashMap &&rhs) noexcept {
                swap(rhs);
                return *this;
            }

            ~ThinHashMap() noexcept {
                clear_values();
            }

            /* -------------- iterators --------------  */
            template<class T>
            struct value_type_impl {
                const Key first;
                T &second;

                value_type_impl(Key key, T &val):
                    first(key), second(val)
                {
                }
            };

            class citer_impl {
                using impl_iter = typename Impl::Impl::const_iterator;
                impl_iter m_iter;

                public:
                    using value_type = value_type_impl<const Val>;
                    using pointer = value_type*;
                    using reference = value_type&;
                    using iterator_category = std::bidirectional_iterator_tag;
                    using difference_type = ptrdiff_t;

                    citer_impl() = default;
                    citer_impl(impl_iter iter):
                        m_iter{iter}
                    {
                    }

                    citer_impl& operator++() {
                        ++ m_iter;
                        return *this;
                    }

                    citer_impl& operator--() {
                        ++ m_iter;
                        return *this;
                    }

                    value_type operator* () const {
                        return {
                            make_key(m_iter->first),
                            *ValStorage::visit(
                                    const_cast<void**>(&m_iter->second))
                        };
                    }

                    value_type* operator-> () const {
                        return new (&m_tmp_val_for_ptr)value_type(
                                make_key(m_iter->first),
                                *ValStorage::visit(
                                    const_cast<void**>(&m_iter->second)));
                    }

                    bool operator == (const citer_impl &rhs) const {
                        return m_iter == rhs.m_iter;
                    }

                    bool operator != (const citer_impl &rhs) const {
                        return m_iter != rhs.m_iter;
                    }

                    impl_iter impl() const {
                        return m_iter;
                    }

                private:
                    mutable std::aligned_storage_t<
                        sizeof(value_type), alignof(value_type)>
                        m_tmp_val_for_ptr;
            };

            class iter_impl: public citer_impl {
                public:
                    using value_type = value_type_impl<Val>;
                    using pointer = value_type*;
                    using reference = value_type&;

                    using citer_impl::citer_impl;

                    value_type operator* () const {
                        auto ret = citer_impl::operator*();
                        return {ret.first, const_cast<Val&>(ret.second)};
                    }

                    value_type* operator-> () const {
                        return aliased_ptr<value_type>(
                                citer_impl::operator->());
                    }
            };

            using iterator = iter_impl;
            using const_iterator = citer_impl;

            iterator begin() {
                return iterator(m_impl.impl().begin());
            }

            iterator end() {
                return iterator(m_impl.impl().end());
            }

            const_iterator begin() const {
                return m_impl.impl().begin();
            }

            const_iterator end() const {
                return m_impl.impl().end();
            }

            const_iterator cbegin() const {
                return m_impl.impl().cbegin();
            }

            const_iterator cend() const {
                return m_impl.impl().cend();
            }

            /* -------------- capacity --------------  */
            bool empty() const {
                return m_impl.impl().empty();
            }

            size_t size() const {
                return m_impl.impl().size();
            }


            /* -------------- modifiers --------------  */
            void clear() {
                clear_values();
                m_impl.clear();
            }

            std::pair<iterator, bool> insert(const value_type &data) {
                auto ret = m_impl.insert(make_impl_key(data.first));
                if (ret.second) {
                    m_val_storage.alloc(&ret.first->second, data.second);
                }
                return {iterator(ret.first), ret.second};
            }

            std::pair<iterator, bool> insert(value_type &&data) {
                auto ret = m_impl.insert(make_impl_key(data.first));
                if (ret.second) {
                    m_val_storage.alloc(&ret.first->second,
                            std::move(data.second));
                }
                return {iterator(ret.first), ret.second};
            }

            template<typename ...Kargs, typename ...Vargs>
            std::pair<iterator, bool> emplace(std::piecewise_construct_t,
                    const std::tuple<Kargs...> &k,
                    const std::tuple<Vargs...> &v) {
                return do_emplace_picewise(k, v,
                        make_index_sequence<sizeof...(Kargs)>{},
                        make_index_sequence<sizeof...(Vargs)>{});
            }

            template<typename U, typename V>
            std::pair<iterator, bool> emplace(U &&u, V &&v) {
                return emplace(std::piecewise_construct,
                        std::forward_as_tuple(std::forward<U>(u)),
                        std::forward_as_tuple(std::forward<V>(v)));
            }

            Val& operator [] (const Key &key) {
                auto ret = m_impl.insert(make_impl_key(key));
                void **ptr = &ret.first->second;
                if (ret.second) {
                    return *m_val_storage.alloc(ptr);
                }
                return *ValStorage::visit(ptr);
            }

            iterator erase(const_iterator pos) {
                return iterator(do_erase(pos.impl()));
            }

            size_t erase(const Key &key) {
                auto iter = m_impl.find(make_impl_key(key));
                if (iter != m_impl.impl().end()) {
                    do_erase(iter);
                    return 1;
                }
                return 0;
            }

            void swap(ThinHashMap &other) noexcept {
                if (this != &other) {
                    m_impl.swap(other.m_impl);
                    m_val_storage.swap(other.m_val_storage);
                }
            }

            void reserve(size_t size) {
                m_impl.reserve(size);
            }

            /* -------------- lookup --------------  */
            size_t count(const Key &key) const {
                return m_impl.find(make_impl_key(key)) != m_impl.impl().end();
            }

            const_iterator find(const Key &key) const {
                return m_impl.find(make_impl_key(key));
            }

            iterator find(const Key &key) {
                return iterator(m_impl.find(make_impl_key(key)));
            }

            const Val& at(const Key& key) const {
                return *ValStorage::visit(const_cast<void**>(&m_impl.at(
                                make_impl_key(key))));
            }

            Val& at(const Key& key) {
                return const_cast<Val&>(
                        static_cast<const ThinHashMap*>(this)->at(key));
            }

        private:
            template<typename ...Kargs, typename ...Vargs,
                size_t ...Kidx, size_t ...Vidx>
            std::pair<iterator, bool> do_emplace_picewise(
                    const std::tuple<Kargs...> &kargs,
                    const std::tuple<Vargs...> &vargs,
                    index_sequence<Kidx...>, index_sequence<Vidx...>) {
                Key key(std::forward<Kargs>(std::get<Kidx>(kargs))...);
                auto ret = m_impl.insert(make_impl_key(key));
                if (ret.second) {
                    m_val_storage.alloc(
                            &ret.first->second,
                            std::forward<Vargs>(std::get<Vidx>(vargs))...);
                }
                return {iterator(ret.first), ret.second};
            }

    };

    /*!
     * \brief a container like std::unordered_set, with small binary size after
     *      compile
     *
     * Key must be location-invariant, and Key types with the same
     * size/alignment would use the same set implementation
     *
     * Note: map is used as underlying impl, to trade runtime memory for binary
     * size
     */
    template<class Key>
    class ThinHashSet {
        static_assert(is_location_invariant<Key>::value,
                "key must be location invariant");
        using ImplKey = std::aligned_storage_t<sizeof(Key), alignof(Key)>;
        using ImplHashEq = PODHashEq<sizeof(Key)>;
        using Impl = NoinlineHashMap<ImplKey, void*, ImplHashEq, ImplHashEq>;

        Impl m_impl;

        static ImplKey make_impl_key(const Key &key) {
            return raw_cast<ImplKey>(key);
        }

        static Key make_key(const ImplKey &key) {
            return raw_cast<Key>(key);
        }

        public:
            /* -------------- constructors --------------  */

            ThinHashSet() = default;

            template<class Iter>
            ThinHashSet(Iter first, Iter last) {
                while (first != last) {
                    m_impl.insert(make_impl_key(*first));
                    ++ first;
                }
            }

            ThinHashSet(std::initializer_list<Key> init):
                ThinHashSet(init.begin(), init.end())
            {
            }

            /* -------------- iterators --------------  */
            class const_iterator {
                using impl_iter = typename Impl::Impl::const_iterator;
                impl_iter m_iter;

                public:
                    using value_type = Key;
                    using pointer = value_type*;
                    using reference = value_type&;
                    using iterator_category = std::bidirectional_iterator_tag;
                    using difference_type = ptrdiff_t;

                    const_iterator() = default;
                    const_iterator(impl_iter iter):
                        m_iter{iter}
                    {
                    }

                    const_iterator& operator++() {
                        ++ m_iter;
                        return *this;
                    }

                    const_iterator& operator--() {
                        ++ m_iter;
                        return *this;
                    }

                    Key operator* () const {
                        return make_key(m_iter->first);
                    }

                    bool operator == (const const_iterator &rhs) const {
                        return m_iter == rhs.m_iter;
                    }

                    bool operator != (const const_iterator &rhs) const {
                        return m_iter != rhs.m_iter;
                    }

                    impl_iter impl() const {
                        return m_iter;
                    }
            };

            using iterator = const_iterator;

            const_iterator begin() const {
                return m_impl.impl().begin();
            }

            const_iterator end() const {
                return m_impl.impl().end();
            }

            const_iterator cbegin() const {
                return m_impl.impl().cbegin();
            }

            const_iterator cend() const {
                return m_impl.impl().cend();
            }

            /* -------------- capacity --------------  */
            bool empty() const {
                return m_impl.impl().empty();
            }

            size_t size() const {
                return m_impl.impl().size();
            }

            /* -------------- modifiers --------------  */
            void clear() {
                m_impl.clear();
            }

            std::pair<iterator, bool> insert(const Key &key) {
                auto ret = m_impl.insert(make_impl_key(key));
                return {iterator(ret.first), ret.second};
            }

            std::pair<iterator, bool> emplace(const Key &key) {
                // since Key is POD, no need to construct inplace
                return insert(key);
            }

            iterator erase(const_iterator pos) {
                return iterator(m_impl.erase(pos.impl()));
            }

            size_t erase(const Key &key) {
                auto iter = m_impl.find(make_impl_key(key));
                if (iter != m_impl.impl().end()) {
                    m_impl.erase(iter);
                    return 1;
                }
                return 0;
            }

            void swap(ThinHashSet &other) noexcept {
                if (this != &other) {
                    m_impl.swap(other.m_impl);
                }
            }

            void reserve(size_t size) {
                m_impl.reserve(size);
            }

            /* -------------- lookup --------------  */
            size_t count(const Key &key) const {
                return find(key) != m_impl.impl().end();
            }

            const_iterator find(const Key &key) const {
                return m_impl.find(make_impl_key(key));
            }
    };

} // namespace thin_hash_table

#if 1
using thin_hash_table::ThinHashSet;
using thin_hash_table::ThinHashMap;
#else
template<class Key>
using ThinHashSet = std::unordered_set<Key>;
template<class Key, class Value>
using ThinHashMap = std::unordered_map<Key, Value>;
#endif

} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

