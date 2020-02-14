/**
 * \file src/core/include/megbrain/utils/big_key_hashmap.h
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
#include "megbrain/utils/hash.h"

#include <functional>
#include <unordered_map>
#include <utility>

namespace mgb {
namespace big_key_hash_map {

namespace detail {
template <class HashEq, class... Keys>
class KeyTuple;
}  // namespace detail

/*!
 * \brief Hash map that is suitable for big key objects that are expensive to
 *      copy
 *
 * Multiple keys are supported. Every key type must be wrapped with Copy or Ref.
 * A Ref key would be kept as a reference, and copied only when it is inserted
 * as a new item. So lookup of existing items can be very fast.
 *
 * Hash and equality test of keys are be implemented by static methods
 * HashEq::hash and HashEq::eq, respectively.
 *
 * As a note, std seems not to mandate such optimization. For example,
 * std::unordered_map::operator[] takes a const reference to Key, but insert()
 * needs a pair. On cpprefence.com page of std::unordered_map::emplace, it is
 * stated that "The element may be constructed even if there already is an
 * element with the key in the container, in which case the newly constructed
 * element will be destroyed immediately."
 */
template <class Value, class HashEq, class... Keys>
class BigKeyHashMap {
    using KT = detail::KeyTuple<HashEq, Keys...>;
    struct HashOp {
        size_t operator()(const KT& key) const { return key.hash(); }
    };
    std::unordered_map<KT, Value, HashOp> m_map;

public:
    /*!
     * \brief get value from given key ref
     *
     * The keys wrapped by Ref are copied only when they need to be inserted
     *
     * \return pair (whether key is inserted (i.e. not existing before),
     *      corresponding value)
     */
    std::pair<bool, Value*> get(const typename Keys::raw_key&... keys);

    size_t size() const { return m_map.size(); }
};

/*!
 * \brief mark a key that should be copied; it must have a default ctor
 *
 * The key is usually a POD scalar type.
 */
template <typename T>
class Copy {
    T m_key;

public:
    using raw_key = T;

    struct ToOwn {
        ToOwn(Copy&) {}
        void apply() {}
    };

    Copy() = default;
    Copy(const T& key) : m_key{key} {}

    const T& visit() const { return m_key; }
    void free() {}
};

//! mark a key that should be referenced
template <typename T>
class Ref {
    const T* m_key = nullptr;

public:
    using raw_key = T;

    struct ToOwn {
        Ref& ref;
        std::unique_ptr<T> ptr;
        ToOwn(Ref& r) : ref{r}, ptr{new T{*ref.m_key}} {}
        void apply() { ref.m_key = ptr.release(); }
    };

    Ref() = default;
    Ref(const T& key) : m_key{&key} {}

    const T& visit() const { return *m_key; }
    void free() { delete const_cast<T*>(m_key); }
};

namespace detail {

template <class Key>
struct key_trait {
    static constexpr bool valid = false;
};
template <class Key>
struct key_trait<Copy<Key>> {
    static constexpr bool valid = true;
};
template <class Key>
struct key_trait<Ref<Key>> {
    static constexpr bool valid = true;
};

template <class Key>
struct check_valid_key {
    static_assert(key_trait<Key>::valid, "Key must be either Copy or Ref");
    using key = Key;
};

template <class HashEq, class Key>
class KeyTuple<HashEq, Key> {
    typename check_valid_key<Key>::key m_key;

    KeyTuple(const KeyTuple&) = delete;
    KeyTuple& operator=(const KeyTuple&) = delete;

protected:
    bool m_own = false;
    size_t m_hash = 0;

public:
    KeyTuple() = default;
    KeyTuple(KeyTuple&& rhs) { swap(rhs); }

    KeyTuple(const Key& key)
            : m_key{key}, m_hash{HashEq::hash(m_key.visit())} {}

    ~KeyTuple() {
        if (m_own) {
            m_key.free();
        }
    }

    KeyTuple& operator=(KeyTuple&& rhs) {
        swap(rhs);
        return *this;
    }

    void swap(KeyTuple& rhs) {
        std::swap(m_hash, rhs.m_hash);
        std::swap(m_own, rhs.m_own);
        std::swap(m_key, rhs.m_key);
    }

    void to_owned() {
        typename Key::ToOwn{m_key}.apply();
        m_own = true;
    }

    bool operator==(const KeyTuple& rhs) const {
        return m_hash == rhs.m_hash &&
               HashEq::eq(m_key.visit(), rhs.m_key.visit());
    }

    size_t hash() const { return m_hash; }
};

template <class HashEq, class Key, class... Others>
class KeyTuple<HashEq, Key, Others...> : protected KeyTuple<HashEq, Others...> {
    using Super = KeyTuple<HashEq, Others...>;

    typename check_valid_key<Key>::key m_key;

    KeyTuple(const KeyTuple&) = delete;
    KeyTuple& operator=(const KeyTuple&) = delete;

public:
    KeyTuple() = default;
    KeyTuple(KeyTuple&& rhs) { swap(rhs); }

    KeyTuple(const Key& key, const Others&... others)
            : Super(others...), m_key{key} {
        this->m_hash =
                hash_pair_combine(this->m_hash, HashEq::hash(m_key.visit()));
    }

    ~KeyTuple() {
        if (this->m_own) {
            m_key.free();
        }
    }

    KeyTuple& operator=(KeyTuple&& rhs) {
        swap(rhs);
        return *this;
    }

    void swap(KeyTuple& rhs) {
        Super::swap(rhs);
        std::swap(m_key, rhs.m_key);
    }

    void to_owned() {
        // two-step for exception safety
        typename Key::ToOwn to{m_key};
        Super::to_owned();
        to.apply();
    }

    bool operator==(const KeyTuple& rhs) const {
        return Super::operator==(rhs) &&
               HashEq::eq(m_key.visit(), rhs.m_key.visit());
    }

    using Super::hash;
};
}  // namespace detail

template <class Value, class HashEq, class... Keys>
std::pair<bool, Value*> BigKeyHashMap<Value, HashEq, Keys...>::get(
        const typename Keys::raw_key&... keys) {
    auto iter = m_map.emplace(KT{keys...}, Value{});
    if (iter.second) {
        MGB_TRY { const_cast<KT&>(iter.first->first).to_owned(); }
        MGB_CATCH(..., {
            m_map.erase(iter.first);
            throw;
        });
        return {true, &iter.first->second};
    }
    return {false, &iter.first->second};
}

}  // namespace big_key_hash_map
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

