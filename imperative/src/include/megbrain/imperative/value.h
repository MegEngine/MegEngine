/**
 * \file imperative/src/include/megbrain/imperative/value.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <list>
#include <map>
#include <memory>
#include <typeindex>
#include <vector>

#include "megbrain/common.h"
#include "megbrain/imperative/subgraph.h"
#include "megbrain/imperative/utils/allocator.h"
#include "megbrain/imperative/utils/debug.h"
#include "megbrain/imperative/utils/local_ptr.h"
#include "megbrain/imperative/utils/span.h"
#include "megbrain/imperative/utils/stats.h"

namespace mgb {
namespace imperative {

class Value;
class ValueRef;

template <typename T>
class TypedValueRef;

template <typename T>
class TypedValueWeakRef;

class Transformation;

class HostValue;
class DeviceValue;
class ShapeValue;
class DTypeValue;
class CompNodeValue;
class StringValue;

class Operator;

class ValueRefList;

template <typename T>
class Type {
private:
    const size_t m_code = T::TYPE_CODE;

public:
    inline size_t code() const { return m_code; }
};

enum class ValueKind {
    Primitive,
    Object,
};

/**
 * \brief an smart reference of value
 *
 * An ValueRef is either empty or refers to a value. Values are organized as linked lists
 * and only the tail node is valid. ValueRef stores a value node, and it may be
 * an invalid internal node. When you dereference it, it will check its successor,
 * automatically find the tail node and return. This list would be modified to reduce
 * list length by change value's successor, but a ValueRef always has steady m_storage
 * when not explicitly modified.
 * So we use m_storage to identify a ValueRef ( hash / equility / id ).
 */
class ValueRef {
public:
    using storage_t = LocalPtr<Value>;

protected:
    mutable storage_t m_storage;
    size_t m_id = std::numeric_limits<size_t>::max();

    inline ValueRef(storage_t storage);

private:
    /**
     * \brief recursive get dest value storage and shorten path
     *
     * \return storage_t dest storage
     */
    storage_t& storage() const;

    const Value* as(size_t typecode) const;

    bool is(size_t typecode) const;

public:
    ValueRef() = default;

    /**
     * \brief whether value is instance of target type or not
     *
     * \tparam TValue target type
     * \return true if type of value is TValue
     * \return false if empty or type of value is not TValue
     */
    template <typename TValue>
    inline bool is(Type<TValue> type = {}) const;

    /**
     * \brief try cast value as target type
     *
     * \tparam TValue target type
     * \return TValue* raw pointer if success, otherwise nullptr
     */
    template <typename TValue>
    inline const TValue* as(Type<TValue> type = {}) const;

    /**
     * \brief cast value to target type
     *
     * \tparam TValue target type
     * \return TValue& reference of value
     */
    template <typename TValue>
    inline const TValue& cast(Type<TValue> type = {}) const;

    /**
     * \brief like as(), but returns TypedValueRef instead
     *
     * \tparam TValue target type
     * \return TypedValueRef<TValue> reference if success, otherwise empty reference
     */
    template <typename TValue>
    inline const TypedValueRef<TValue>& as_ref(Type<TValue> type = {}) const;

    template <typename TValue>
    inline const TypedValueRef<TValue>& cast_ref(Type<TValue> type = {}) const;

    template <typename TValue>
    void on_cast_failure() const;

    operator bool() const { return bool(m_storage); }

    TypedValueRef<DeviceValue> dev_tensor() const;
    TypedValueRef<HostValue> numpy() const;
    TypedValueRef<CompNodeValue> device() const;
    TypedValueRef<ShapeValue> shape() const;
    TypedValueRef<DTypeValue> dtype() const;
    TypedValueRef<StringValue> name() const;
    bool is_scalar() const;

    void watch() const;
    void unwatch() const;
    bool watching() const;

    ValueRef unwrap() const;
    std::string to_string() const;
    std::string raw_type() const;
    uint64_t id() const { return m_id; }
    size_t hash() const { return id(); }

    static ValueRef make(storage_t storage);

    static bool any_watching();

    static const ValueRef nil;

    friend class ValueWeakRef;
    template <typename>
    friend class TypedValueRef;
    template <typename, ValueKind>
    friend class ValueImpl;
    friend ValueRefList apply(const Operator& op, Span<ValueRef> inputs);
};

inline const ValueRef ValueRef::nil;

template <>
struct ToStringTrait<ValueRef> {
public:
    std::string operator()(const ValueRef& value) const { return value.to_string(); }
};

class ValueWeakRef {
public:
    using storage_t = ValueRef::storage_t::weak_type;

protected:
    uint64_t m_id = std::numeric_limits<uint64_t>::max();
    mutable storage_t m_storage;

public:
    ValueWeakRef() = default;
    ValueWeakRef(ValueRef value) : m_id(value.id()), m_storage(value.m_storage) {}

    /**
     * \brief try promote to ValueRef
     *
     * \return ValueRef strong ref if value exists, otherwise empty ref
     */
    ValueRef lock();
    size_t hash() const { return m_id; }

    bool operator==(const ValueWeakRef& rhs) const {
        return m_storage == rhs.m_storage;
    }
    bool operator!=(const ValueWeakRef& rhs) const { return !(*this == rhs); }
};

/**
 * \brief base class for all generic value involved in dispatch system
 *
 */
class Value : public NonCopyableObj {
private:
    uint64_t m_id = std::numeric_limits<uint64_t>::max();
    size_t m_typecode = 0;
    ValueRef m_successor;
    size_t m_watching = 0;

protected:
    Value(size_t typecode);

public:
    size_t typecode() const { return m_typecode; }
    const std::type_index type() const { return registered_types()[m_typecode]; }

    static size_t register_type(std::type_index type);
    static const std::vector<std::type_index>& registered_types();

    static void register_value(ValueRef value);
    static ValueRef get_value_by_id(uint64_t id);
    static void begin_record_values();
    static std::vector<ValueRef> end_record_values();

    virtual std::string to_string() const = 0;

    /**
     * \brief clear all states of this value
     *
     */
    virtual void clear() = 0;

    virtual void on_watch() {}
    virtual void on_unwatch() {}

    friend class ValueRef;
    friend class ValueWeakRef;

    template <typename, ValueKind>
    friend class ValueImpl;
    template <typename T>
    friend class TypedValueRef;

    ~Value();

private:
    void try_rethrow();
};

/**
 * \brief base class of values, with typecode and factory method support
 *
 * \tparam T type of value
 */
template <typename T, ValueKind Kind>
class ValueImpl : public Value {
protected:
    ValueImpl() : Value(TYPE_CODE) {}

public:
    using ref_t = TypedValueRef<T>;
    using weak_ref_t = TypedValueWeakRef<T>;

    static inline const size_t TYPE_CODE = [] { return register_type(typeid(T)); }();
    static constexpr ValueKind KIND = Kind;

    /**
     * \brief helper function for construct a value
     *
     * \tparam TArgs types of arguments
     * \param args arguments
     * \return TypedValueRef<T> reference of value
     */
    template <typename... TArgs>
    static MGB_NOINLINE TypedValueRef<T> make(TArgs&&... args) {
        static_assert(std::is_final_v<T>);
        return ValueRef::make(LocalPtr<Value>::make<T>(std::forward<TArgs&&>(args)...));
    }
};

/**
 * \brief base class of values, with mixin support
 *
 * \tparam T type of value
 * \tparam TMixin type of mixin class
 */
template <typename T, ValueKind Kind, typename TMixin>
class MixinValueImpl : public ValueImpl<T, Kind>, public TMixin {
public:
    using TMixin::TMixin;

    MixinValueImpl(TMixin mixin) : TMixin(std::move(mixin)) {}

public:
    void clear() override final { ((TMixin&)*this) = {}; }

    bool eq(const TMixin& value) const { return ((const TMixin&)*this) == value; }
};

inline ValueRef::ValueRef(storage_t storage) {
    // mgb_assert(storage);
    m_storage = storage;
    m_id = m_storage->m_id;
}

template <typename TValue>
inline const TValue* ValueRef::as(Type<TValue> type) const {
    // auto _ = Stats::time_value_as.time_scope();
    static_assert(std::is_base_of_v<Value, TValue>);
    return static_cast<const TValue*>(as(type.code()));
}

template <typename TValue>
inline const TValue& ValueRef::cast(Type<TValue> type) const {
    // auto _ = Stats::time_value_cast.time_scope();
    auto* ptr = as<TValue>(type);
    if (mgb_unlikely(!ptr)) {
        on_cast_failure<TValue>();
    }
    return static_cast<const TValue&>(*ptr);
}

template <typename TValue>
inline bool ValueRef::is(Type<TValue> type) const {
    // auto _ = Stats::time_value_is.time_scope();
    return is(type.code());
}

template <typename TValue>
inline const TypedValueRef<TValue>& ValueRef::as_ref(Type<TValue> type) const {
    if (!is<TValue>(type)) {
        return TypedValueRef<TValue>::nil;
    }
    return *reinterpret_cast<const TypedValueRef<TValue>*>(this);
}

template <typename TValue>
inline const TypedValueRef<TValue>& ValueRef::cast_ref(Type<TValue> type) const {
    if (!m_storage) {
        return TypedValueRef<TValue>::nil;
    }
    if (mgb_unlikely(!is<TValue>(type))) {
        on_cast_failure<TValue>();
    }
    return *reinterpret_cast<const TypedValueRef<TValue>*>(this);
}

template <typename TValue>
void ValueRef::on_cast_failure() const {
    // if this is ErrorValue, rethrow directly
    storage()->try_rethrow();
    mgb_assert(
            storage()->m_typecode != TValue::TYPE_CODE, "expect type %s, got %s",
            typeid(TValue).name(), to_string().c_str());
}

/**
 * \brief ValueRef with concrete type, convenient for dereference
 *
 * \tparam T type of value
 */
template <typename T>
class TypedValueRef : public ValueRef {
private:
    TypedValueRef(ValueRef value) : ValueRef(std::move(value)) {}

public:
    TypedValueRef() = default;
    const T& operator*() const {
        if constexpr (T::KIND == ValueKind::Object) {
            return this->template cast<T>();
        } else if constexpr (T::KIND == ValueKind::Primitive) {
            if (!m_storage) {
                on_cast_failure<T>();
            }
            return static_cast<const T&>(*m_storage);
        } else {
            static_assert(!std::is_same_v<T, T>);
        }
    }
    const T* operator->() const {
        if constexpr (T::KIND == ValueKind::Object) {
            return this->template as<T>();
        } else if constexpr (T::KIND == ValueKind::Primitive) {
            return static_cast<const T*>(m_storage.get());
        } else {
            static_assert(!std::is_same_v<T, T>);
        }
    }

    /**
     * \brief reset underlying value to another value
     *
     * \param successor new value
     */
    inline void reset(ValueRef successor) {
        static_assert(T::KIND == ValueKind::Object);
        mgb_assert(m_storage);
        mgb_assert(!m_storage->m_successor);
        if (m_storage->m_watching) {
            debug::notify_event("reset");
        }
        m_storage->clear();
        m_storage->m_successor = ValueRef(successor.storage());
    }

    static inline const TypedValueRef nil;

    friend class ValueRef;

    template <typename, ValueKind>
    friend class ValueImpl;
};

template <typename T>
class TypedValueWeakRef : public ValueWeakRef {
private:
public:
    TypedValueWeakRef(ValueRef value) : ValueWeakRef(value) {}
    TypedValueWeakRef(ValueWeakRef value) : ValueWeakRef(value) {}
    TypedValueRef<T> lock() {
        auto value = ValueWeakRef::lock();
        if (value) {
            return value.template as_ref<T>();
        } else {
            return {};
        }
    }
};

// TODO: add proxy value type, which is meant to be reset in the end

class ValueRefList {
private:
    ValueRef* m_data = nullptr;
    size_t m_size = 0;
    std::aligned_storage_t<sizeof(ValueRef), alignof(ValueRef)> m_storage;

private:
    void init(size_t nr_elems);
    ValueRef* inline_storage() { return reinterpret_cast<ValueRef*>(&m_storage); }

public:
    ValueRefList() = default;
    ValueRefList(size_t nr_elems);
    ValueRefList(ValueRef item);
    // ValueRefList(std::initializer_list<ValueRef> values);
    template <typename TIterator>
    ValueRefList(TIterator begin, TIterator end);
    ValueRefList(const ValueRefList& rhs);
    ValueRefList(ValueRefList&& rhs);
    ValueRefList& operator=(const ValueRefList& rhs);
    ValueRefList& operator=(ValueRefList&& rhs);
    ~ValueRefList();
    void clear();

    ValueRef* begin() { return m_data; }
    ValueRef* end() { return m_data + m_size; }
    const ValueRef* cbegin() const { return m_data; }
    const ValueRef* cend() const { return m_data + m_size; }
    size_t size() const { return m_size; }
    ValueRef& at(size_t idx) {
        mgb_assert(idx < m_size);
        return m_data[idx];
    }
    const ValueRef& at(size_t idx) const {
        mgb_assert(idx < m_size);
        return m_data[idx];
    }
    ValueRef& operator[](size_t idx) { return m_data[idx]; }
    const ValueRef& operator[](size_t idx) const { return m_data[idx]; }
    ValueRef* data() { return m_data; }
    const ValueRef* data() const { return m_data; }
    bool empty() const { return m_size == 0; }
    ValueRef& front() {
        mgb_assert(m_size > 1);
        return m_data[0];
    }
    ValueRef& back() {
        mgb_assert(m_size > 1);
        return m_data[m_size - 1];
    }
};

template <typename TIterator>
ValueRefList::ValueRefList(TIterator begin, TIterator end) : ValueRefList(end - begin) {
    for (size_t i = 0; i < m_size; ++i) {
        m_data[i] = *(begin + i);
    }
}

inline ValueRefList::ValueRefList(ValueRef item) : m_data(inline_storage()), m_size(1) {
    new (m_data) ValueRef();
    m_data[0] = std::move(item);
}

/*class ValueRefList : public SmallVector<ValueRef, 1> {
public:
    using SmallVector::SmallVector;
};*/

}  // namespace imperative
}  // namespace mgb

namespace std {

template <>
struct hash<mgb::imperative::ValueWeakRef> {
    std::size_t operator()(const mgb::imperative::ValueWeakRef& weak_ref) const {
        return weak_ref.hash();
    }
};

template <>
struct hash<mgb::imperative::ValueRef> {
    std::size_t operator()(const mgb::imperative::ValueRef& ref) const {
        return ref.hash();
    }
};

}  // namespace std
