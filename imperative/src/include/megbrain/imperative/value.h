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

    ValueRef(storage_t storage) { m_storage = storage; }

private:
    /**
     * \brief recursive get dest value storage and shorten path
     *
     * \return storage_t dest storage
     */
    storage_t& storage() const;

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
    bool is() const;

    /**
     * \brief try cast value as target type
     *
     * \tparam TValue target type
     * \return TValue* raw pointer if success, otherwise nullptr
     */
    template <typename TValue>
    const TValue* as() const;

    /**
     * \brief cast value to target type
     *
     * \tparam TValue target type
     * \return TValue& reference of value
     */
    template <typename TValue>
    const TValue& cast() const;

    /**
     * \brief like as(), but returns TypedValueRef instead
     *
     * \tparam TValue target type
     * \return TypedValueRef<TValue> reference if success, otherwise empty reference
     */
    template <typename TValue>
    inline TypedValueRef<TValue> as_ref() const;

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
    uint64_t id() const;
    size_t hash() const { return id(); }

    static ValueRef make(storage_t storage);

    static bool any_watching();

    friend class ValueWeakRef;
    template <typename T>
    friend class TypedValueRef;
    template <typename T>
    friend class ValueImpl;
    friend std::vector<ValueRef> apply(const Operator& op, Span<ValueRef> inputs);
};

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

    template <typename T>
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
template <typename T>
class ValueImpl : public Value {
protected:
    ValueImpl() : Value(TYPE_CODE) {}

public:
    using ref_t = TypedValueRef<T>;
    using weak_ref_t = TypedValueWeakRef<T>;

    static inline size_t TYPE_CODE = [] { return register_type(typeid(T)); }();

    /**
     * \brief helper function for construct a value
     *
     * \tparam TArgs types of arguments
     * \param args arguments
     * \return TypedValueRef<T> reference of value
     */
    template <typename... TArgs>
    static TypedValueRef<T> make(TArgs&&... args) {
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
template <typename T, typename TMixin>
class MixinValueImpl : public ValueImpl<T>, public TMixin {
public:
    using TMixin::TMixin;

    MixinValueImpl(TMixin mixin) : TMixin(std::move(mixin)) {}

public:
    void clear() override final { ((TMixin&)*this) = {}; }

    bool eq(const TMixin& value) const { return ((const TMixin&)*this) == value; }
};

template <typename TValue>
const TValue* ValueRef::as() const {
    static_assert(std::is_base_of_v<ValueImpl<TValue>, TValue>);
    auto storage = this->storage();
    if (!storage) {
        return nullptr;
    }
    if (storage->m_typecode != TValue::TYPE_CODE) {
        return nullptr;
    }
    return static_cast<TValue*>(storage.get());
}

template <typename TValue>
const TValue& ValueRef::cast() const {
    auto* ptr = as<TValue>();
    if (!ptr) {
        // if this is ErrorValue, rethrow directly
        storage()->try_rethrow();
        mgb_assert(
                ptr, "expect type %s, got %s", typeid(TValue).name(),
                to_string().c_str());
    }
    return *ptr;
}

template <typename TValue>
bool ValueRef::is() const {
    auto* ptr = as<TValue>();
    return ptr != nullptr;
}

template <typename TValue>
TypedValueRef<TValue> ValueRef::as_ref() const {
    if (!is<TValue>()) {
        return {};
    }
    return TypedValueRef<TValue>(*this);
}

/**
 * \brief ValueRef with concrete type, convenient for dereference
 *
 * \tparam T type of value
 */
template <typename T>
class TypedValueRef : public ValueRef {
private:
    TypedValueRef(ValueRef value) : ValueRef(value) {}

public:
    TypedValueRef() = default;
    const T& operator*() const { return this->template cast<T>(); }
    const T* operator->() const { return this->template as<T>(); }

    /**
     * \brief reset underlying value to another value
     *
     * \param successor new value
     */
    inline void reset(ValueRef successor) {
        mgb_assert(m_storage);
        mgb_assert(!m_storage->m_successor);
        if (m_storage->m_watching) {
            debug::notify_event("reset");
        }
        m_storage->clear();
        m_storage->m_successor = ValueRef(successor.storage());
    }

    friend class ValueRef;

    template <typename U>
    friend class ValueImpl;
};

template <typename T>
class TypedValueWeakRef : public ValueWeakRef {
private:
public:
    TypedValueWeakRef(ValueRef value) : ValueWeakRef(value) {}
    TypedValueWeakRef(ValueWeakRef value) : ValueWeakRef(value) {}
    TypedValueRef<T> lock() { return ValueWeakRef::lock().template as_ref<T>(); }
};

// TODO: add proxy value type, which is meant to be reset in the end

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
