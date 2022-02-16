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

/**
 * \brief base class of all value types
 */
class IType : public NonCopyableObj {
private:
    std::string m_name;
    // TODO: count values, or make an linkedlist

public:
    IType(std::string name) : m_name(std::move(name)) {}

    const std::string& name() const { return m_name; }

    bool operator==(const IType& rhs) const { return this == &rhs; }

    bool operator!=(const IType& rhs) const { return this != &rhs; }
};

/**
 * \brief type of values.
 *
 * \tparam T ctype of value
 */
template <typename T>
class Type : public IType {
protected:
    Type(std::string name) : IType(std::move(name)) {}
    // TODO: each type owns an allocator

public:
    /**
     * \brief helper function for construct a value
     *
     * \tparam TArgs types of arguments
     * \param args arguments
     * \return TypedValueRef<T> reference of value
     */
    template <typename... TArgs>
    TypedValueRef<T> make(TArgs&&... args) const;
};

/**
 * \brief type of primitive values.
 *
 * \tparam T ctype of value
 */
template <typename T>
class PrimitiveType : public Type<T> {
private:
    PrimitiveType();

public:
    static inline PrimitiveType instance;
};

/**
 * \brief type of object values.
 *
 * \tparam T ctype of value
 */
template <typename T>
class ObjectType : public Type<T> {
public:
    ObjectType(std::string name) : Type<T>(name) {}
};

/**
 * \brief an smart reference of value
 *
 * An ValueRef is either empty or refers to a value. Values are organized as linked lists
 * and only the tail node is valid. ValueRef stores a value node, and it may be
 * an invalid internal node. When you dereference it, it will check its successor,
 * automatically find the tail node and return. This list would be modified to reduce
 * list length by change value's successor, but a steady id was kept in ValueRef
 * so we can use it for identify a ValueRef ( hash / equility / id ).
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

    const Value* as(const IType& type) const;

public:
    ValueRef() = default;

    /**
     * \brief whether value is instance of target type or not
     *
     * \param type target type
     * \return true if type of value is instance of type
     * \return false if empty or type of value is not instance of type
     */
    bool is(const IType& type) const;

    /**
     * \brief try cast value as target type
     *
     * \tparam type target type
     * \return TValue* raw pointer if success, otherwise nullptr
     */
    template <typename TValue>
    inline const TValue* as(const Type<TValue>& type) const;

    /**
     * \brief cast value to target type
     *
     * \param type target type
     * \return TValue& reference of value
     */
    template <typename TValue>
    inline const TValue& cast(const Type<TValue>& type) const;

    /**
     * \brief like as(), but returns TypedValueRef instead
     *
     * \param type target type
     * \return TypedValueRef<TValue> reference if success, otherwise empty reference
     */
    template <typename TValue>
    inline const TypedValueRef<TValue>& as_ref(const Type<TValue>& type) const;

    /**
     * \brief like cast(), but allow empty value and returns TypedValueRef instead
     *
     * \param type target type
     * \return TypedValueRef<TValue> reference if success, otherwise empty reference
     */
    template <typename TValue>
    inline const TypedValueRef<TValue>& cast_ref(const Type<TValue>& type) const;

    template <typename TValue>
    inline std::enable_if_t<TValue::is_primitive, bool> is() const {
        return is(PrimitiveType<TValue>::instance);
    }

    template <typename TValue>
    inline std::enable_if_t<TValue::is_primitive, const TValue*> as() const {
        return as(PrimitiveType<TValue>::instance);
    }

    template <typename TValue>
    inline std::enable_if_t<TValue::is_primitive, const TValue&> cast() const {
        return cast(PrimitiveType<TValue>::instance);
    }

    template <typename TValue>
    inline std::enable_if_t<TValue::is_primitive, const TypedValueRef<TValue>&> as_ref()
            const {
        return as_ref(PrimitiveType<TValue>::instance);
    }

    template <typename TValue>
    inline std::enable_if_t<TValue::is_primitive, const TypedValueRef<TValue>&>
    cast_ref() const {
        return cast_ref(PrimitiveType<TValue>::instance);
    }

    void on_cast_failure(const IType& type) const;

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
    ValueWeakRef(const ValueRef& value)
            : m_id(value.id()), m_storage(value.m_storage) {}

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
    const IType* m_type = nullptr;
    ValueRef m_successor;
    size_t m_watching = 0;

protected:
    Value();

public:
    const IType& type() const { return *m_type; }

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
    friend class TypedValueRef;

    template <typename T>
    friend class Type;

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
class ObjectValue : public Value {
protected:
    ObjectValue() {}

public:
    using ref_t = TypedValueRef<T>;
    using weak_ref_t = TypedValueWeakRef<T>;

    static constexpr bool is_primitive = false;
    static constexpr bool is_object = true;
};

/**
 * \brief base class of values, with mixin support
 *
 * \tparam T type of value
 * \tparam TMixin type of mixin class
 */
template <typename T, typename TMixin>
class PrimitiveValue : public Value, public TMixin {
public:
    using ref_t = TypedValueRef<T>;
    using weak_ref_t = TypedValueWeakRef<T>;

    using TMixin::TMixin;

    PrimitiveValue(TMixin&& mixin) : TMixin(std::move(mixin)) {}
    PrimitiveValue(const TMixin& mixin) : TMixin(mixin) {}

public:
    void clear() override final { ((TMixin&)*this) = {}; }

    bool eq(const TMixin& value) const { return ((const TMixin&)*this) == value; }

    /**
     * \brief helper function for construct a value
     *
     * \tparam TArgs types of arguments
     * \param args arguments
     * \return TypedValueRef<T> reference of value
     */
    template <typename... TArgs>
    static TypedValueRef<T> make(TArgs&&... args) {
        return PrimitiveType<T>::instance.make(std::forward<TArgs&&>(args)...);
    }

    static constexpr bool is_primitive = true;
    static constexpr bool is_object = false;
};

template <typename T>
PrimitiveType<T>::PrimitiveType() : Type<T>(typeid(T).name()) {
    static_assert(std::is_base_of_v<Value, T>);
    static_assert(!std::is_base_of_v<ObjectValue<T>, T>);
}

inline ValueRef::ValueRef(storage_t storage) {
    m_storage = storage;
    m_id = m_storage->m_id;
}

template <typename TValue>
inline const TValue* ValueRef::as(const Type<TValue>& type) const {
    static_assert(std::is_base_of_v<Value, TValue>);
    return static_cast<const TValue*>(as((const IType&)type));
}

template <typename TValue>
inline const TValue& ValueRef::cast(const Type<TValue>& type) const {
    auto* ptr = as<TValue>(type);
    if (mgb_unlikely(!ptr)) {
        on_cast_failure(type);
    }
    return static_cast<const TValue&>(*ptr);
}

template <typename TValue>
inline const TypedValueRef<TValue>& ValueRef::as_ref(const Type<TValue>& type) const {
    if (!is(type)) {
        return TypedValueRef<TValue>::nil;
    }
    return *reinterpret_cast<const TypedValueRef<TValue>*>(this);
}

template <typename TValue>
inline const TypedValueRef<TValue>& ValueRef::cast_ref(const Type<TValue>& type) const {
    if (!m_storage) {
        return TypedValueRef<TValue>::nil;
    }
    if (mgb_unlikely(!is(type))) {
        on_cast_failure(type);
    }
    return *reinterpret_cast<const TypedValueRef<TValue>*>(this);
}

inline void ValueRef::on_cast_failure(const IType& type) const {
    // if this is ErrorValue, rethrow directly
    storage()->try_rethrow();
    mgb_assert(
            storage()->type() != type, "expect type %s, got %s", type.name().c_str(),
            to_string().c_str());
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
        mgb_assert(m_storage, "empty storage");
        return static_cast<const T&>(*m_storage);
    }
    const T* operator->() const { return static_cast<const T*>(m_storage.get()); }

    /**
     * \brief reset underlying value to another value
     *
     * \param successor new value
     */
    inline void reset(ValueRef successor) {
        static_assert(std::is_base_of_v<ObjectValue<T>, T>);
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
    friend class Type<T>;
    friend class TypedValueWeakRef<T>;
};

template <typename T>
class TypedValueWeakRef : public ValueWeakRef {
private:
    TypedValueWeakRef(const ValueRef& value) : ValueWeakRef(value) {}
    TypedValueWeakRef(const ValueWeakRef& value) : ValueWeakRef(value) {}

public:
    TypedValueWeakRef(const TypedValueRef<T>& value) : ValueWeakRef(value) {}
    TypedValueRef<T> lock() { return (TypedValueRef<T>)ValueWeakRef::lock(); }
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

template <typename T>
template <typename... TArgs>
TypedValueRef<T> Type<T>::make(TArgs&&... args) const {
    static_assert(std::is_final_v<T>);
    auto storage = LocalPtr<Value>::make<T>(std::forward<TArgs&&>(args)...);
    storage->m_type = this;
    return ValueRef::make(std::move(storage));
}

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
