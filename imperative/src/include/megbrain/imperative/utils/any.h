#pragma once

#include <typeindex>

#include "megbrain/imperative/utils/local_ptr.h"

namespace mgb::imperative {

class AnyMixinBase {
private:
    const std::type_info* m_type = nullptr;

public:
    AnyMixinBase() = default;

    const std::type_info& type() const { return *m_type; }

    friend class AnyPtr;
};

template <typename T>
class AnyMixin : public AnyMixinBase, public T {
public:
    AnyMixin(T&& val) : T(std::move(val)) {}
};

class AnyPtr {
public:
    using storage_t = LocalPtr<AnyMixinBase>;

private:
    storage_t m_storage;

public:
    const std::type_info& type() const { return m_storage->type(); }
    template <typename T>
    const T& cast() const {
        mgb_assert(is_exactly<T>(), "type mismatch");
        return *static_cast<const AnyMixin<T>*>(m_storage.get());
    }
    template <typename T>
    bool is_exactly() const {
        return std::type_index{typeid(T)} == std::type_index{type()};
    }
    bool operator==(std::nullptr_t nptr) const { return m_storage == nullptr; }
    bool operator!=(std::nullptr_t nptr) const { return m_storage != nullptr; }
    operator bool() const { return m_storage != nullptr; }

    template <typename T, typename... TArgs>
    static AnyPtr make(TArgs&&... args) {
        AnyPtr ret;
        ret.m_storage = LocalPtr<AnyMixinBase>::make<AnyMixin<T>>(
                std::forward<TArgs&&>(args)...);
        ret.m_storage->m_type = &typeid(T);
        return ret;
    }
};

}  // namespace mgb::imperative
