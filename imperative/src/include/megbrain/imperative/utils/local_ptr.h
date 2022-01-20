/**
 * \file imperative/src/include/megbrain/imperative/utils/local_ptr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <optional>

#include "megbrain/imperative/utils/mempool.h"
#include "megbrain/utils/metahelper.h"

#define MGB_FAT_LOCAL_PTR 0

namespace mgb::imperative {

template <typename T>
class LocalPtrStorage : public NonCopyableObj {
private:
    size_t m_ref_count = 0;
    size_t m_weak_count = 0;
    T* m_pointer = nullptr;
    void (*reset)(LocalPtrStorage*) = nullptr;
    void (*free)(LocalPtrStorage*) = nullptr;

    void inc_ref() { m_ref_count++; }

    void dec_ref() {
        m_ref_count--;
        if (m_ref_count == 0) {
            reset(this);
            m_pointer = nullptr;
            reset = nullptr;
            if (m_weak_count == 0) {
                free(this);
                // dead
            }
        }
    }

    void inc_weak_ref() { m_weak_count++; }

    void dec_weak_ref() {
        m_weak_count--;
        if ((m_weak_count + m_ref_count) == 0) {
            free(this);
            // dead
        }
    }

    size_t ref_count() const { return m_ref_count; }

    template <typename U>
    friend class LocalPtr;

    template <typename U>
    friend class LocalWeakPtr;

public:
};

template <typename T, typename TDerived>
class LocalPtrStorgeImpl : public LocalPtrStorage<T> {
private:
    std::optional<TDerived> m_value;
    void* m_pool = nullptr;

    template <typename U>
    friend class LocalPtr;

    template <typename U>
    friend class LocalWeakPtr;
};

template <typename T>
class LocalWeakPtr;

/**
 * \brief thread-unsafe smart pointer
 *
 * \tparam T type of value
 */
template <typename T>
class LocalPtr {
public:
    using storage_t = LocalPtrStorage<T>;
    using pool_t = MemPool<storage_t>;
    using weak_type = LocalWeakPtr<T>;
    using pointer_t = T*;

private:
    storage_t* m_storage = nullptr;

#if MGB_FAT_LOCAL_PTR
    pointer_t m_pointer = nullptr;
#endif

    // (m_storage == nullptr) == (m_pointer == nullptr)

    void emplace(storage_t* ptr) {
        if (ptr) {
            ptr->inc_ref();
            m_storage = ptr;
#if MGB_FAT_LOCAL_PTR
            m_pointer = ptr->m_pointer;
#endif
        }
    }

    LocalPtr(storage_t* ptr) { emplace(ptr); }

public:
    LocalPtr() = default;
    LocalPtr(const LocalPtr& rhs) {
        auto storage = rhs.m_storage;
        if (storage) {
            storage->inc_ref();
        }
        m_storage = storage;
#if MGB_FAT_LOCAL_PTR
        m_pointer = rhs.m_pointer;
#endif
    }
    LocalPtr(LocalPtr&& rhs) {
        std::swap(m_storage, rhs.m_storage);
#if MGB_FAT_LOCAL_PTR
        std::swap(m_pointer, rhs.m_pointer);
#endif
    }
    LocalPtr& operator=(const LocalPtr& rhs) {
        if (this == &rhs) {
            return *this;
        }
        auto storage = rhs.m_storage;
        if (storage) {
            storage->inc_ref();
        }
        if (m_storage) {
            m_storage->dec_ref();
        }
        m_storage = storage;
#if MGB_FAT_LOCAL_PTR
        m_pointer = rhs.m_pointer;
#endif
        return *this;
    }
    LocalPtr& operator=(LocalPtr&& rhs) {
        if (this == &rhs) {
            return *this;
        }
        std::swap(m_storage, rhs.m_storage);
#if MGB_FAT_LOCAL_PTR
        std::swap(m_pointer, rhs.m_pointer);
#endif
        rhs.reset();
        return *this;
    }
    bool operator==(const LocalPtr& rhs) const { return m_storage == rhs.m_storage; }
    bool operator!=(const LocalPtr& rhs) const { return m_storage != rhs.m_storage; }
    size_t hash() const { return reinterpret_cast<uintptr_t>(m_storage); }

    ~LocalPtr() { reset(); }

    /**
     * \brief Construct an instance of TDerived and return an LocalPtr
     *
     * There is an memory pool for each (T, TDerived) pair
     *
     * \tparam TDerived type of concrete instance, should be subclass of T
     * \tparam TArgs
     * \param args constructor arguments
     * \return LocalPtr points to the instance
     */
    template <typename TDerived = T, typename... TArgs>
    static LocalPtr make(TArgs&&... args) {
        static_assert(std::is_base_of_v<T, TDerived>);
        using storage_impl_t = LocalPtrStorgeImpl<T, TDerived>;
        constexpr auto normalize_size = [](size_t size) {
            size_t normalized_size = 64;
            while (normalized_size < size) {
                normalized_size *= 2;
            }
            return normalized_size;
        };
        using raw_storage_t =
                std::aligned_storage_t<normalize_size(sizeof(storage_impl_t))>;
        static_assert(alignof(raw_storage_t) % alignof(storage_impl_t) == 0);
        static_assert(sizeof(raw_storage_t) >= sizeof(storage_impl_t));
        using pool_t = MemPool<raw_storage_t>;
        pool_t& pool = MemPoolUtils<raw_storage_t>::get_thread_local();
        auto* raw_storage = pool.alloc_raw();
        auto* storage = reinterpret_cast<storage_impl_t*>(raw_storage);
        new (storage) storage_impl_t();
        storage->m_value.emplace(std::forward<TArgs&&>(args)...);
        storage->m_pointer = &*storage->m_value;
        storage->reset = [](storage_t* storage) {
            auto* storage_impl = static_cast<storage_impl_t*>(storage);
            storage_impl->m_value.reset();
            storage_impl->m_pointer = nullptr;
        };
        storage->free = [](storage_t* storage_base) {
            auto* storage = static_cast<storage_impl_t*>(storage_base);
            auto* pool = reinterpret_cast<pool_t*>(storage->m_pool);
            storage->m_pool = nullptr;
            storage->~storage_impl_t();
            auto* raw_storage = reinterpret_cast<raw_storage_t*>(storage);
            pool->free_raw(raw_storage);
        };
        storage->m_pool = &pool;
        return {(storage_t*)storage};
    }

    T& operator*() const { return *get(); }

    T* get() const {
#if MGB_FAT_LOCAL_PTR
        return m_pointer;
#else
        return m_storage ? m_storage->m_pointer : nullptr;
#endif
    }

    T* operator->() const { return get(); }

    size_t ref_count() const { return m_storage->m_ref_count; }

    bool unique() const { return ref_count() == 1; }

    void reset() {
        if (m_storage) {
            m_storage->dec_ref();
            m_storage = nullptr;
#if MGB_FAT_LOCAL_PTR
            m_pointer = nullptr;
#endif
        }
    }

    operator bool() const { return bool(m_storage); }
    bool operator==(std::nullptr_t nptr) const { return m_storage == nullptr; }
    bool operator!=(std::nullptr_t nptr) const { return m_storage != nullptr; }

    template <typename U>
    friend class LocalWeakPtr;
};

template <typename T>
class LocalWeakPtr {
public:
    using storage_t = LocalPtrStorage<T>;

private:
    storage_t* m_storage = nullptr;

    void emplace(storage_t* ptr) {
        if (ptr) {
            ptr->inc_weak_ref();
            m_storage = ptr;
        }
    }

public:
    LocalWeakPtr() = default;
    LocalWeakPtr(const LocalPtr<T>& rhs) { emplace(rhs.m_storage); }
    LocalWeakPtr(const LocalWeakPtr& rhs) { (*this) = rhs; }
    LocalWeakPtr(LocalWeakPtr&& rhs) { (*this) = std::move(rhs); }
    LocalWeakPtr& operator=(const LocalWeakPtr& rhs) {
        if (this == &rhs) {
            return *this;
        }
        reset();
        emplace(rhs.m_storage);
        return *this;
    }
    LocalWeakPtr& operator=(LocalWeakPtr&& rhs) {
        if (this == &rhs) {
            return *this;
        }
        std::swap(m_storage, rhs.m_storage);
        rhs.reset();
        return *this;
    }

    ~LocalWeakPtr() { reset(); }

    void reset() {
        if (m_storage) {
            m_storage->dec_weak_ref();
            m_storage = nullptr;
        }
    }

    LocalPtr<T> lock() const {
        if (m_storage && m_storage->m_ref_count) {
            return {m_storage};
        }
        return {};
    }

    bool operator==(const LocalWeakPtr& rhs) const {
        return m_storage == rhs.m_storage;
    }

    bool operator!=(const LocalWeakPtr& rhs) const {
        return m_storage != rhs.m_storage;
    }

    size_t hash() const { return reinterpret_cast<uintptr_t>(m_storage); }
};

template <typename T, typename TDerived, typename... TArgs>
LocalPtr<T> make_local(TArgs&&... args) {
    return LocalPtr<T>::template make<TDerived>(std::forward<TArgs&&>(args)...);
}

}  // namespace mgb::imperative
