#pragma once

#include <optional>
#include <typeindex>
#include <vector>

#include "megbrain/utils/mempool.h"
#include "megbrain/utils/metahelper.h"

namespace mgb::imperative {

template <typename T>
class Allocator {
public:
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using value_type = T;
    using size_type = std::size_t;
    using diffenence_type = std::ptrdiff_t;
    using pool_type = MemPoolStorage;

private:
    pool_type* m_pool = nullptr;

public:
    Allocator(pool_type* pool) : m_pool(pool) {}

    pointer allocate(size_type n) {
        mgb_assert(n == 1);
        return m_pool->alloc(sizeof(T));
    }

    void deallocate(pointer p, size_type n) {
        mgb_assert(n == 1);
        m_pool->free(p);
    }

    bool operator==(const Allocator& rhs) const { return m_pool == rhs.m_pool; }

    bool operator!=(const Allocator& rhs) const { return m_pool != rhs.m_pool; }
};

template <typename T>
class ThreadLocalAllocatorAdapter {
public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;

public:
    T* allocate(size_type n) { mgb_assert(false); }

    void deallocate(pointer* p, size_type n) { mgb_assert(false); }

    bool operator==(const ThreadLocalAllocatorAdapter& rhs) const { return true; }

    bool operator!=(const ThreadLocalAllocatorAdapter& rhs) const { return false; }
};

template <typename T>
class ForwardAllocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using pointer = T*;

    static constexpr size_t alignment = alignof(T);
    static constexpr size_t element_offset =
            sizeof(T) +
            ((sizeof(T) % alignment) ? 0 : (alignment - sizeof(T) % alignment));

private:
    struct Block {
        std::unique_ptr<std::byte[]> data;
        size_t size = 0;
        size_t capacity = 0;

        T* allocate(size_type n) {
            static_assert(element_offset > std::max(alignment, sizeof(T)));
            size_t begin = size;
            size_t end = begin + element_offset * n;
            if (end > capacity) {
                return nullptr;
            }
            size = end;
            return reinterpret_cast<T*>(data.get() + begin);
        }

        void reset() { size = 0; }
    };
    std::vector<Block> m_used;
    std::optional<Block> m_current;
    size_t block_size = 16 * 1024 * 1024;
    size_t nr_allocated = 0;

private:
    Block allocate_block() {
        block_size *= 2;
        return Block{std::make_unique<std::byte[]>(block_size), 0, block_size};
    }

public:
    pointer allocate(size_type n) {
        if (!m_current) {
            m_current.emplace(allocate_block());
        }
        pointer pointer = m_current->allocate(n);
        while (pointer == nullptr) {
            m_used.push_back(allocate_block());
            std::swap(m_used.back(), *m_current);
            pointer = m_current->allocate(n);
        }
        nr_allocated++;
        return pointer;
    }

    void deallocate(pointer p, size_type n) {
        mgb_assert(nr_allocated > 0);
        nr_allocated--;
    }

    void clear() {
        if (mgb_likely(m_used.empty())) {
            // fastpath
            if (m_current) {
                m_current->reset();
            }
        } else {
            // trim
            *m_current = allocate_block();
            m_used.clear();
        }
        mgb_assert(nr_allocated == 0);
    }

    bool operator==(const ForwardAllocator& rhs) const { return &rhs == this; }
    bool operator!=(const ForwardAllocator& rhs) const { return &rhs != this; }
};

template <typename T, template <typename> typename TAllocator>
class ProxyAllocator {
public:
    using value_type = T;
    using size_type = typename TAllocator<T>::size_type;
    using pointer = typename TAllocator<T>::pointer;

private:
    TAllocator<T>* m_impl;

public:
    T* allocate(size_type n) { return m_impl->allocate(n); }

    void deallocate(pointer* p, size_type n) { return m_impl->deallocate(p, n); }

    bool operator==(const ProxyAllocator<T, TAllocator>& rhs) const {
        if (m_impl == rhs.m_impl) {
            return true;
        } else if (bool(m_impl) ^ bool(rhs.m_impl)) {
            return false;
        } else {
            return *m_impl == *rhs.m_impl;
        }
    }

    bool operator!=(const ProxyAllocator<T, TAllocator>& rhs) const {
        return !((*this) == rhs);
    }
};

}  // namespace mgb::imperative
