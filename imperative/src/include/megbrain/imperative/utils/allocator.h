/**
 * \file imperative/src/include/megbrain/imperative/utils/allocator.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <typeindex>

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

    T* allocate(size_type n) {
        mgb_assert(n == 1);
        return m_pool->alloc(sizeof(T));
    }

    void deallocate(pointer* p, size_type n) {
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

}  // namespace mgb::imperative