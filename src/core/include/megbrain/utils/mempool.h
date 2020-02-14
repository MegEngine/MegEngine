/**
 * \file src/core/include/megbrain/utils/mempool.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <algorithm>
#include <vector>
#include <memory>
#include <cstdint>

namespace mgb {

    class MemPoolStorage {
        bool m_disable_freelist = false;
        size_t m_cur_buf_pos = 0, m_cur_buf_size_bytes = 0;
        std::vector<std::unique_ptr<uint8_t[]>> m_buf;
        std::vector<void*> m_free;

        public:
            MemPoolStorage() noexcept;
            MemPoolStorage(MemPoolStorage &&rhs) noexcept;
            ~MemPoolStorage() noexcept;
            MemPoolStorage& operator = (MemPoolStorage &&rhs) noexcept;

            void swap(MemPoolStorage &other);

            /*!
             * \brief allocate sotrage for an object of specified size
             * \param elem_size size of the object; it must remain unchanged
             *      during lifespan of this MemPoolStorage
             */
            void *alloc(size_t elem_size);
            void free(void *ptr);
            void reorder_free();

            //! clear all allocated storage
            void clear();

            void disable_freelist() {
                m_disable_freelist = true;
            }
    };

    /*!
     * \brief a memory pool for abundant small objects
     *
     * Note that the memory would not be released and returned to upstream
     * allocator until the mem pool is destructed.
     *
     * The caller must match alloc() and free() calls; no additional check is
     * performed.
     */
    template<typename T>
    class MemPool {

        // use another template so T only needs to be complete when alloc() or
        // free() is called
        template<typename=void>
        struct Const {
            static constexpr size_t
                ELEM_SIZE = ((sizeof(T) - 1) / alignof(T) + 1) * alignof(T);
        };
        MemPoolStorage m_storage;

        public:
            class Deleter {
                MemPool *m_pool = nullptr;
                public:
                    Deleter() = default;

                    Deleter(MemPool *pool):
                        m_pool{pool}
                    {}

                    void operator()(T*ptr) const {
                        m_pool->free(ptr);
                    }
            };
            using UniquePtr = std::unique_ptr<T, Deleter>;

            template<typename...Args>
            T* alloc(Args&&... args) {
                auto ptr = static_cast<T*>(
                        m_storage.alloc(Const<>::ELEM_SIZE));
                new(ptr) T(std::forward<Args>(args)...);
                return ptr;
            }

            template<typename...Args>
            UniquePtr alloc_unique(Args&&... args) {
                auto ptr = alloc(std::forward<Args>(args)...);
                return {ptr, {this}};
            }

            void free(T *ptr) {
                ptr->~T();
                m_storage.free(ptr);
            }

            //! reorder free list for cache friendly in future alloc
            void reorder_free() {
                m_storage.reorder_free();
            }

            //! clear all the storage without calling the destructors
            void clear() {
                m_storage.clear();
            }

            /*!
             * \brief disable free list for memory reuse
             *
             * This is only useful in the destructor of an enclosing object, so
             * no extra memory allocation is needed to hold the released objects
             */
            void disable_freelist() {
                m_storage.disable_freelist();
            }
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

