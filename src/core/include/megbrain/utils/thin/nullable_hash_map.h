/**
 * \file src/core/include/megbrain/utils/thin/nullable_hash_map.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/utils/thin/hash_table.h"
#include "megbrain/utils/metahelper.h"

namespace mgb {

    /*!
     * \brief a hash map whose value can be NULL
     *
     * The values are stored in a memory pool. This container is mainly used for
     * reducing runtime memory usage.
     */
    template<typename Key, typename Val>
    class NullableHashMap: public NonCopyableObj {
        ThinHashMap<Key, Val*> m_map;
        MemPool<Val> m_val_pool;

        public:
            ~NullableHashMap() noexcept {
                clear();
            }

            using UniquePtr = typename MemPool<Val>::UniquePtr;

            //! get an item; return NULL if it does not exist
            Val* get(const Key &key) {
                auto iter = m_map.find(key);
                return iter == m_map.end() ? nullptr : iter->second;
            }

            //! set an item using allocated value
            Val* set(const Key &key, UniquePtr ptr) {
                auto &&item = m_map[key];
                if (item) {
                    m_val_pool.free(item);
                }
                item = ptr.release();
                return item;
            }

            //! allocate a value
            template<typename...Args>
            UniquePtr alloc(Args&&...args) {
                return m_val_pool.alloc_unique(std::forward<Args>(args)...);
            }

            void clear() noexcept {
                for (auto &&i: m_map) {
                    Val *p = i.second;
                    if (p)
                        p->~Val();
                }
                m_map.clear();
                m_val_pool.clear();
            }
    };

} // namespace mgb


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
