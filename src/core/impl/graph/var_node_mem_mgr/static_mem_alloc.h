/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#include <cstddef>
#include <memory>

namespace mgb {
namespace cg {

class VarNode;

/*!
 * \brief memory allocator for known full sequence
 */
class StaticMemAlloc {
    public:
        using UserKeyType = const void*;

        enum class AllocatorAlgo {
            //! O(n^2) allocator that works by move conflicting intervals
            //! higher; good performance
            INTERVAL_MOVE,

            //! O(n log n) best-fit allocator
            BEST_FIT,

            //! O(n log n) allocator with better performance
            PUSHDOWN,
        };

        static std::unique_ptr<StaticMemAlloc> make(AllocatorAlgo algo);

        virtual ~StaticMemAlloc() = default;

        /*!
         * \brief add a memory alloc request, which is used during time interval
         *      [begin, end)
         * \return interval id
         */
        virtual size_t add(
                size_t begin, size_t end, size_t size, UserKeyType key) = 0;

        /*!
         * \brief add an overwrite spec: *iid_src* could overwrite *iid_dest*
         */
        virtual StaticMemAlloc& add_overwrite_spec(
                size_t iid_src, size_t iid_dest, size_t offset) = 0;

        /*!
         * \brief solve allocation scheme after add() and add_overwrite_spec()
         *      has been called
         */
        virtual StaticMemAlloc& solve() = 0;

        /*!
         * \brief get peak memory usage
         */
        virtual size_t tot_alloc() const = 0;

        /*!
         * \brief get lower bound (not necessarily achievable) of peak memory
         *      usage
         */
        virtual size_t tot_alloc_lower_bound() const = 0;

        /*!
         * \brief get allocated address for an interval
         */
        virtual size_t get_start_addr(UserKeyType key) const = 0;

        /*!
         * \brief set memory address alignment (except for overwritters)
         *
         * Must be called before calling solve()
         *
         * \param alignment address alignment, must be power of 2
         */
        virtual StaticMemAlloc& alignment(size_t alignment) = 0;

        /*!
         * \brief set interval padding at the end(except for overwritters)
         *
         * Must be called before calling add()
         *
         * \param padding interval padding
         */
        virtual StaticMemAlloc& padding(size_t padding) = 0;

#if MGB_ENABLE_DEBUG_UTIL
        //! set by the caller to convert key to VarNode* for debug logging
        VarNode* (*dbg_key2varnode)(UserKeyType) = nullptr;
#endif
};

} // cg
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

