/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/pushdown.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./impl.h"

namespace mgb {
namespace cg {

class StaticMemAllocPushdown final: public StaticMemAllocImplHelper {
    class BestfitPrealloc;

    size_t m_peak_usage = 0;

    /*!
     * intervals that lie directly below this interval; address of each interval
     * is max end address of those in below. Indexed by interval ID
     */
    std::vector<IntervalPtrArray> m_interval_below;

    /*!
     * \brief compute topology order of inervals; result represented in
     *      m_interval_below
     */
    void init_topo_order();

    size_t get_interval_addr_end(Interval *interval);

    public:

        void do_solve() override;

        size_t tot_alloc() const override {
            return m_peak_usage;
        }
};

} // cg
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

