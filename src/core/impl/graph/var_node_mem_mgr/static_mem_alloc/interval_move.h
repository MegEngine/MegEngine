/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/interval_move.h
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

class StaticMemAllocIntervalMove final: public StaticMemAllocImplHelper {
    struct MergedInterval {
        size_t begin, end;
    };

    size_t m_peak = 0, m_move_space_size_version = 0;

    struct IntervalExtraInfo {
        //! conflicting intervals if trying to move this interval to higher
        //! address
        IntervalPtrArray move_conflict;

        //! max dist to move without increasing peak
        struct MoveSpaceSizeRecord {
            size_t version = 0, size;
        };
        MoveSpaceSizeRecord move_space_size;
    };

    //! extra info for each interval, indexed by interval id
    std::vector<IntervalExtraInfo> m_interval_extra;

    void sort_intervals();

    /*!
     * \brief get max move distance without increasing peak usage
     * \param from the interval that initiates this query, to avoid infinite
     *      recursion
     */
    size_t get_move_space_size(Interval *interval);

    /*!
     * \brief move interval higher so addr_begin >= prev_end
     * \param from the interval that initiates this action, to avoid infinite
     *      recursion
     */
    void move_interval_higher(Interval *interval, size_t prev_end);

    void insert_interval(Interval &dest, const IntervalPtrArray &conflict);

    std::vector<MergedInterval> merge_interval_by_addr(
            const IntervalPtrArray &intervals);

    /*!
     * \brief find best fit
     *
     * minimize peak_add
     * 1. if dest.size < free_space_size, then minimize remaining space
     * 2. otherwise, minimize move distance
     *
     * \return start address, peak_incr
     */
    std::pair<size_t, size_t> find_best_fit(
            const IntervalPtrArray &conflict, size_t dest_size);

    void do_solve() override;

    public:
        size_t tot_alloc() const override {
            mgb_assert(m_peak);
            return m_peak;
        }
};

}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

