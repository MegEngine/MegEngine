/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/interval_move.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl.h"
#include "./interval_move.h"

#if !MGB_BUILD_SLIM_SERVING

#include <algorithm>
#include <cstring>

using namespace mgb;
using namespace mgb::cg;

void StaticMemAllocIntervalMove::do_solve() {
    m_peak = 0;
    m_interval_extra.resize(m_interval.size());

    // extend overwritten intervals and align sizes
    for (auto i: m_interval) {
        if (i->is_overwrite_root())
            i->size = align(i->size);
        else
            update_max(i->overwrite_dest_root()->time_end, i->time_end);
    }

    sort_intervals();

    IntervalPtrArray conflict;
    for (size_t cur_idx = 0; cur_idx < m_interval.size(); ++ cur_idx) {
        Interval* cur = m_interval[cur_idx];

        if (!cur->is_overwrite_root()) {
            continue;
        }

        conflict.clear();
        for (size_t i = 0; i < cur_idx; i ++)
            if (m_interval[i]->is_overwrite_root() &&
                    m_interval[i]->time_overlap(*cur)) {
                conflict.push_back(m_interval[i]);
            }
        insert_interval(*cur, conflict);

        m_interval_extra[cur->id].move_conflict.clear();
        for (auto i: conflict) {
            mgb_assert(!cur->addr_overlap(*i),
                "detected conflict: cur=[%zu, %zu)@[%zu, %zu) "
                "conflict=[%zu, %zu)@[%zu, %zu)",
                cur->addr_begin, cur->addr_end(), cur->time_begin, cur->time_end,
                i->addr_begin, i->addr_end(), i->time_begin, i->time_end);

            if (i->addr_begin < cur->addr_begin) {
                m_interval_extra[i->id].move_conflict.push_back(cur);
                mgb_assert(i->addr_end() <= cur->addr_begin);
            }
            else {
                m_interval_extra[cur->id].move_conflict.push_back(i);
                mgb_assert(cur->addr_end() <= i->addr_begin);
            }
        }
    }

    for (auto i: m_interval) {
        if (!i->is_overwrite_root()) {
            mgb_assert(i->addr_begin == INVALID);
            i->addr_begin = i->overwrite_dest_root()->addr_begin +
                i->offset_in_overwrite_dest_root();
        }
    }
}

void StaticMemAllocIntervalMove::sort_intervals() {
    auto cmp = [](const Interval *a, const Interval *b) {
        auto t0 = a->time_length(), t1 = b->time_length();
        return (t0 > t1) || (t0 == t1 &&
                (a->time_begin < b->time_begin ||
                 (a->time_begin == b->time_begin && a->size > b->size)));
    };
    std::sort(m_interval.begin(), m_interval.end(), cmp);
}

void StaticMemAllocIntervalMove::insert_interval(
        Interval &dest, const IntervalPtrArray &conflict) {

    if (conflict.empty()) {
        dest.addr_begin = 0;
        update_max(m_peak, dest.addr_end());
        return;
    }

    size_t peak_incr, orig_peak = m_peak;
    std::tie(dest.addr_begin, peak_incr) = find_best_fit(conflict, dest.size);
    auto dest_end = dest.addr_end();
    update_max(m_peak, dest_end);
    for (auto i: conflict) {
        if (i->addr_end() > dest.addr_begin)
            move_interval_higher(i, dest_end);
        mgb_assert(!i->addr_overlap(dest));
    }
    mgb_assert(m_peak == orig_peak + peak_incr);
}

std::pair<size_t, size_t> StaticMemAllocIntervalMove::find_best_fit(
        const IntervalPtrArray &conflict, size_t dest_size) {

    ++ m_move_space_size_version;

    size_t best_fit_peak_add = std::numeric_limits<size_t>::max(),
           best_fit_move = best_fit_peak_add,  // min move for conflicted
           best_fit_space = best_fit_peak_add, // remaining size in free chunk
           best_fit_addr = INVALID;

    /*
     * First minimize peak_add. If it could be zero, minimize space; otherwise
     * miminize move
     */

    auto consider_free_chunk = [&](size_t free_begin, size_t free_end) {
        mgb_assert(free_end >= free_begin);
        size_t free_size = free_end - free_begin;
        if (free_size >= dest_size) {
            size_t remain = free_size - dest_size;
            if (remain < best_fit_space) {
                best_fit_peak_add = best_fit_move = 0;
                best_fit_space = remain;
                best_fit_addr = free_begin;
            }
        } else {
            size_t chunk_end_incr = dest_size - free_size,
                   peak_add = std::max(free_begin + dest_size, m_peak) - m_peak;
            for (auto i: conflict) {
                if (i->addr_end() <= free_begin)
                    continue;
                mgb_assert(free_end <= i->addr_begin);
                size_t max_dist =
                    i->addr_begin - free_end + get_move_space_size(i);
                if (max_dist < chunk_end_incr)
                    update_max(peak_add, chunk_end_incr - max_dist);
            }
            if (peak_add < best_fit_peak_add ||
                    (peak_add == best_fit_peak_add &&
                     chunk_end_incr < best_fit_move)) {
                best_fit_peak_add = peak_add;
                best_fit_move = chunk_end_incr;
                best_fit_addr = free_begin;
            }
        }
    };

    auto merged = merge_interval_by_addr(conflict);
    size_t prev_end = 0;
    for (auto &&i: merged) {
        consider_free_chunk(prev_end, i.begin);
        prev_end = i.end;
    }
    consider_free_chunk(prev_end, m_peak);
    mgb_assert(best_fit_addr != INVALID);

    return {best_fit_addr, best_fit_peak_add};
}

std::vector<StaticMemAllocIntervalMove::MergedInterval>
StaticMemAllocIntervalMove::merge_interval_by_addr(
        const IntervalPtrArray &intervals) {
    std::vector<MergedInterval> result;
    std::vector<std::pair<size_t, size_t>> addrs; // addr_begin, addr_end
    for (auto i: intervals) {
        addrs.emplace_back(i->addr_begin, i->addr_end());
    }
    std::sort(addrs.begin(), addrs.end());

    MergedInterval *current = nullptr;
    for (auto &&i: addrs) {
        if (!current || i.first >= current->end) {
            result.emplace_back();
            current = &result.back();
            current->begin = i.first;
            current->end = i.second;
        } else {
            update_min(current->begin, i.first);
            update_max(current->end, i.second);
        }
    }

    return result;
}

size_t StaticMemAllocIntervalMove::get_move_space_size(Interval *interval) {
    auto &&extra_info = m_interval_extra[interval->id];
    auto &&rec = extra_info.move_space_size;
    if (rec.version == m_move_space_size_version)
        return rec.size;

    auto end = interval->addr_end();
    size_t sz = m_peak - end;
    for (auto i: extra_info.move_conflict) {
        mgb_assert(i->addr_begin >= end);
        size_t psize = get_move_space_size(i) + i->addr_begin - end;
        update_min(sz, psize);
    }

    rec.version = m_move_space_size_version;
    rec.size = sz;
    return sz;
}

void StaticMemAllocIntervalMove::move_interval_higher(
        Interval *interval, size_t prev_end) {
    if (interval->addr_begin >= prev_end)
        return;

    interval->addr_begin = prev_end;
    size_t cur_end = interval->addr_end();
    update_max(m_peak, cur_end);
    for (auto i: m_interval_extra[interval->id].move_conflict) {
        move_interval_higher(i, cur_end);
        mgb_assert(!interval->addr_overlap(*i));
    }
}

#endif // !MGB_BUILD_SLIM_SERVING

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

