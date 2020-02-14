/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/best_fit_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./best_fit_helper.h"
#include <map>

using namespace mgb;
using namespace cg;
using IntervalPtrArray = StaticMemAllocImplHelper::IntervalPtrArray;

void BestFitHelper::run(const IntervalPtrArray &intervals) {

    // time => pair(alloc, free)
    using TimeEvent = std::pair<IntervalPtrArray, IntervalPtrArray>;
    std::map<size_t, TimeEvent> time2event;
    for (auto i: intervals) {
        time2event[i->time_begin].first.push_back(i);
        time2event[i->time_end].second.push_back(i);
    }

    IntervalPtrArray to_overwrite;

    for (auto &&tpair: time2event) {
        // free
        for (auto i: tpair.second.second) {
            // if it is overwritten by others, the interval should already freed
            // in last alloc phase

            if (!i->overwrite_src())
                free(i);
        }

        // alloc
        to_overwrite.clear();
        for (auto i: tpair.second.first) {
            if (i->is_overwrite_root())
                alloc(i);
            else
                to_overwrite.push_back(i);
        }

        // free original interval and alloc for overwrite after normal alloc, to
        // avoid double-alloc of same address
        for (auto i: to_overwrite) {
            Interval *dest = i->overwrite_dest();
            alloc_overwrite(dest, i->offset_in_overwrite_dest(), i);
        }
    }
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

