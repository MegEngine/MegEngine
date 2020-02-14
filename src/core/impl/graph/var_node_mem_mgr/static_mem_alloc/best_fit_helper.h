/**
 * \file src/core/impl/graph/var_node_mem_mgr/static_mem_alloc/best_fit_helper.h
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
#include "megbrain/utils/thin/function.h"

namespace mgb {
namespace cg {

struct BestFitHelper {
    using Interval = StaticMemAllocImplHelper::Interval;
    thin_function<void(Interval*)> alloc;
    thin_function<void(Interval *dest, size_t offset, Interval*)>
        alloc_overwrite;
    thin_function<void(Interval*)> free;

    /*!
     * \brief run on intervals and call corresponding methods
     */
    void run(const StaticMemAllocImplHelper::IntervalPtrArray &intervals);
};

} // cg
} // mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

