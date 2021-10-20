/**
 * \file src/gopt/impl/profiling_based_solver.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/gopt/profiler.h"
#include "megbrain/gopt/solver.h"
#include "megbrain/opr/dnn/pooling.h"
#include "megbrain/opr/imgproc.h"

using namespace mgb;
using namespace gopt;
using namespace opr;

/* =================== ProfilingBasedSolverSolver ======================*/
ProfilingBasedSolver::ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler)
        : m_profiler{std::move(profiler)} {
    static const ThinHashSet<Typeinfo*> format_aware_oprs = {
#define cb(_Opr) _Opr::typeinfo()
            cb(Convolution),    cb(ConvBiasForward), cb(ConvolutionBackwardData),
            cb(PoolingForward), cb(WarpPerspective), cb(Resize),
    };

    m_graph_partition_filter = [](const GraphPartition& partition) {
        bool has_format_aware_opr = false;
        for (auto&& opr : partition.all_oprs()) {
            if (!has_format_aware_opr && format_aware_oprs.count(opr->dyn_typeinfo())) {
                has_format_aware_opr = true;
                break;
            }
        }
        return has_format_aware_opr;
    };
}

ProfilingBasedSolver::Solution ProfilingBasedSolver::solve(
        const Problem& problem) const {
    const auto& partition = problem.graph_partition();
    if (!m_graph_partition_filter(partition))
        return Solution{};
    return do_solve(problem);
}

// vim: syntax=cpp.doxygen
