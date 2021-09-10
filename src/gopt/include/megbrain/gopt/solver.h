/**
 * \file src/gopt/include/megbrain/gopt/solver.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/layout_transform_context.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/plugin/opr_footprint.h"
#include "megbrain/gopt/inference.h"

namespace mgb {
namespace gopt {

class ProfilerBase;

/*! 
 * \brief abstract solver 
 */
class SolverBase {
public:
    using OprFormat = Problem::OprFormat;
    using Solution = ThinHashMap<cg::OperatorNodeBase*, OprFormat>;
    SolverBase() = default;
    virtual ~SolverBase() = default;
    /*!
     * \brief solve the given problem
     */
    virtual Solution solve(const Problem& problem) const = 0;
    /*!
     * \brief check whether the given problem can be solved by the
     * algorithm(i.e. solver).
     */
    virtual bool can_solve(const Problem& problem) const = 0;
};

/*!
 * \brief solvers that will first collect the costs of operators in different op
 * format and the costs of layout transform of varnode with a user provided
 * profiler on the target device. This will lead to time consuming. 
 */
class ProfilingBasedSolver : public SolverBase {
public:
    using GraphPartitionFilter =
            thin_function<bool(const GraphPartition& graph_partition)>;
    ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler);
    /*!
     * \note some graph partition (for example, graph partition without format
     * aware operators like conv, deconv, warp, resize etc.) will be filtered by
     * the GraphPartitionFilter, which can reduce the profiling time. */
    ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler,
                         GraphPartitionFilter graph_partition_filter)
            : m_profiler{std::move(profiler)},
              m_graph_partition_filter{std::move(graph_partition_filter)} {}
    virtual ~ProfilingBasedSolver() = default;
    Solution solve(const Problem& problem) const override;
    virtual Solution do_solve(const Problem& problem) const = 0;

protected:
    std::unique_ptr<ProfilerBase> m_profiler;

private:
    GraphPartitionFilter m_graph_partition_filter;
};

/*!
 * \brief A solver that solves the layout selection problem using dynamic
 * programming algorithm (Markov decision process).
 */
class DynamicProgrammingSolver final : public ProfilingBasedSolver {
public:
    DynamicProgrammingSolver(std::unique_ptr<ProfilerBase> profiler)
            : ProfilingBasedSolver(std::move(profiler)){};
    DynamicProgrammingSolver(std::unique_ptr<ProfilerBase> profiler,
                             GraphPartitionFilter graph_partition_filter)
            : ProfilingBasedSolver(std::move(profiler),
                                   std::move(graph_partition_filter)){};
    ~DynamicProgrammingSolver() noexcept = default;
    Solution do_solve(const Problem& problem) const override;
    bool can_solve(const Problem& problem) const override;

private:
    class Impl;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
