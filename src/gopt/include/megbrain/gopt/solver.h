#pragma once
#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/layout_transform_context.h"
#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/plugin/opr_footprint.h"

namespace mgb {
namespace gopt {

class ProfilerBase;

/*!
 * \brief abstract solver
 */
class SolverBase {
public:
    using OprFormat = Problem::OprFormat;
    using OprFormatConfigID = Problem::OprFormatConfigID;
    using Solution = ThinHashMap<cg::OperatorNodeBase*, OprFormatConfigID>;
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
    using ProblemFilter = thin_function<bool(const Problem&)>;
    ProfilingBasedSolver(std::unique_ptr<ProfilerBase> profiler);
    /*!
     * \note some graph partition (for example, graph partition without format
     * aware operators like conv, deconv, warp, resize etc.) will be filtered by
     * the ProblemFilter, which can reduce the profiling time. */
    ProfilingBasedSolver(
            std::unique_ptr<ProfilerBase> profiler, ProblemFilter problem_filter)
            : m_profiler{std::move(profiler)},
              m_problem_filter{std::move(problem_filter)} {}
    virtual ~ProfilingBasedSolver() = default;
    Solution solve(const Problem& problem) const override;
    virtual Solution do_solve(const Problem& problem) const = 0;

protected:
    std::unique_ptr<ProfilerBase> m_profiler;

private:
    ProblemFilter m_problem_filter;
};

/*!
 * \brief A solver that solves the layout selection problem using dynamic
 * programming algorithm (Markov decision process).
 */
class DynamicProgrammingSolver final : public ProfilingBasedSolver {
public:
    DynamicProgrammingSolver(std::unique_ptr<ProfilerBase> profiler)
            : ProfilingBasedSolver(std::move(profiler)){};
    DynamicProgrammingSolver(
            std::unique_ptr<ProfilerBase> profiler, ProblemFilter problem_filter)
            : ProfilingBasedSolver(std::move(profiler), std::move(problem_filter)){};
    ~DynamicProgrammingSolver() noexcept = default;
    Solution do_solve(const Problem& problem) const override;
    bool can_solve(const Problem& problem) const override;

private:
    class Impl;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
