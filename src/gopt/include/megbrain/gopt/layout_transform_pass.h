#pragma once
#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/inference.h"

namespace mgb {
namespace gopt {

class LayoutTransformContext;
class SolverBase;

/*!
 * \brief A layout transform pass, which convert the operator's format to the
 * optimal format using the results of the solver.
 */
class LayoutTransformPass final : public Pass {
public:
    const char* name() const override { return "layout assignment pass"; }
    void apply(OptState& opt) const override;
    LayoutTransformPass(
            std::unique_ptr<LayoutTransformContext> ctx,
            std::unique_ptr<SolverBase> solver)
            : m_ctx{std::move(ctx)}, m_solver{std::move(solver)} {}
    static std::unique_ptr<LayoutTransformPass> make(GraphTuningOptions::Target target);

private:
    std::unique_ptr<LayoutTransformContext> m_ctx;
    std::unique_ptr<SolverBase> m_solver;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
