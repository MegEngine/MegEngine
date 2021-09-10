/**
 * \file src/gopt/include/megbrain/gopt/global_layout_transformation.h
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
    LayoutTransformPass(std::unique_ptr<LayoutTransformContext> ctx,
                         std::unique_ptr<SolverBase> solver)
            : m_ctx{std::move(ctx)}, m_solver{std::move(solver)} {}

private:
    std::unique_ptr<LayoutTransformContext> m_ctx;
    std::unique_ptr<SolverBase> m_solver;
};

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
