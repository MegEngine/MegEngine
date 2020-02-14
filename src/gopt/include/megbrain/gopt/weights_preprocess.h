/**
 * \file src/gopt/include/megbrain/gopt/weights_preprocess.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/gopt/framework.h"

namespace mgb {
namespace gopt {

class WinogradTransformReplacePass final : public Pass {
    class Impl;

public:
    const char* name() const override;
    void apply(OptState& opt) const override;
};

void transform_vars_inplace_with_winograd(mgb::cg::VarNodeArray& dest_vars);

}  // namespace gopt
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
