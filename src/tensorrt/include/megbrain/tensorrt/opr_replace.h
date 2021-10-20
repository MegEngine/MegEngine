/**
 * \file src/tensorrt/include/megbrain/tensorrt/opr_replace.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/gopt/framework.h"

#if MGB_ENABLE_TENSOR_RT

#include "megbrain/tensorrt/tensorrt_opr.h"

namespace mgb {
namespace gopt {

class TensorRTReplacePass final : public Pass {
    class Impl;

public:
    const char* name() const override;
    void apply(OptState& opt) const override;
};

}  // namespace gopt

namespace tensorrt {

void transform_dest_vars_inplace(
        mgb::cg::VarNodeArray& dest_vars, cg::GraphCommonOptimizeOptions& options);
}

}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
