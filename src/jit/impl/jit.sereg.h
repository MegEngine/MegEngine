/**
 * \file src/jit/impl/jit.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/jit/executor_opr.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace jit {
cg::OperatorNodeBase* opr_shallow_copy_jit_executor_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<JITExecutor>();
    return JITExecutor::make(opr.internal_graph_ptr(), inputs, config)
            .node()
            ->owner_opr();
}

MGB_REG_OPR_SHALLOW_COPY(JITExecutor, opr_shallow_copy_jit_executor_opr);
}  // namespace jit
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
