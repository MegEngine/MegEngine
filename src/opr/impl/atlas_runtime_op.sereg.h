/**
 * \file src/opr/impl/atlas_runtime_opr.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/atlas_runtime_op.h"
#include "megbrain/serialization/sereg.h"

#if MGB_ATLAS
namespace mgb {

namespace serialization {

template <>
struct OprLoadDumpImpl<opr::AtlasRuntimeOpr, 0> {

    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<opr::AtlasRuntimeOpr>();
        auto&& buf = opr.buffer();
        ctx.dump_buf_with_len(buf.data(), buf.size());
    }
    static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                      const cg::VarNodeArray& inputs,
                                      const OperatorNodeConfig& config) {
        inputs.at(0)->comp_node().activate();
        auto buf = ctx.load_shared_buf_with_len();
        return opr::AtlasRuntimeOpr::make(
                       std::move(buf), cg::to_symbol_var_array(inputs),
                       config)
                .at(0)
                .node()
                ->owner_opr();
    }
};
}  // namespace serialization

namespace opr {

cg::OperatorNodeBase* opr_shallow_copy_atlas_runtime_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    MGB_MARK_USED_VAR(ctx);
    auto&& opr = opr_.cast_final_safe<AtlasRuntimeOpr>();
    return AtlasRuntimeOpr::make(opr.buffer(), opr.model(),
                                 cg::to_symbol_var_array(inputs),
                                 config)
            .at(0)
            .node()
            ->owner_opr();
}

MGB_SEREG_OPR(AtlasRuntimeOpr, 0);
MGB_REG_OPR_SHALLOW_COPY(AtlasRuntimeOpr, opr_shallow_copy_atlas_runtime_opr);

}  // namespace opr
}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
