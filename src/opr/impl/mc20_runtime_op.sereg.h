/**
 * \file src/opr/impl/mc20_runtime_op.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain/opr/mc20_runtime_op.h"
#include "megbrain/serialization/sereg.h"

#if MGB_MC20
namespace mgb {
using MC20RuntimeOpr = opr::MC20RuntimeOpr;
namespace serialization {

template <>
struct OprLoadDumpImpl<MC20RuntimeOpr, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<opr::MC20RuntimeOpr>();
        auto&& buf = opr.buffer();
        auto&& name = opr.name();
        ctx.dump_buf_with_len(buf.data(), buf.size());
        ctx.dump_buf_with_len(name.c_str(), name.size());
    }

    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        inputs.at(0)->comp_node().activate();
        auto buf = ctx.load_shared_buf_with_len();
        auto name = ctx.load_shared_buf_with_len();
        std::string c_name(reinterpret_cast<const char*>(name.data()), name.size());
        OperatorNodeConfig& c_config = const_cast<OperatorNodeConfig&>(config);
        c_config.name(c_name);
        return opr::MC20RuntimeOpr::make(
                       std::move(buf), cg::to_symbol_var_array(inputs), c_config)
                .at(0)
                .node()
                ->owner_opr();
    }
};

}  // namespace serialization

namespace opr {
cg::OperatorNodeBase* opr_shallow_copy_mc20_runtime_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    MGB_MARK_USED_VAR(ctx);
    auto&& opr = opr_.cast_final_safe<MC20RuntimeOpr>();
    return MC20RuntimeOpr::make(
                   opr.buffer(), opr.model_handle(), cg::to_symbol_var_array(inputs),
                   config)
            .at(0)
            .node()
            ->owner_opr();
}

MGB_SEREG_OPR(MC20RuntimeOpr, 0);
MGB_REG_OPR_SHALLOW_COPY(MC20RuntimeOpr, opr_shallow_copy_mc20_runtime_opr);
}  // namespace opr
}  // namespace mgb

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
