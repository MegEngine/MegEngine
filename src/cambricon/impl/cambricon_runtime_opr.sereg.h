/**
 * \file src/cambricon/impl/cambricon_runtime_opr.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/cambricon/cambricon_runtime_opr.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {

template <>
struct OprLoadDumpImpl<opr::CambriconRuntimeOpr, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<opr::CambriconRuntimeOpr>();
        auto&& buf = opr.buffer();
        ctx.dump_buf_with_len(buf.data(), buf.size());
        auto&& symbol = opr.symbol();
        ctx.dump_buf_with_len(symbol.data(), symbol.size());
        bool tensor_dim_mutable = opr.is_tensor_dim_mutable();
        ctx.dump_buf_with_len(&tensor_dim_mutable, sizeof(bool));
    }
    static cg::OperatorNodeBase* load(OprLoadContext& ctx,
                                      const cg::VarNodeArray& inputs,
                                      const OperatorNodeConfig& config) {
        inputs.at(0)->comp_node().activate();
        auto buf = ctx.load_shared_buf_with_len();
        auto symbol = ctx.load_buf_with_len();
        auto tensor_dim_mutable_storage = ctx.load_buf_with_len();
        bool tensor_dim_mutable;
        memcpy(&tensor_dim_mutable, tensor_dim_mutable_storage.data(),
               sizeof(bool));
        return opr::CambriconRuntimeOpr::make(std::move(buf), std::move(symbol),
                                              cg::to_symbol_var_array(inputs),
                                              tensor_dim_mutable, config)
                .at(0)
                .node()
                ->owner_opr();
    }
};
}  // namespace serialization

namespace opr {
cg::OperatorNodeBase* opr_shallow_copy_cambricon_runtime_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<CambriconRuntimeOpr>();
    return CambriconRuntimeOpr::make(opr.buffer(), opr.symbol(),
                                     cg::to_symbol_var_array(inputs),
                                     opr.is_tensor_dim_mutable(), config)
            .at(0)
            .node()
            ->owner_opr();
}

MGB_SEREG_OPR(CambriconRuntimeOpr, 0);
MGB_REG_OPR_SHALLOW_COPY(CambriconRuntimeOpr,
                         opr_shallow_copy_cambricon_runtime_opr);
}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}


