/**
 * \file src/cambricon/impl/magicmind_runtime_opr.sereg.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/cambricon/magicmind_runtime_opr.h"
#include "megbrain/serialization/sereg.h"

namespace mgb {
namespace serialization {

template <>
struct OprLoadDumpImpl<opr::MagicMindRuntimeOpr, 0> {
    static void dump(OprDumpContext& ctx, const cg::OperatorNodeBase& opr_) {
        auto&& opr = opr_.cast_final_safe<opr::MagicMindRuntimeOpr>();
        auto&& model = opr.inference_model();
        size_t size = 0;
        MM_CHECK(model->GetSerializedModelSize(&size));
        std::string buf;
        buf.resize(size);
        MM_CHECK(model->SerializeToMemory(
                reinterpret_cast<void*>(buf.data()), buf.size()));
        ctx.dump_buf_with_len(buf.data(), buf.size());
    }
    static cg::OperatorNodeBase* load(
            OprLoadContext& ctx, const cg::VarNodeArray& inputs,
            const OperatorNodeConfig& config) {
        auto buf = ctx.load_shared_buf_with_len();
        return opr::MagicMindRuntimeOpr::make(
                       reinterpret_cast<const void*>(buf.data()), buf.size(),
                       cg::to_symbol_var_array(inputs), config)
                .at(0)
                .node()
                ->owner_opr();
    }
};
}  // namespace serialization

namespace opr {
cg::OperatorNodeBase* opr_shallow_copy_magicmind_runtime_opr(
        const serialization::OprShallowCopyContext& ctx,
        const cg::OperatorNodeBase& opr_, const VarNodeArray& inputs,
        const OperatorNodeConfig& config) {
    auto&& opr = opr_.cast_final_safe<MagicMindRuntimeOpr>();
    return MagicMindRuntimeOpr::make(
                   opr.inference_model(), opr.cambricon_allocator(),
                   cg::to_symbol_var_array(inputs), config)
            .at(0)
            .node()
            ->owner_opr();
}

MGB_SEREG_OPR(MagicMindRuntimeOpr, 0);
MGB_REG_OPR_SHALLOW_COPY(MagicMindRuntimeOpr, opr_shallow_copy_magicmind_runtime_opr);

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
