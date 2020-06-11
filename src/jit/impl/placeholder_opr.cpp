/**
 * \file src/jit/impl/placeholder_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/jit/placeholder_opr.h"

#include "megbrain/common.h"
#include "megbrain/graph.h"

#if MGB_JIT

using namespace mgb;
using namespace jit;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(JITPlaceholder);

JITPlaceholder::JITPlaceholder(VarNode* src_var, size_t id, InpType inp_type)
        : Super(src_var->owner_graph(), {}, ssprintf("JITPlaceholder@%zu", id),
                {}),
          m_inp_type{inp_type},
          m_id{id} {
    mgb_assert(src_var->dtype().category() == DTypeCategory::FLOAT ||
                       src_var->dtype().category() == DTypeCategory::INT,
               "JIT can only be applied to float/int operators, got %s",
               src_var->dtype().name());
    add_equivalence_component<ScalarHash<DTypeEnum>>(src_var->dtype().enumv());
    add_equivalence_component<ScalarHash<InpType>>(m_inp_type);
    add_equivalence_component<ScalarHash<size_t>>(m_id);
    if (m_inp_type == InpType::HOST_VALUE_FOR_SHAPE) {
        mgb_assert(src_var->dtype() == dtype::Int32{},
                   "src dtype should be int32 for SHAPE InpType, got %s",
                   src_var->dtype().name());
    }
    add_output(None)->dtype(src_var->dtype());
}

void JITPlaceholder::init_output_comp_node() {
    output(0)->comp_node(CompNode::default_cpu());
}

void JITPlaceholder::scn_do_execute() {
    mgb_throw(InternalError, "JITPlaceholder opr can not be executed");
}

void JITPlaceholder::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    auto infer_shape = [](TensorShape& dst, const InpVal&) {
        // do not infer shape to avoid shape mismatch errors (which may occur in
        // reduce)
        return false;
    };
    mgr.register_shape_infer(output(0), {SourceType::MUTABLE, {}, infer_shape});
}

SymbolVar JITPlaceholder::make(VarNode* src_var, size_t id, InpType inp_type) {
    return src_var->owner_graph()
            ->insert_opr(
                    std::make_unique<JITPlaceholder>(src_var, id, inp_type))
            ->output(0);
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
