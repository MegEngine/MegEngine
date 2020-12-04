/**
 * \file imperative/src/impl/ops/cond_take.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/misc.h"
#include "../dnn_op_helper.h"
#include "../op_trait.h"

using namespace megdnn;

namespace mgb::imperative {

namespace {

class MegDNNDynOutMallocImpl final: public megdnn::DynOutMallocPolicy {
    using Output = std::array<TensorPtr, 2>;

    CompNode m_cn;
    Output m_out;

    public:
        MegDNNDynOutMallocImpl(CompNode cn): m_cn{cn} {}

        megdnn::TensorND alloc_output(
                size_t id, DType dtype, const TensorShape &shape,
                void *user_data) override;

        void* alloc_workspace(size_t sz, void *user_data) override;
        void free_workspace(void *ptr, void *user_data) override;
        TensorPtr at(size_t id);
};

megdnn::TensorND MegDNNDynOutMallocImpl::alloc_output(
        size_t id, DType dtype, const TensorShape &shape,
        void * /*user_data*/) {
    TensorLayout m_layout(shape, dtype);
    m_out[id] = Tensor::make(m_layout, m_cn);
    return m_out[id]->dev_tensor().as_megdnn();
}

void* MegDNNDynOutMallocImpl::alloc_workspace(size_t sz, void * /*user_data*/) {
    return m_cn.alloc_device(sz);
}

void MegDNNDynOutMallocImpl::free_workspace(void *ptr, void * /*user_data*/) {
    m_cn.free_device(ptr);
}

TensorPtr MegDNNDynOutMallocImpl::at(size_t id) {
    return m_out[id];
}

cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    def.cast_final_safe<CondTake>();
    auto&& graph = inputs[0]->owner_graph();

    opr::CondTake::Param param;
    param.val = 1;
    cg::OperatorNodeConfig config;
    cg::OperatorNodeBase* opr = graph->insert_opr(
            std::make_unique<opr::CondTake>(
                    inputs[0], inputs[1], param, config));
    return opr;
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto&& opr = def.cast_final_safe<CondTake>();
    mgb_assert(opr.same_type<CondTake>());
    mgb_assert(inputs.size() == 2, "CondTake take 2 inputs, got %lu",
               inputs.size());

    auto&& inp = inputs[0];
    auto&& msk = inputs[1];
    mgb_assert(inp->layout().eq_shape(msk->layout()),
               "input shape does not match mask shape");
    mgb_assert(msk->get_value().dtype().enumv() == DTypeEnum::Bool,
               "mask dtype must be bool");
    DnnOprCaller<megdnn::CondTake> dnn_op(inp->comp_node());
    dnn_op.op->param().val = 1;

    TensorLayout m_layout({dnn_op.op->get_workspace_in_bytes(inp->layout())},
                           dtype::Byte());

    auto dnn_workspace = dnn_op.create_workspace(m_layout);
    MegDNNDynOutMallocImpl policy{inp->comp_node()};

    dnn_op.op->exec(inp->dev_tensor().as_megdnn(),
                  msk->dev_tensor().as_megdnn(),
                  dnn_workspace,
                  &policy);

    SmallVector<TensorPtr> out;
    out.push_back(policy.at(0));
    out.push_back(policy.at(1));
    return out;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
    const OpDef& def,
    const SmallVector<LogicalTensorDesc>& inputs) {
    auto cn = inputs[0].comp_node;
    return {{
        {TensorLayout(inputs[0].layout.dtype), cn},
        {TensorLayout(dtype::Int32()), cn}
    }, true};
}

OP_TRAIT_REG(CondTake, CondTake, opr::CondTake)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .fallback();

} // namespace

} // namespace mgb::imperative
