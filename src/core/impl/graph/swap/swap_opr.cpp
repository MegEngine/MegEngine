/**
 * \file src/core/impl/graph/swap/swap_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./swap_opr.h"

#include "megbrain/comp_node_env.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#if MGB_ENABLE_MEMORY_SWAP
using namespace mgb;
using namespace swap::opr;

/* ===================== SwapInMS ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SwapInMS);
SwapInMS::SwapInMS(ComputingGraph& graph, VarNode* swap_out_var,
                   VarNode* dep_var, const Param& param,
                   const OperatorNodeConfig& config)
        : Super{&graph, config, "swap-in-ms", {swap_out_var}},
          m_recorder{param.swap_var_recorder_ptr},
          m_param{param} {
    add_input({swap_out_var});
    add_input({dep_var});
    add_output(None)->dtype(input(0)->dtype());
    output(0)->add_flag(VarNode::Flag::DISALLOW_VAR_SANITY_CHECK);
}

void SwapInMS::scn_do_execute() {
    auto&& od = output(0)->dev_tensor();
    m_recorder->pop_value(input(0)->owner_opr()->id(), od);
}

void SwapInMS::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0),
            ShapeInferDesc::make_identity(input(0)->owner_opr()->input(0)));

    owner_graph()->static_infer_manager().register_value_infer(
            output(0),
            ValueInferDesc::make_identity(input(0)->owner_opr()->input(0)));
}

cg::OperatorNodeBase::NodeProp* SwapInMS::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    return ret;
}

SymbolVar SwapInMS::make(ComputingGraph& graph, SymbolVar inp, SymbolVar dep,
                         const Param& param, const OperatorNodeConfig& config) {
    return graph
            .insert_opr(std::make_unique<SwapInMS>(graph, inp.node(),
                                                   dep.node(), param, config))
            ->output(0);
}

/* ===================== SwapOutMS ===================== */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(SwapOutMS);
SwapOutMS::SwapOutMS(ComputingGraph& graph, VarNode* inp, const Param& param,
                     const OperatorNodeConfig& config)
        : Super{&graph, config, "swap-out-ms", {inp}},
          m_recorder{param.swap_var_recorder_ptr},
          m_param{param} {
    add_input({inp});
    add_output(None);
}

void SwapOutMS::scn_do_execute() {
    auto&& id = input(0)->dev_tensor();
    m_recorder->on_val_produced(this->id(), id);
    // copy things in id to m_host_data
    // assert_tensor_eq(id, *(m_host_data.get()));
}

void SwapOutMS::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    /*!
     * This is a virutal edge with size {1} to SwapInMS, its content is
     * pointless
     */
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), ShapeInferDesc::make_const({1}));
}

cg::OperatorNodeBase::NodeProp* SwapOutMS::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    return ret;
}

SymbolVar SwapOutMS::make(ComputingGraph& graph, const SymbolVar inp,
                          const Param& param,
                          const OperatorNodeConfig& config) {
    return graph
            .insert_opr(std::make_unique<SwapOutMS>(graph, inp.node(), param,
                                                    config))
            ->output(0);
}

/* ===================== WaitSwapInMS ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(WaitSwapInMS);
WaitSwapInMS::WaitSwapInMS(ComputingGraph& graph, const VarNodeArray& inputs,
                           const OperatorNodeConfig& config)
        : Super{&graph, config, "wait-swap-in", inputs} {
    mgb_assert(inputs.size() == 2);
    for (auto x : inputs)
        add_input({x});
    add_output(None)->dtype(input(0)->dtype());
}

void WaitSwapInMS::scn_do_execute() {
    mgb_assert(input(0)->owner_opr()->same_type<SwapInMS>());
    // auto x = input(0)->owner_opr()->cast_final_safe<SwapInMS*>();
    auto x = static_cast<SwapInMS*>(input(0)->owner_opr());
    auto ptr = &(input(0)->dev_tensor());
    mgb_assert(x);
    mgb_assert(ptr);
    x->wait_bucket_copy(ptr);
    mixin_scn_do_execute(*this);
}

void WaitSwapInMS::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), ShapeInferDesc::make_identity(input(0)));
    owner_graph()->static_infer_manager().register_value_infer(
            output(0), ValueInferDesc::make_identity(input(0)));
}

cg::OperatorNodeBase::NodeProp* WaitSwapInMS::do_make_node_prop() const {
    auto prop = Super::do_make_node_prop();
    if (input().size() > 1) {
        SmallVector<NodeProp::DepType> dep_types{NodeProp::DepType::DEV_VALUE};
        for (size_t i = 1; i < input().size(); ++i) {
            dep_types.push_back(NodeProp::DepType::DEV_COMP_ORDER);
        }
        prop->reset_dep_type(input(), dep_types);
    }
    prop->add_flag(
            cg::OperatorNodeBase::NodeProp::Flag::CROSS_COMP_NODE_MEMORY);
    return prop;
}

SymbolVar WaitSwapInMS::make(ComputingGraph& graph,
                             const SymbolVarArray& inputs,
                             const OperatorNodeConfig& config) {
    mgb_assert(inputs.size() == 2);
    auto nodes = to_var_node_array(inputs);
    return inputs[0].insert_single_output_opr<WaitSwapInMS>(graph, nodes,
                                                            config);
}

/* ===================== SwapIn ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SwapIn);
SwapIn::SwapIn(ComputingGraph& graph, const SymbolVarArray inputs,
               const std::shared_ptr<HostTensorND>& host_data,
               const OperatorNodeConfig& config)
        : Super{&graph, config, "swap-in", {inputs[0].node()}},
          m_host_data{host_data} {
    // inputs[0] is the swap_out_var, and the following inputs are
    // dependency vars
    auto out_cn = inputs[0].node()->comp_node();

    if (config.has_comp_node_set())
        out_cn = config.get_single_comp_node();
    mgb_assert(out_cn.valid(), "can not get output comp node");
    add_output(None)->dtype(host_data->dtype());

    add_equivalence_component<ScalarHash<void*>>(host_data.get());

    this->comp_node(out_cn);
    for (auto& x : inputs)
        add_input({x.node()});
}

void SwapIn::scn_do_execute() {
    auto&& od = output(0)->dev_tensor();
    od.copy_from_fixlayout(*m_host_data);
}

void SwapIn::init_output_mem_plan(bool dynamic) {
    Super::init_output_mem_plan(dynamic);
}

void SwapIn::init_output_static_infer_desc() {
    using namespace cg::static_infer;
#if 1
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0),
            ShapeInferDesc::make_identity(input(0)->owner_opr()->input(0)));

    owner_graph()->static_infer_manager().register_value_infer(
            output(0),
            ValueInferDesc::make_identity(input(0)->owner_opr()->input(0)));

#else
    /*!
     * This logic is for a deprecated multi-stream method, which assumes
     * that a virtual_dep is set between SwapIn and SwapOut
     */
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0),
            ShapeInferDesc::make_identity(
                    input(0)->owner_opr()->input(1)->owner_opr()->input(0)));
    owner_graph()->static_infer_manager().register_value_infer(
            output(0),
            ValueInferDesc::make_identity(
                    input(0)->owner_opr()->input(1)->owner_opr()->input(0)));
#endif
}

cg::OperatorNodeBase::NodeProp* SwapIn::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    if (input().size() > 1) {
        SmallVector<NodeProp::DepType> dep_types;
        for (size_t i = 0; i < input().size(); ++i)
            dep_types.push_back(NodeProp::DepType::DEV_COMP_ORDER);
        ret->reset_dep_type(input(), dep_types);
    } else {
        SmallVector<NodeProp::DepType> dep_types{
                NodeProp::DepType::DEV_COMP_ORDER};
        ret->reset_dep_type(input(), dep_types);
    }
    return ret;
}

SymbolVar SwapIn::make(ComputingGraph& graph, const SymbolVarArray inputs,
                       const std::shared_ptr<HostTensorND>& host_data,
                       const OperatorNodeConfig& config) {
    return graph
            .insert_opr(
                    std::make_unique<SwapIn>(graph, inputs, host_data, config))
            ->output(0);
}

/* ===================== SwapOut ===================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SwapOut);
SwapOut::SwapOut(ComputingGraph& graph, VarNode* inp, const Param& param,
                 const OperatorNodeConfig& config)
        : Super{&graph, config, "swap-out", {inp}},
          m_host_data{param.host_tensor_ptr},
          m_param{param} {
    add_input({inp});
    add_output(None);
}

void SwapOut::scn_do_execute() {
    auto&& id = input(0)->dev_tensor();
    (m_host_data.get())->copy_from(id);
}

void SwapOut::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    owner_graph()->static_infer_manager().register_shape_infer(
            output(0), ShapeInferDesc::make_const({1}));
}

cg::OperatorNodeBase::NodeProp* SwapOut::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    return ret;
}

SymbolVar SwapOut::make(ComputingGraph& graph, const SymbolVar inp,
                        const Param& param, const OperatorNodeConfig& config) {
    return graph
            .insert_opr(
                    std::make_unique<SwapOut>(graph, inp.node(), param, config))
            ->output(0);
}

#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
