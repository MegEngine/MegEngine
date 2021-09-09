/**
 * \file src/opr/impl/custom_opnode.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/custom_opnode.h"

namespace mgb {
namespace opr {

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CustomOpNode);

void CustomOpNode::infer_output_comp_node(void) {
    SmallVector<CompNode> input_comp_nodes(input_num());
    for (size_t i=0; i<input_num(); ++i) {
        input_comp_nodes[i] = input(i)->comp_node();
    }

    SmallVector<CompNode> output_comp_nodes = custom::to_builtin<CompNode, custom::Device>(
        m_op->infer_output_device(
            custom::to_custom<CompNode, custom::Device>(input_comp_nodes), m_param
        )
    );
    
    for (size_t i=0; i<output_num(); ++i) {
        mgb_assert(output_comp_nodes[i] == output_comp_nodes[0], 
                  "only single comp node operator is supported");
        output(i)->comp_node(output_comp_nodes[i]);
    }

    m_comp_node = output_comp_nodes[0];
}

void CustomOpNode::infer_output_dtype(void) {
    SmallVector<DType> input_dtypes(input_num());
    for (size_t i=0; i<input_num(); ++i) {
        input_dtypes[i] = input(i)->dtype();
    }

    SmallVector<DType> output_dtypes = custom::to_builtin<megdnn::DType, custom::DType>(
        m_op->infer_output_dtype(
            custom::to_custom<megdnn::DType, custom::DType>(input_dtypes), m_param
        )
    );

    for (size_t i=0; i<output_num(); ++i) {
        output(i)->dtype(output_dtypes[i]);
    }
}

void CustomOpNode::infer_output_format(void) {
    SmallVector<TensorFormat> input_formats(input_num());
    for (size_t i=0; i<input_num(); ++i) {
        input_formats[i] = input(i)->format();
    }

    SmallVector<TensorFormat> output_formats = custom::to_builtin<TensorFormat, custom::Format>(
        m_op->infer_output_format(
            custom::to_custom<TensorFormat, custom::Format>(input_formats), m_param
        )
    );

    for (size_t i=0; i<output_num(); ++i) {
        output(i)->format(output_formats[i]);
    }
}

void CustomOpNode::infer_output_shape(void) {
    SmallVector<TensorShape> input_shapes(input_num());
    for (size_t i=0; i<input_num(); ++i) {
        input_shapes[i] = input(i)->shape();
    }

    SmallVector<TensorShape> output_shapes = custom::to_builtin<TensorShape, custom::Shape>(
        m_op->infer_output_shape(
            custom::to_custom<TensorShape, custom::Shape>(input_shapes), m_param
        )
    );

    for (size_t i=0; i<output_num(); ++i) {
        output(i)->shape(output_shapes[i]);
    }
}

void CustomOpNode::infer_output_shape(const TensorShapeArray &input_shapes,
                                      TensorShapeArray &output_shapes) {
    output_shapes = custom::to_builtin<TensorShape, custom::Shape>(
        m_op->infer_output_shape(
            custom::to_custom<TensorShape, custom::Shape>(input_shapes), m_param
        )
    );
}

// called by computing_graph for each output varnode
bool CustomOpNode::infer_desc(size_t out_idx, TensorShape &output_shape,
                              const StaticInferInpVal &input_vals) {
    TensorShapeArray input_shapes(input_vals.val.size());
    TensorShapeArray output_shapes(output_num());

    for (size_t i = 0; i < input_shapes.size(); ++ i) {
        input_shapes[i] = input_vals.val[i].shape();
    }

    infer_output_shape(input_shapes, output_shapes);
    output_shape = output_shapes.at(out_idx);
    return true;
}

void CustomOpNode::init_output_dtype() {
    infer_output_dtype();
}

void CustomOpNode::init_output_format() {
    infer_output_format();
}

void CustomOpNode::init_output_comp_node() {
    infer_output_comp_node();
}

void CustomOpNode::do_execute(ExecEnv &env) {
    auto runner = [this]() {
        this->owner_graph()->event().signal_inplace<cg::event::BeforeKernel>(
            this, m_comp_node
        );
        m_comp_node.activate();

        SmallVector<DeviceTensorND> inputs, outputs;
        for(size_t i=0; i<input_num(); i++)
            inputs.push_back(input(i)->dev_tensor());
        for(size_t i=0; i<output_num(); i++)
            outputs.push_back(output(i)->dev_tensor());

        std::vector<custom::Tensor> custom_inputs = custom::to_custom<DeviceTensorND, custom::Tensor>(inputs);
        std::vector<custom::Tensor> custom_outputs = custom::to_custom<DeviceTensorND, custom::Tensor>(outputs);
        m_op->compute(custom_inputs, m_param, custom_outputs);
        // [TODO] sync should be modified
        CompNode::sync_all();

        this->owner_graph()->event().signal_inplace<cg::event::AfterKernel>(
            this, m_comp_node
        );
    };
    env.dispatch_on_comp_node(m_comp_node, runner);
}

void CustomOpNode::init_output_static_infer_desc() {
    using namespace std::placeholders;
    using namespace cg::static_infer;

    m_out_shape.resize(output_num());
    auto &&mgr = owner_graph()->static_infer_manager();

    DepVal dep;
    // [TODO] need design a interface to allow user to decide it
    if (true) {     
        for (auto input_var: input())                       
            dep.push_back({input_var, DepType::SHAPE});
    }
    else {
        for (auto input_var: input())                       
            dep.push_back({input_var, DepType::VALUE});
    }

    for (size_t i = 0; i < output_num(); ++ i) {
        mgr.register_shape_infer(output(i), {
            dep.empty() ? SourceType::CONSTANT : SourceType::DEP, dep,
            std::bind(&CustomOpNode::infer_desc, this, i, _1, _2)
        });
    }
}

void CustomOpNode::init_output_mem_plan(bool dynamic) {
    for (auto output_var: output()) {
        if (cg::is_static_var_storage(output_var) == !dynamic 
                && !output_var->contain_flag(VarNode::Flag::NO_SYS_MEM_ALLOC))
            output_var->init_mem_plan();
    }
}

void CustomOpNode::init_rt_force_dynamic_mem_alloc_imply_chain() {
    
}

void CustomOpNode::add_input_layout_constraint() {         
    for (auto &&input_var: input()) {
        input_var->add_layout_constraint_contiguous();
    }
}

void CustomOpNode::mem_plan_fwd_in2out_readonly() {        

}

void CustomOpNode::mem_plan_fwd_in2out_writable() {        

}

cg::OperatorNodeBase::OprEventCallback CustomOpNode::get_opr_event_callback() {  
    return {};
}

void CustomOpNode::on_output_comp_node_stream_changed() {  
    for (auto output_var: output()) {
        if (output_var->comp_node() != m_comp_node) {
            mgb_assert(output_var->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
            output_var->comp_node(m_comp_node);
        }
    }
}

cg::OperatorNodeBase::NodeProp* CustomOpNode::do_make_node_prop() const {    
    return OperatorNodeBase::do_make_node_prop();
}

bool CustomOpNode::update_priority() const {
    if (output_num() == 1 
            && output()[0]->contain_flag(VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
        node_prop().attribute().priority 
            = std::numeric_limits<decltype(NodeProp::Attribute::priority)>::min();
        return true;
    }
    return false;
}

CustomOpNode::CustomOpNode(const std::shared_ptr<const custom::CustomOp> &op,
                           VarNodeArray inputs,
                           const custom::Param &param,
                           const OperatorNodeConfig &config):
        OperatorNodeBase(inputs[0]->owner_graph(), config, op->op_type(), inputs), m_op(op), m_param(param) {
    mgb_assert(input_num() == inputs.size(), "wrong input tensors list length");
    for (size_t i=0; i < input_num(); ++i) 
        add_input({inputs[i]});

    for (size_t i=0; i<output_num(); ++i) 
        add_output(output_info(i).name());
    
    if (!std::is_empty<custom::Param>::value) {
        using step = unsigned long;
        size_t STEP_SIZE = sizeof(step);
        std::string hash_str = std::to_string(op->runtime_id());
        for (auto &&val: param.raw()) {
            hash_str += val.first;
            hash_str += val.second.str();
        }
        if (hash_str.size() % STEP_SIZE != 0)
            hash_str += std::string(STEP_SIZE - (hash_str.size() % STEP_SIZE), ' ');
        for (size_t pos=0; pos <hash_str.size(); pos+=STEP_SIZE)
            add_equivalence_component<PODHash<step>>(reinterpret_cast<const step*>(hash_str.c_str()+pos));
    }
}

VarNodeArray CustomOpNode::make(const std::shared_ptr<const custom::CustomOp> &op,
                                VarNodeArray inputs,
                                const custom::Param &param,
                                const OperatorNodeConfig &config) {
    auto &&outputs = inputs[0]->owner_graph()->insert_opr(
        std::make_unique<CustomOpNode>(op, inputs, param, config))->output();
    return outputs;
}

SymbolVarArray CustomOpNode::make(const std::shared_ptr<const custom::CustomOp> &op,
                                  SymbolVarArray inputs,
                                  const custom::Param &param,
                                  const OperatorNodeConfig &config) {
    VarNodeArray input_vars(inputs.size());
    for (size_t i=0; i<input_vars.size(); ++i)
        input_vars[i] = inputs[i].node();

    auto &&outputs = inputs[0].node()->owner_graph()->insert_opr(
        std::make_unique<CustomOpNode>(op, input_vars, param, config))->output();
    SymbolVarArray ret(outputs.size());
    for (size_t i=0; i<ret.size(); ++i)
        ret[i] = outputs[i];
    return ret;
}

custom::RunTimeId CustomOpNode::runtime_id() const {
    return m_op->runtime_id();
}

uint32_t CustomOpNode::param_tag(void) const {
    return m_op->param_info().tag();
}

custom::Param& CustomOpNode::param(void) {
    return m_param;
}

custom::Param CustomOpNode::param(void) const {
    return m_param;
}

// a series of functions with the same names as CustomOpImpl
std::string CustomOpNode::op_type(void) const {
    return m_op->op_type();
}

std::string CustomOpNode::op_desc(void) const {
    return m_op->op_desc();
}

size_t CustomOpNode::input_num(void) const {
    return m_op->input_num();
}

size_t CustomOpNode::output_num(void) const {
    return m_op->output_num();
}

custom::ArgInfo CustomOpNode::input_info(size_t idx) const {
    return m_op->input_info(idx);
}

custom::ArgInfo CustomOpNode::output_info(size_t idx) const {
    return m_op->output_info(idx);
}

}
}
