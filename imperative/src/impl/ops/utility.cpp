/**
 * \file imperative/src/impl/ops/utility.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

 #include <queue>

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/graph_cache.h"
#include "megbrain/imperative/subgraph_detail.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/io.h"
#include "../op_trait.h"

namespace mgb::imperative {

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GenericPyOp);
OP_TRAIT_REG(GenericPyOp, GenericPyOp).fallback();

namespace { namespace fastpathcopy {
    auto apply_on_var_node(
            const OpDef& def,
            const VarNodeArray& inputs) {
        return inputs;
    }

OP_TRAIT_REG(FastpathCopy,FastpathCopy)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
}} // fastpathcopy

namespace  { namespace shape_infer {
auto apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    size_t nr_inputs = inputs.size();
    mgb_assert(nr_inputs > 0, "no inputs for ShapeInfer");
    SmallVector<LogicalTensorDesc> input_descs;
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto input = inputs[i]->get_value();
        TensorLayout layout;
        layout.ndim = input.shape(0);
        for (size_t i = 0; i < layout.ndim; ++i) {
            layout[i] = input.ptr<int32_t>()[i];
        }
        layout.dtype = op.dtypes[i];
        layout.init_contiguous_stride();
        input_descs.push_back({layout, op.devices[i]});
    }
    auto [output_descs, valid] = OpDef::infer_output_attrs_fallible(*op.op, input_descs);
    mgb_assert(valid, "shape inference incomplete");
    SmallVector<TensorPtr> outputs;
    for (auto&& output_desc: output_descs) {
        HostTensorND shape_tensor{output_desc.comp_node, {output_desc.layout.ndim}, dtype::Int32()};
        for (size_t i = 0; i < output_desc.layout.ndim; ++i) {
            shape_tensor.ptr<int32_t>()[i] = output_desc.layout[i];
        }
        auto output = Tensor::make(shape_tensor);
        outputs.push_back(output);
    }
    return outputs;
}
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    size_t nr_inputs = inputs.size();
    VarNodeArray input_values, outputs;
    mgb_assert(nr_inputs > 0, "no inputs for ShapeInfer");
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto input_value = opr::Alloc::make(SymbolVar(inputs[i]), op.dtypes[i], {op.devices[i]});
        input_values.push_back(input_value.node());
    }
    auto output_values = OpDef::apply_on_var_node(*op.op, input_values);
    for (auto&& output_value: output_values) {
        outputs.push_back(opr::GetVarShape::make(output_value).node());
    }
    return outputs;
}

auto infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& input_descs) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    SmallVector<LogicalTensorDesc> input_shape_descs;
    size_t nr_inputs = op.devices.size();
    mgb_assert(op.dtypes.size() == nr_inputs, "number of input devices and dtypes mismatch");
    for (size_t i = 0; i < nr_inputs; ++i) {
        LogicalTensorDesc input_shape_desc;
        input_shape_desc.comp_node = op.devices[i];
        input_shape_desc.layout.ndim = 0;
        input_shape_desc.layout.dtype = op.dtypes[i];
        input_shape_descs.push_back(input_shape_desc);
    }
    auto [output_shape_descs, _] = OpDef::infer_output_attrs_fallible(*op.op, input_shape_descs);
    SmallVector<LogicalTensorDesc> output_descs;
    for (auto&& output_shape_desc: output_shape_descs) {
        LogicalTensorDesc output_desc;
        output_desc.comp_node = output_shape_desc.comp_node;
        output_desc.layout.ndim = 1;
        output_desc.layout.dtype = dtype::Int32();
        output_descs.push_back(output_desc);
    }
    return std::make_tuple(output_descs, false);
}

auto props(const OpDef& def) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    return OpDef::props(*op.op);
}

auto make_name(const OpDef& def) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    MGB_MARK_USED_VAR(op);
    return ssprintf("ShapeInfer[%s]", op.op->make_name().c_str());
}

auto hash(const OpDef& def) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    return op.op->hash();
}

auto is_same_st(const OpDef& def, const OpDef& another) {
    if (!another.same_type<ShapeInfer>()) {
        return false;
    }
    auto& lhs = def.cast_final_safe<ShapeInfer>();
    auto& rhs = another.cast_final_safe<ShapeInfer>();
    if (!lhs.op->is_same(*rhs.op)) {
        return false;
    }
    return std::tie(lhs.devices, lhs.dtypes) ==
           std::tie(rhs.devices, rhs.dtypes);
}

OP_TRAIT_REG(ShapeInfer,ShapeInfer)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .make_name(make_name)
    .props(props)
    .hash(hash)
    .is_same_st(is_same_st)
    .fallback();
}}


MGB_DYN_TYPE_OBJ_FINAL_IMPL(ShapeInfer);

namespace { namespace identity {
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Identity>();
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::Identity::make(inputs[0], config);
}

auto apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    return SmallVector<TensorPtr>{inputs[0]};
}
OP_TRAIT_REG(Identity, Identity)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .fallback();
}} // identity

namespace { namespace subgraph {

EncodedSubgraph make_forward_graph(const OpDef& def, SmallVector<LogicalTensorDesc> inputs) {
    return EncodedSubgraph::make(*def.cast_final_safe<SubgraphOp>().graph);
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, 
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        SmallVector<bool> output_has_grad) {
    auto& op = def.cast_final_safe<SubgraphOp>();
    mgb_assert(output_has_grad.size() == op.output_grad_mask.size());
    for (size_t i = 0; i < output_has_grad.size(); ++i) {
        if (!op.output_grad_mask[i]) {
            output_has_grad[i] = false;
        }
    }
    auto bgraph = subgraph_detail::make_backward_graph(def, inputs, input_requires_grad, output_has_grad);
    return EncodedSubgraph::make_single(
            SubgraphOp::make(op.name + "Grad",
                             std::make_shared<Subgraph>(bgraph.graph)),
            bgraph.input_mask, bgraph.output_mask);
}

std::vector<std::pair<const char*, std::string>> props(const OpDef& def) {
    auto& op = def.cast_final_safe<SubgraphOp>();
    return {
        {"name", op.name},
        {"inputs", mgb::imperative::to_string(op.graph->inputs)},
        {"exprs", mgb::imperative::to_string(op.graph->exprs)},
        {"outputs", mgb::imperative::to_string(op.graph->outputs)},
    };
}

std::string make_name(const OpDef& def) {
    auto& op = def.cast_final_safe<SubgraphOp>();
    if (op.name.empty()) {
        return "SubgraphOp";
    } else {
        return op.name;
    }
}

auto hash(const OpDef& def) {
    auto& op = def.cast_final_safe<SubgraphOp>();
    if (!op.graph_key) {
        return (size_t)reinterpret_cast<uintptr_t>(op.graph.get());
    }
    return op.graph_key->hash();
}

auto is_same_st(const OpDef& def, const OpDef& another) {
    if (!another.same_type<SubgraphOp>()) {
        return false;
    }
    auto& lhs = def.cast_final_safe<SubgraphOp>();
    auto& rhs = another.cast_final_safe<SubgraphOp>();
    auto has_graph_key = bool(lhs.graph_key);
    bool graph_same = false;
    if (has_graph_key) {
        graph_same = rhs.graph_key && lhs.graph_key->is_same(*rhs.graph_key);
    } else {
        graph_same = !rhs.graph_key && lhs.graph.get() == rhs.graph.get();
    }
    return graph_same;
}

OP_TRAIT_REG(SubgraphOp, SubgraphOp)
    .make_forward_graph(make_forward_graph)
    .make_backward_graph(make_backward_graph)
    .props(props)
    .make_name(make_name)
    .hash(hash)
    .is_same_st(is_same_st)
    .fallback();

}}

namespace { namespace compiled_op {

struct DeviceMemoryAllocatorImpl: cg::DeviceMemoryAllocator {
    std::shared_ptr<OpDef> current_op;
    void alloc_static(ComputingGraph* graph, DeviceTensorStorage& dest, size_t size) override {
        mgb_assert(0, "alloc_static is not allowed in CompiledOp");
    }
    void alloc_dynamic(VarNode* var, DeviceTensorStorage& dest, size_t size) override {
        auto comp_node = var->comp_node();
        auto storage = current_op->allocate(comp_node, size);
        dest.reset(comp_node, size, storage);
    }
};

struct ComputingGraphHolder {
    std::shared_ptr<ComputingGraph> graph;
    std::unique_ptr<cg::AsyncExecutable> executable;
    SmallVector<std::shared_ptr<DeviceTensorND>> inputs;
    SmallVector<std::shared_ptr<DeviceTensorND>> outputs;
    std::shared_ptr<DeviceMemoryAllocatorImpl> allocator;
    SmallVector<std::unique_ptr<CompNode::Event>> events;
};

ComputingGraphHolder& get_computing_graph(std::shared_ptr<OpDef> compiled_op, SmallVector<LogicalTensorDesc> descs) {
    using ComputingGraphHolderCache = OpMethResultCache<std::queue<std::unique_ptr<ComputingGraphHolder>>>;
    thread_local ComputingGraphHolderCache cache;
    thread_local size_t nr_cg_holders = 0;
    ComputingGraphHolderCache::key_t cache_key = {compiled_op, descs};
    auto& cg_holder_queue = cache[cache_key];
    std::unique_ptr<ComputingGraphHolder> holder;
    if(!cg_holder_queue.empty()) {
        // pick one
        std::swap(cg_holder_queue.front(), holder);
        // check all events finished
        for (auto&& event: holder->events) {
            if (!event->finished()) {
                bool queue_limited = event->comp_node().contain_flag(CompNode::Flag::QUEUE_LIMITED);
                bool many_graph = cg_holder_queue.size() > 10;
                if (queue_limited || !many_graph) {
                    std::swap(cg_holder_queue.front(), holder);
                    break;
                } else {
                    // graph limit
                    mgb_log_debug("computing graph limit for compiled op exceeded, waiting for prev graph");
                    event->host_wait();
                }
            }
        }
        if (holder) {
            cg_holder_queue.pop();
        }
    }
    if (!holder) {
        // create new computing graph
        holder = std::make_unique<ComputingGraphHolder>();
        auto& cg_holder = *holder;
        cg_holder.allocator = std::make_shared<DeviceMemoryAllocatorImpl>();
        cg_holder.graph = ComputingGraph::make();
        cg_holder.graph->options().force_dynamic_alloc = true;
        cg_holder.graph->options().async_exec_level = 0;
        cg_holder.graph->options().graph_opt_level = compiled_op->cast_final_safe<CompiledOp>().gopt_level;
        cg_holder.graph->options().enable_var_mem_defragment = false;
        cg_holder.graph->options().comp_seq_sync_device = false;
        // set allocator for DTR support
        cg_holder.graph->set_device_memory_allocator(cg_holder.allocator);
        VarNodeArray input_vars;
        for (auto&& desc: descs) {
            auto input_device_nd = std::make_shared<DeviceTensorND>();
            input_device_nd->dtype(desc.layout.dtype);
            input_device_nd->comp_node(desc.comp_node);
            input_device_nd->resize(desc.layout);
            cg_holder.inputs.push_back(input_device_nd);
            auto callback = [input_device_nd]{
                return *input_device_nd;
            };
            auto* input_var = opr::InputCallback::make(*cg_holder.graph, callback, desc.comp_node, desc.layout.dtype, TensorShape())[0].node();
            input_vars.push_back(input_var);
        }
        // forward to inner op
        auto output_vars = OpDef::apply_on_var_node(*compiled_op, input_vars);
        ComputingGraph::OutputSpec output_spec;
        size_t nr_outputs = output_vars.size();
        for (size_t i = 0; i < nr_outputs; ++i) {
            auto* output_var = output_vars[i];
            auto output_ptr = std::make_shared<DeviceTensorND>();
            auto callback = [output_ptr](DeviceTensorND output){
                output_ptr->reset(output.storage(), output.layout());
            };
            output_spec.push_back({output_var, callback});
            cg_holder.outputs.push_back(output_ptr);
        }
        cg_holder.executable = cg_holder.graph->compile(output_spec);
        CompNode::UnorderedSet comp_nodes;
        for (auto&& output_var: output_vars) {
            comp_nodes.insert(output_var->comp_node());
        }
        for (auto&& comp_node: comp_nodes) {
            cg_holder.events.push_back(comp_node.create_event());
            cg_holder.events.back()->record();
        }
        nr_cg_holders++;
        mgb_log_debug("add new computing graph for compiled op, now %zu graphs", nr_cg_holders);
    }
    cg_holder_queue.push(std::move(holder));
    return *cg_holder_queue.back();
}

auto apply_on_physical_tensor(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    SmallVector<LogicalTensorDesc> input_descs;
    for (auto&& input: inputs) {
        input_descs.push_back({input->layout(), input->comp_node()});
    }
    size_t nr_inputs = inputs.size();
    auto shared_def = const_cast<OpDef&>(def).shared_from_this();
    auto& cg_holder = get_computing_graph(shared_def, input_descs);
    // wait for last execution
    cg_holder.executable->wait();
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto input_dev_tensor = inputs[i]->dev_tensor();
        cg_holder.inputs[i]->reset(input_dev_tensor.storage(), input_dev_tensor.layout());
    }
    cg_holder.allocator->current_op = shared_def;
    cg_holder.executable->execute();
    for (auto&& event: cg_holder.events) {
        event->record();
    }
    SmallVector<TensorPtr> outputs;
    for (auto input_nd: cg_holder.inputs) {
        *input_nd = {};
    }
    for (auto output_nd: cg_holder.outputs) {
        outputs.push_back(Tensor::make(*output_nd));
        *output_nd = {};
    }
    cg_holder.executable->clear_device_memory();
    cg_holder.allocator->current_op = nullptr;
    return outputs;
}
auto apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto& op = def.cast_final_safe<CompiledOp>();
    op.op->set_scope(op.scope());
    return OpDef::apply_on_var_node(*op.op, inputs);
}

auto infer_output_attrs_fallible(
        const OpDef& def,
        const SmallVector<LogicalTensorDesc>& input_descs) {
    return OpDef::infer_output_attrs_fallible(*def.cast_final_safe<CompiledOp>().op, input_descs);
}

auto props(const OpDef& def) {
    return OpDef::props(*def.cast_final_safe<CompiledOp>().op);
}

auto make_name(const OpDef& def) {
    auto& op = def.cast_final_safe<CompiledOp>();
    MGB_MARK_USED_VAR(op);
    return ssprintf("CompiledOp[%s]", op.op->make_name().c_str());
}

std::tuple<SmallVector<MemoryDesc>, SmallVector<MemoryDesc>> infer_output_mem_desc(
        const OpDef& def,
        const SmallVector<TensorPtr>& inputs_tensors,
        const SmallVector<MemoryDesc>& inputs_mems) {
    return {};
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, 
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    auto& op = def.cast_final_safe<CompiledOp>();
    auto backward_graph = OpDef::make_backward_graph(*op.op, inputs, input_requires_grad, output_has_grad);
    auto name = def.trait()->make_name(def);
    auto key = std::make_shared<BackwardOpKey>();
    key->op = op.op;
    key->inputs = inputs;
    key->extras = {input_requires_grad, output_has_grad};
    SmallVector<bool> grad_outputs_has_grad(backward_graph.graph.outputs.size(), true);
    std::shared_ptr<OpDef> bgraph_op;
    if (backward_graph.graph.is_single()) {
        bgraph_op = backward_graph.graph.as_single();
    } else {
        bgraph_op = SubgraphOp::make(
                name + "Grad", std::make_shared<Subgraph>(backward_graph.graph),
                grad_outputs_has_grad, key);
    }
    auto compiled_op = CompiledOp::make(bgraph_op, op.gopt_level);
    auto encoded_graph = EncodedSubgraph::make_single(compiled_op, backward_graph.input_mask, backward_graph.output_mask);
    return encoded_graph;
}

auto hash(const OpDef& def) {
    auto& op = def.cast_final_safe<CompiledOp>();
    return mgb::hash_pair_combine(op.op->hash(), op.gopt_level);
}

auto is_same_st(const OpDef& def, const OpDef& another) {
    if (!another.same_type<CompiledOp>()) {
        return false;
    }
    auto& lhs = def.cast_final_safe<CompiledOp>();
    auto& rhs = another.cast_final_safe<CompiledOp>();
    return lhs.op->is_same(*rhs.op) && lhs.gopt_level == rhs.gopt_level;
}

OP_TRAIT_REG(CompiledOp, CompiledOp)
    .apply_on_var_node(apply_on_var_node)
    .apply_on_physical_tensor(apply_on_physical_tensor)
    .infer_output_attrs_fallible(infer_output_attrs_fallible)
    .make_backward_graph(make_backward_graph)
    .make_name(make_name)
    .infer_output_mem_desc(infer_output_mem_desc)
    .props(props)
    .hash(hash)
    .is_same_st(is_same_st)
    .fallback();
}}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(UniqueKey);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SubgraphOp);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BackwardOpKey);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CompiledOp);

} // namespace mgb::imperative
