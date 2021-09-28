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

#include <deque>

#include "megbrain/imperative/graph_cache.h"
#include "megbrain/imperative/opr_utility.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/opr_attr.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/subgraph_detail.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_gen.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

#if MGB_JIT
#include "megbrain/jit/executor_opr.h"
#endif

#include "../event_pool.h"
#include "../op_trait.h"

namespace mgb::imperative {

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GenericPyOp);
OP_TRAIT_REG(GenericPyOp, GenericPyOp).fallback();

namespace {
namespace fastpathcopy {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    return inputs;
}

auto make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    Subgraph graph;
    graph.inputs = {1, 2, 3};
    graph.outputs = {3};
    graph.exprs = {};
    return EncodedSubgraph::make(graph);
}

OP_TRAIT_REG(FastpathCopy, FastpathCopy)
        .apply_on_var_node(apply_on_var_node)
        .make_backward_graph(make_backward_graph)
        .fallback();
}  // namespace fastpathcopy
}  // namespace

namespace {
namespace shape_infer {
auto apply_on_physical_tensor(const OpDef& def, const SmallVector<TensorPtr>& inputs) {
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
    auto [output_descs, valid] =
            OpDef::infer_output_attrs_fallible(*op.op, input_descs);
    mgb_assert(valid, "shape inference incomplete");
    SmallVector<TensorPtr> outputs;
    for (auto&& output_desc : output_descs) {
        HostTensorND shape_tensor{
                output_desc.comp_node, {output_desc.layout.ndim}, dtype::Int32()};
        for (size_t i = 0; i < output_desc.layout.ndim; ++i) {
            shape_tensor.ptr<int32_t>()[i] = output_desc.layout[i];
        }
        auto output = Tensor::make(shape_tensor);
        outputs.push_back(output);
    }
    return outputs;
}
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    size_t nr_inputs = inputs.size();
    VarNodeArray input_values, outputs;
    mgb_assert(nr_inputs > 0, "no inputs for ShapeInfer");
    for (size_t i = 0; i < nr_inputs; ++i) {
        auto input_value =
                opr::Alloc::make(SymbolVar(inputs[i]), op.dtypes[i], {op.devices[i]});
        input_values.push_back(input_value.node());
    }
    auto output_values = OpDef::apply_on_var_node(*op.op, input_values);
    for (auto&& output_value : output_values) {
        outputs.push_back(opr::GetVarShape::make(output_value).node());
    }
    return outputs;
}

auto infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    auto& op = def.cast_final_safe<ShapeInfer>();
    SmallVector<LogicalTensorDesc> input_shape_descs;
    size_t nr_inputs = op.devices.size();
    mgb_assert(
            op.dtypes.size() == nr_inputs,
            "number of input devices and dtypes mismatch");
    for (size_t i = 0; i < nr_inputs; ++i) {
        LogicalTensorDesc input_shape_desc;
        input_shape_desc.comp_node = op.devices[i];
        input_shape_desc.layout.ndim = 0;
        input_shape_desc.layout.dtype = op.dtypes[i];
        input_shape_descs.push_back(input_shape_desc);
    }
    auto [output_shape_descs, _] =
            OpDef::infer_output_attrs_fallible(*op.op, input_shape_descs);
    SmallVector<LogicalTensorDesc> output_descs;
    for (auto&& output_shape_desc : output_shape_descs) {
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
    return std::tie(lhs.devices, lhs.dtypes) == std::tie(rhs.devices, rhs.dtypes);
}

OP_TRAIT_REG(ShapeInfer, ShapeInfer)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .make_name(make_name)
        .props(props)
        .hash(hash)
        .is_same_st(is_same_st)
        .fallback();
}  // namespace shape_infer
}  // namespace

MGB_DYN_TYPE_OBJ_FINAL_IMPL(ShapeInfer);

namespace {
namespace identity {
auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto&& op = def.cast_final_safe<Identity>();
    mgb_assert(inputs.size() == 1);
    OperatorNodeConfig config{op.make_name()};
    return opr::Identity::make(inputs[0], config);
}

auto apply_on_physical_tensor(const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    return SmallVector<TensorPtr>{inputs[0]};
}
OP_TRAIT_REG(Identity, Identity)
        .apply_on_var_node(apply_on_var_node)
        .apply_on_physical_tensor(apply_on_physical_tensor)
        .fallback();
}  // namespace identity
}  // namespace

namespace {
namespace subgraph {

EncodedSubgraph make_forward_graph(
        const OpDef& def, SmallVector<LogicalTensorDesc> inputs) {
    return EncodedSubgraph::make(*def.cast_final_safe<SubgraphOp>().graph);
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        SmallVector<bool> output_has_grad) {
    auto& op = def.cast_final_safe<SubgraphOp>();
    mgb_assert(output_has_grad.size() == op.output_grad_mask.size());
    for (size_t i = 0; i < output_has_grad.size(); ++i) {
        if (!op.output_grad_mask[i]) {
            output_has_grad[i] = false;
        }
    }
    auto bgraph = subgraph_detail::make_backward_graph(
            def, inputs, input_requires_grad, output_has_grad);
    return EncodedSubgraph::make_single(
            SubgraphOp::make(
                    op.name + "Grad", std::make_shared<Subgraph>(bgraph.graph)),
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
        return (size_t) reinterpret_cast<uintptr_t>(op.graph.get());
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

}  // namespace subgraph
}  // namespace

namespace {
namespace compiled_op {

struct DeviceMemoryAllocatorImpl : cg::DeviceMemoryAllocator {
    std::shared_ptr<OpDef> current_op;
    void alloc_static(
            ComputingGraph* graph, DeviceTensorStorage& dest, size_t size) override {
        mgb_assert(0, "alloc_static is not allowed in CompiledOp");
    }
    void alloc_dynamic(VarNode* var, DeviceTensorStorage& dest, size_t size) override {
        auto comp_node = var->comp_node();
        auto storage = current_op->allocate(comp_node, size);
        dest.reset(comp_node, size, storage);
    }
};

enum class HolderKind {
    ShapeInfer,
    Execute,
};

template <HolderKind Kind>
struct ComputingGraphHolder {
    struct Input {
        std::shared_ptr<DeviceTensorND> device_value;
        std::shared_ptr<HostTensorND> host_value;
        std::shared_ptr<HostTensorND> host_shape;
    };
    std::shared_ptr<ComputingGraph> graph;
    std::unique_ptr<cg::AsyncExecutable> executable;
    SmallVector<Input> inputs;
    SmallVector<std::shared_ptr<DeviceTensorND>> device_outputs;
    SmallVector<VarNode*> input_vars;
    SmallVector<VarNode*> output_vars;
    std::shared_ptr<DeviceMemoryAllocatorImpl> allocator;
    SmallVector<std::shared_ptr<CompNode::Event>> events;
    std::unique_ptr<cg::static_infer::StaticInferUpdater> updater;

    void initialize(
            const CompiledOp& op, const SmallVector<LogicalTensorDesc>& input_descs) {
        allocator = std::make_shared<DeviceMemoryAllocatorImpl>();
        graph = ComputingGraph::make();
        graph->options().force_dynamic_alloc = true;
        graph->options().async_exec_level = 0;
        graph->options().graph_opt_level = op.gopt_level;
        graph->options().enable_var_mem_defragment = false;
        graph->options().comp_seq_sync_device = false;
        // set allocator for DTR support
        graph->set_device_memory_allocator(allocator);
        if constexpr (Kind == HolderKind::ShapeInfer) {
            updater = cg::static_infer::StaticInferUpdater::make();
        }
        for (auto&& desc : input_descs) {
            Input input;
            VarNode* input_var = nullptr;
            if constexpr (Kind == HolderKind::Execute) {
                input.device_value = std::make_shared<DeviceTensorND>();
                input.device_value->dtype(desc.layout.dtype);
                input.device_value->comp_node(desc.comp_node);
                input.device_value->resize(desc.layout);
                auto callback = [value = input.device_value] { return *value; };
                if (!desc.value.empty()) {
                    input.host_value = std::make_shared<HostTensorND>();
                    input.host_value->dtype(desc.layout.dtype);
                    input.host_value->comp_node(desc.comp_node);
                }
                input_var = opr::MutableTensor::make(
                                    *graph, input.device_value, input.host_value, {})
                                    .node();
                // input_var = opr::VolatileSharedDeviceTensor::make(*graph,
                // input.device_value).node();
            } else if constexpr (Kind == HolderKind::ShapeInfer) {
                if (desc.value.empty()) {
                    input.host_shape = std::make_shared<HostTensorND>();
                    input.host_shape->dtype(dtype::Int32());
                    input.host_shape->comp_node(desc.comp_node);
                    auto input_shape_var =
                            opr::Host2DeviceCopy::make(*graph, input.host_shape);
                    input_var =
                            opr::Alloc::make(input_shape_var, desc.layout.dtype).node();
                } else {
                    input.host_value = std::make_shared<HostTensorND>();
                    input.host_value->dtype(desc.layout.dtype);
                    input.host_value->comp_node(desc.comp_node);
                    input_var =
                            opr::Host2DeviceCopy::make(*graph, input.host_value).node();
                }
            } else {
                static_assert((Kind != Kind), "unknown holder kind");
            }
            input_vars.push_back(input_var);
            inputs.push_back(input);
        }
        // forward to inner op
        output_vars = OpDef::apply_on_var_node(*op.op, input_vars);
        ComputingGraph::OutputSpec output_spec;
        CompNode::UnorderedSet comp_nodes;
        for (auto&& output_var : output_vars) {
            using namespace cg::static_infer;
            auto output_ptr = std::make_shared<DeviceTensorND>();
            auto callback = [output_ptr](DeviceTensorND output) {
                output_ptr->reset(output.storage(), output.layout());
                output = {};
            };
            if constexpr (Kind == HolderKind::ShapeInfer) {
                output_spec.push_back({output_var, callback});
                auto it = graph->static_infer_manager().get_infer_type(output_var);
                if (it.shape == InferType::RT_STATIC) {
                    updater->add_dest({output_var, DepType::SHAPE});
                }
                if (it.value == InferType::RT_STATIC) {
                    updater->add_dest({output_var, DepType::VALUE});
                }
            } else {
                auto output_callback_var =
                        opr::OutputCallback::make({callback}, output_var);
                output_spec.push_back({output_callback_var, {}});
            }
            device_outputs.push_back(output_ptr);
        }
        executable = graph->compile(output_spec);
        executable->iter_opr_seq([&](cg::OperatorNodeBase* opr) -> bool {
            for (auto&& output : opr->output()) {
                comp_nodes.insert(output->comp_node());
            }
            return true;
        });
        for (auto&& comp_node : comp_nodes) {
            events.push_back(EventPool::without_timer().alloc_shared(comp_node));
            events.back()->record();
        }
    }

    template <
            HolderKind ThisKind = Kind,
            typename = std::enable_if_t<ThisKind == HolderKind::Execute>>
    SmallVector<TensorPtr> apply_on_physical_tensor(
            const OpDef& def, const SmallVector<LogicalTensorDesc> input_descs,
            const SmallVector<TensorPtr>& input_tensors) {
        // wait for last execution
        executable->wait();
        size_t nr_inputs = inputs.size();
        for (size_t i = 0; i < nr_inputs; ++i) {
            auto input_dev_tensor = input_tensors[i]->dev_tensor();
            inputs[i].device_value->reset(
                    input_dev_tensor.storage(), input_dev_tensor.layout());
            if (inputs[i].host_value) {
                inputs[i].host_value->copy_from(input_descs[i].value);
            }
        }
        allocator->current_op = const_cast<OpDef&>(def).shared_from_this();
        executable->execute();
        for (auto&& event : events) {
            event->record();
        }
        SmallVector<TensorPtr> outputs_tensors;
        for (auto input : inputs) {
            *input.device_value = {};
            if (input.host_value) {
                *input.host_value = {};
            }
        }
        for (auto output_nd : device_outputs) {
            outputs_tensors.push_back(Tensor::make(*output_nd));
            *output_nd = {};
        }
        executable->clear_device_memory();
        allocator->current_op = nullptr;
        return outputs_tensors;
    }

    template <
            HolderKind ThisKind = Kind,
            typename = std::enable_if_t<ThisKind == HolderKind::ShapeInfer>>
    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
            const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
        executable->wait();
        size_t nr_inputs = input_vars.size(), nr_outputs = output_vars.size();
        SmallVector<LogicalTensorDesc> output_descs(nr_outputs);
        for (size_t i = 0; i < nr_inputs; ++i) {
            if (inputs[i].host_shape) {
                DeviceTensorND input_shape_device_nd;
                cg::copy_shape_to_tensor_value(
                        input_shape_device_nd, input_descs[i].layout);
                inputs[i].host_shape->copy_from(input_shape_device_nd);
                mgb_assert(input_descs[i].layout.ndim, "ndim == 0");
            } else if (inputs[i].host_value) {
                inputs[i].host_value->copy_from(input_descs[i].value);
            }
        }
        updater->update();
        bool validated = true;
        for (size_t i = 0; i < nr_outputs; ++i) {
            auto infer_type =
                    graph->static_infer_manager().get_infer_type(output_vars[i]);
            const TensorShape* output_shape = nullptr;
            const DeviceTensorND* output_value = nullptr;
            auto& desc = output_descs[i];
            if (infer_type.shape != cg::static_infer::InferType::NO_DESC) {
                output_shape = graph->static_infer_manager().infer_shape_fallible(
                        output_vars[i]);
            }
            if (infer_type.value != cg::static_infer::InferType::NO_DESC) {
                output_value = graph->static_infer_manager().infer_value_fallible(
                        output_vars[i]);
            }
            if (output_shape && output_value) {
                mgb_assert(
                        output_shape->eq_shape(output_value->shape()),
                        "shape infer result mismatch, %s vs %s",
                        output_shape->to_string().c_str(),
                        output_value->shape().to_string().c_str());
            }
            if (output_shape) {
                ((TensorShape&)desc.layout) = *output_shape;
            }
            if (output_value) {
                ((TensorShape&)desc.layout) = output_value->shape();
                desc.value = *output_value;
            }
            desc.layout.dtype = output_vars[i]->dtype();
            desc.comp_node = output_vars[i]->comp_node();
            if (!desc.layout.ndim) {
                validated = false;
            }
            desc.layout.init_contiguous_stride();
        }
        return {output_descs, validated};
    }
};

template <HolderKind Kind>
ComputingGraphHolder<Kind>& get_computing_graph(
        std::shared_ptr<OpDef> compiled_op,
        const SmallVector<LogicalTensorDesc>& descs) {
    using ComputingGraphHolderCache =
            OpMethResultCache<std::deque<std::unique_ptr<ComputingGraphHolder<Kind>>>>;
    thread_local auto cache = std::make_unique<ComputingGraphHolderCache>();
    thread_local size_t nr_cg_holders = 0;
    typename ComputingGraphHolderCache::key_t cache_key = {compiled_op, descs};
    auto& cg_holder_queue = (*cache)[cache_key];
    std::unique_ptr<ComputingGraphHolder<Kind>> holder;
    if (!cg_holder_queue.empty()) {
        // pick one
        std::swap(cg_holder_queue.front(), holder);
        // check all events finished
        for (auto&& event : holder->events) {
            if (!event->finished()) {
                bool queue_limited =
                        event->comp_node().contain_flag(CompNode::Flag::QUEUE_LIMITED);
                bool many_graph = cg_holder_queue.size() > 10;
                if (queue_limited || !many_graph) {
                    std::swap(cg_holder_queue.front(), holder);
                    break;
                } else {
                    // graph limit
                    mgb_log_debug(
                            "computing graph limit for compiled op exceeded, waiting "
                            "for prev graph");
                    event->host_wait();
                }
            } else {
                event->host_wait();
            }
        }
        if (holder) {
            cg_holder_queue.pop_front();
        }
    }
    if (!holder) {
        // create new computing graph
        auto create_holder = [&] {
            auto holder = std::make_unique<ComputingGraphHolder<Kind>>();
            auto& cg_holder = *holder;
            cg_holder.initialize(compiled_op->cast_final_safe<CompiledOp>(), descs);
            nr_cg_holders++;
            mgb_log_debug(
                    "add new computing graph for compiled op, now %zu graphs",
                    nr_cg_holders);
            return holder;
        };
        size_t nr_graphs = std::max(cg_holder_queue.size(), (size_t)1);
        for (size_t i = 1; i < nr_graphs; ++i) {
            cg_holder_queue.push_front(create_holder());
        }
        holder = create_holder();
    }
    cg_holder_queue.push_back(std::move(holder));
    return *cg_holder_queue.back();
}

auto apply_on_physical_tensor(const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<LogicalTensorDesc> input_descs;
    for (auto&& input : inputs) {
        input_descs.push_back({input->layout(), input->comp_node()});
        if (auto* host_value = input->try_get_value()) {
            if (host_value->layout().total_nr_elems() <=
                MEGDNN_MAX_NDIM) {  // infer small tensor
                input_descs.back().value = host_value->proxy_to_default_cpu();
            }
        }
    }
    auto shared_def = const_cast<OpDef&>(def).shared_from_this();
    auto& cg_holder = get_computing_graph<HolderKind::Execute>(shared_def, input_descs);
    return cg_holder.apply_on_physical_tensor(def, input_descs, inputs);
}

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto& op = def.cast_final_safe<CompiledOp>();
    op.op->set_scope(op.scope());
    return OpDef::apply_on_var_node(*op.op, inputs);
}

auto infer_output_attrs_fallible(
        const OpDef& def, SmallVector<LogicalTensorDesc> input_descs) {
    bool shape_all_valid = true;
    for (auto&& input_desc : input_descs) {
        if (!input_desc.layout.ndim) {
            shape_all_valid = false;
            break;
        }
    }
    if (!shape_all_valid) {
        return OpDef::infer_output_attrs_fallible(
                *def.cast_final_safe<CompiledOp>().op, input_descs);
    }
    auto shared_def = const_cast<OpDef&>(def).shared_from_this();
    for (auto& input_desc : input_descs) {
        if (input_desc.layout.total_nr_elems() >
            MEGDNN_MAX_NDIM) {  // skip large tensor
            input_desc.value = {};
        }
    }
    auto& cg_holder =
            get_computing_graph<HolderKind::ShapeInfer>(shared_def, input_descs);
    return cg_holder.infer_output_attrs_fallible(def, input_descs);
}

auto props(const OpDef& def) {
    return OpDef::props(*def.cast_final_safe<CompiledOp>().op);
}

auto make_name(const OpDef& def) {
    auto& op = def.cast_final_safe<CompiledOp>();
    MGB_MARK_USED_VAR(op);
    return ssprintf("CompiledOp[%s]", op.op->make_name().c_str());
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    auto& op = def.cast_final_safe<CompiledOp>();
    auto backward_graph = OpDef::make_backward_graph(
            *op.op, inputs, input_requires_grad, output_has_grad);
    auto name = def.trait()->make_name(def);
    std::shared_ptr<OpDef> bgraph_op =
            SubgraphOp::wrap(name + "Grad", backward_graph.graph);
    auto compiled_op = CompiledOp::make(bgraph_op, op.gopt_level);
    auto encoded_graph = EncodedSubgraph::make_single(
            compiled_op, backward_graph.input_mask, backward_graph.output_mask);
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
        .props(props)
        .hash(hash)
        .is_same_st(is_same_st)
        .fallback();
}  // namespace compiled_op
}  // namespace

namespace {
namespace jit_fusion {

static thread_local bool tm_enabled = true;

auto apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    auto& op = def.cast_final_safe<JITFusionOp>();
    op.op->set_scope(op.scope());
    auto outputs = OpDef::apply_on_var_node(*op.op, inputs);
    if (!tm_enabled) {
        // skip for dump (JITExecutor can not be dumped)
        return outputs;
    }
#if MGB_JIT
    for (auto& output : outputs) {
        jit::InternalGraphGenerator igg{output->owner_opr()};
        std::vector<cg::OperatorNodeBase*> reverse_order;
        cg::DepOprIter iter{
                [&](cg::OperatorNodeBase* opr) { reverse_order.push_back(opr); }};
        for (auto&& input : inputs) {
            iter.set_visited(input->owner_opr());
        }
        iter.add(output->owner_opr());
        std::reverse(reverse_order.begin(), reverse_order.end());
        for (auto&& opr : reverse_order) {
            igg.add_opr(opr);
        }
        auto ig = igg.generate();
        output = jit::JITExecutor::make(ig, igg.orig_inps()).node();
    }
#else
    mgb_assert(false, "MGB_WITH_JIT was disabled");
#endif
    return outputs;
}

auto infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& input_descs) {
    return OpDef::infer_output_attrs_fallible(
            *def.cast_final_safe<JITFusionOp>().op, input_descs);
}

auto props(const OpDef& def) {
    return OpDef::props(*def.cast_final_safe<JITFusionOp>().op);
}

auto hash(const OpDef& def) {
    return def.cast_final_safe<JITFusionOp>().op->hash();
}

auto is_samt_st(const OpDef& def, const OpDef& another) {
    if (!another.same_type<JITFusionOp>()) {
        return false;
    }
    auto& lhs = def.cast_final_safe<JITFusionOp>();
    auto& rhs = another.cast_final_safe<JITFusionOp>();
    return lhs.op->is_same(*rhs.op);
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    return {};
}

OP_TRAIT_REG(JITFusionOp, JITFusionOp)
        .apply_on_var_node(apply_on_var_node)
        .infer_output_attrs_fallible(infer_output_attrs_fallible)
        .props(props)
        .hash(hash)
        .is_same_st(is_samt_st)
        .make_backward_graph(make_backward_graph)
        .fallback();

}  // namespace jit_fusion
}  // namespace

bool JITFusionOp::set_enabled(bool enabled) {
    std::swap(enabled, jit_fusion::tm_enabled);
    return enabled;
}

MGB_DYN_TYPE_OBJ_FINAL_IMPL(UniqueKey);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(SubgraphOp);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BackwardOpKey);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CompiledOp);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(JITFusionOp);

}  // namespace mgb::imperative
