/**
 * \file imperative/src/impl/subgraph_detail.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/subgraph_detail.h"
#include "megbrain/imperative/graph_builder.h"

#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/io.h"

#include "./op_trait.h"

using LayoutConstraintCallback = mgb::VarNode::LayoutConstraintCallback;

namespace mgb {
namespace imperative {
namespace subgraph_detail {

VarNodeArray apply_on_var_node(const OpDef& def, const VarNodeArray& inputs) {
    SmallVector<LogicalTensorDesc> input_descs;
    for (auto&& input : inputs) {
        input_descs.push_back({TensorLayout{input->dtype()}, input->comp_node()});
    }
    auto apply_functor = [&](const std::shared_ptr<OpDef>& op,
                             const VarNodeArray& inputs, size_t nr_outputs) {
        op->set_scope(def.scope());
        return OpDef::apply_on_var_node(*op, inputs);
    };
    auto const_functor = [&](const TensorPtr& value) {
        return opr::ImmutableTensor::make(*inputs[0]->owner_graph(), value->get_value())
                .node();
    };
    auto subgraph = def.trait()->make_forward_graph(def, input_descs);
    auto outputs = subgraph.apply<VarNode*>(inputs, apply_functor, const_functor);
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto subgraph = def.trait()->make_forward_graph(def, inputs);
    bool all_validated = true;
    auto apply_functor = [&](const std::shared_ptr<OpDef>& op,
                             const SmallVector<LogicalTensorDesc>& inputs,
                             size_t nr_outputs) {
        auto [outputs, validated] = OpDef::infer_output_attrs_fallible(*op, inputs);
        all_validated = all_validated && validated;
        return outputs;
    };
    auto const_functor = [&](const TensorPtr& value) {
        return LogicalTensorDesc{
                value->layout(), value->comp_node(),
                value->get_value().proxy_to_default_cpu()};
    };
    auto outputs =
            subgraph.apply<LogicalTensorDesc>(inputs, apply_functor, const_functor);
    return {outputs, all_validated};
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs,
        SmallVector<LogicalTensorDesc>& output_descs, const bool& validated) {
    SmallVector<LogicalTensorDesc> input_descs;
    for (auto&& input : inputs) {
        input_descs.push_back({input->layout(), input->comp_node()});
    }
    auto subgraph = def.trait()->make_forward_graph(def, input_descs);
    auto apply_functor = [&output_descs](
                                 const std::shared_ptr<OpDef>& op,
                                 const SmallVector<TensorPtr>& inputs,
                                 size_t nr_outputs) {
        auto&& constraints = OpDef::get_input_layout_constraint(*op, inputs);
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            auto&& layout_checker = constraints[idx];
            if (layout_checker) {
                inputs[idx]->to_contiguous_inplace(layout_checker);
            }
        }
        // do not use infered output_desc in subgraph
        return OpDef::apply_on_physical_tensor(*op, inputs, output_descs, false);
    };
    auto const_functor = [&](const TensorPtr& value) { return value; };
    auto outputs = subgraph.apply<TensorPtr>(inputs, apply_functor, const_functor);
    return outputs;
}

SmallVector<LayoutConstraintCallback> get_input_layout_constraint(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    SmallVector<LayoutConstraintCallback> res(inputs.size());
    return res;
}

EncodedSubgraph make_backward_graph_from_forward(
        const EncodedSubgraph& forward_graph,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    using namespace std::placeholders;
    using var_t = Subgraph::var_t;
    using vars_t = Subgraph::vars_t;
    Subgraph::Builder<LogicalTensorDesc> builder(
            [](auto&& op, auto&& input_descs, size_t nr_outputs) {
                auto [descs, _] = OpDef::infer_output_attrs_fallible(*op, input_descs);
                mgb_assert(
                        descs.size() == nr_outputs, "nr_outputs mismatch for %s",
                        op->to_string().c_str());
                return descs;
            });
    auto accum_grad = [&](var_t lhs, var_t rhs) {
        return builder.write_expr(
                Elemwise::make(Elemwise::Mode::ADD), {lhs, rhs}, 1)[0];
    };
    GradContext<std::shared_ptr<OpDef>, var_t> grad_context{accum_grad};
    auto input_vars = builder.write_inputs(inputs);
    auto outputs = forward_graph.apply<var_t>(
            input_vars, std::bind(&decltype(builder)::write_expr, &builder, _1, _2, _3),
            [&](TensorPtr constant) {
                return builder.write_constant(
                        constant, {constant->layout(), constant->comp_node()});
            });
    size_t nr_outputs = outputs.size();
    auto apply_mask = [](auto&& values, SmallVector<bool> mask) {
        mgb_assert(mask.size() == values.size());
        std::decay_t<decltype(values)> results;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i]) {
                results.push_back(values[i]);
            }
        }
        return results;
    };
    grad_context.mark_require_grads(apply_mask(input_vars, input_requires_grad));
    builder.iterate([&](std::list<Subgraph::expr_t>::iterator iter) {
        grad_context.record_expr(iter->op, iter->inputs, iter->outputs);
    });
    auto output_descs = builder.get_descs(outputs);
    auto computed_outputs = builder.write_inputs(output_descs);
    auto output_grads = builder.write_inputs(output_descs);

    grad_context.backward(
            apply_mask(outputs, output_has_grad),
            apply_mask(output_grads, output_has_grad),
            [&](Subgraph::op_t op, vars_t inputs, vars_t outputs, vars_t output_grads) {
                auto bg = OpDef::make_backward_graph(
                        *op, builder.get_descs(inputs),
                        grad_context.get_require_grads(inputs),
                        grad_context.get_has_grads(outputs));
                if (bg.graph.empty()) {
                    return vars_t(inputs.size(), 0);
                }
                vars_t grad_inputs;
                grad_inputs.insert(grad_inputs.end(), inputs.begin(), inputs.end());
                grad_inputs.insert(grad_inputs.end(), outputs.begin(), outputs.end());
                grad_inputs.insert(
                        grad_inputs.end(), output_grads.begin(), output_grads.end());
                auto apply_functor =
                        std::bind(&decltype(builder)::write_expr, &builder, _1, _2, _3);
                auto const_functor = [&](TensorPtr constant) {
                    return builder.write_constant(
                            constant, {constant->layout(), constant->comp_node()});
                };
                return bg.apply<var_t>(grad_inputs, apply_functor, const_functor);
            });
    builder.add_outputs(grad_context.get_grads(input_vars));
    for (size_t i = 0; i < nr_outputs; ++i) {
        builder.replace_var(outputs[i], computed_outputs[i]);
    }
    auto backward_graph = builder.encode();
    return backward_graph;
}

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    auto forward_graph = OpDef::make_forward_graph(def, inputs);
    return make_backward_graph_from_forward(
            forward_graph, inputs, input_requires_grad, output_has_grad);
}

EncodedSubgraph make_from_computing_graph(
        const VarNodeArray& inputs, const VarNodeArray& outputs) {
    Subgraph subgraph;
    std::unordered_map<VarNode*, size_t> var2idx;
    size_t next_idx = 0;
    var2idx[nullptr] = next_idx++;
    for (auto&& input : inputs) {
        if (input) {
            var2idx[input] = next_idx++;
        }
    }
    auto is_tensor_holder = [](cg::OperatorNodeBase* op) {
        return op->input().empty();
    };
    auto as_tensor = [](VarNode* var) -> TensorPtr {
        auto* opr = var->owner_opr();
        if (auto* imm_tensor = opr->try_cast_final<opr::ImmutableTensor>()) {
            auto&& dv = imm_tensor->value();
            HostTensorND hv(dv.comp_node(), dv.shape(), dv.dtype());
            // get host value
            auto&& cpu_value = imm_tensor->host_value();
            mgb_assert(cpu_value.comp_node() == CompNode::default_cpu());
            // default_cpu is synchronous with respect to caller
            hv.proxy_to_default_cpu().copy_from_fixlayout(cpu_value);
            return Tensor::make(dv, hv);
        } else if (
                auto* shared_tensor = opr->try_cast_final<opr::SharedDeviceTensor>()) {
            return Tensor::make(shared_tensor->get_dev_tensor());
        } else {
            mgb_assert(
                    false, "unsupported tensor holder opr %s",
                    opr->dyn_typeinfo()->name);
        }
    };
    cg::DepOprIter iter{[&](cg::OperatorNodeBase* op) {
        // TODO: implement make_backward_graph for mm ops
        // mgb_assert(!op->node_prop().contain(cg::OperatorNodeProp::Flag::IMPURE_FUNC));
        if (is_tensor_holder(op)) {
            for (auto&& output : op->usable_output()) {
                subgraph.constants.push_back(
                        {var2idx[output] = next_idx++, as_tensor(output)});
            }
        } else {
            Subgraph::vars_t inputs;
            Subgraph::vars_t outputs;
            for (auto&& input : op->input()) {
                inputs.push_back(var2idx.at(input));
            }
            // NOTE: use usable_output
            for (auto&& output : op->usable_output()) {
                outputs.push_back(var2idx[output] = next_idx++);
            }
            auto opdef = OpDef::make_from_op_node(op);
            subgraph.exprs.push_back({opdef, inputs, outputs});
        }
    }};
    for (auto&& input : inputs) {
        if (input) {
            iter.set_visited(input->owner_opr());
        }
        subgraph.inputs.push_back(var2idx.at(input));
    }
    for (auto&& output : outputs) {
        if (output) {
            iter.add(output);
        }
        subgraph.outputs.push_back(var2idx.at(output));
    }
    return EncodedSubgraph::make(subgraph);
}

}  // namespace subgraph_detail
}  // namespace imperative
}  // namespace mgb
