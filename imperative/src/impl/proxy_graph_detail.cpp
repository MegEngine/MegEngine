/**
 * \file imperative/src/impl/proxy_graph_detail.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/imperative/proxy_graph_detail.h"
#include "./proxy_graph.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb {
namespace imperative {
namespace proxy_graph_detail {

namespace {
SmallVector<Tensor*> to_raw_ptr_array(
        const SmallVector<TensorPtr>& inputs, bool ensure_storage = true) {
    SmallVector<Tensor*> ret;
    for (auto&& i : inputs) {
        mgb_assert(i);
        ret.push_back(i.get());
        if (ensure_storage) {
            // apply lazy allocation
            i->blob()->storage();
        }
    }
    return ret;
}

SmallVector<LogicalTensorDesc> infer_output_attrs(
        const OpDef& def, const SmallVector<TensorPtr>& inputs) {
    auto&& graph = ProxyGraph::get_default_graph();
    return graph->infer_output_attrs(def, to_raw_ptr_array(inputs));
}
}  // anonymous namespace

void exec(
        const OpDef& def, const SmallVector<TensorPtr>& inputs,
        const SmallVector<TensorPtr>& outputs,
        const SmallVector<TensorPtr>& workspaces) {
    auto&& graph = ProxyGraph::get_default_graph();
    auto raw_inputs = to_raw_ptr_array(inputs), raw_outputs = to_raw_ptr_array(outputs),
         raw_workspaces = to_raw_ptr_array(workspaces);
    CompNode::UnorderedSet used_cns;
    for (auto&& out : raw_outputs) {
        auto cn = out->comp_node();
        if (used_cns.insert(cn).second) {
            for (auto&& in : inputs) {
                if (in->comp_node() != cn) {
                    auto&& e = in->get_or_create_event();
                    e->device_wait_by(cn);
                }
            }
        }
    }
    graph->invoke_op(def, raw_inputs, raw_outputs, raw_workspaces);
    for (auto&& cn : used_cns) {
        for (auto&& in : inputs) {
            if (in->comp_node() != cn) {
                in->add_release_callback(cn);
            }
        }
    }
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs) {
    auto output_descs = infer_output_attrs(def, inputs);
    SmallVector<TensorPtr> outputs(output_descs.size(), {});
    for (size_t i = 0; i < outputs.size(); i++) {
        outputs[i] = Tensor::make(output_descs[i].layout, output_descs[i].comp_node);
    }
    exec(def, inputs, outputs, {});
    auto async_error = ProxyGraph::get_async_error();
    if (async_error) {
        throw *async_error;
    }
    return outputs;
}

// std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const
// OpDef& def,
//         const SmallVector<LogicalTensorDesc>& inputs) {
//     auto&& graph = ProxyGraph::get_default_graph();
//     return graph->infer_output_attrs_fallible(def, inputs);
// }

EncodedSubgraph make_backward_graph(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    return ProxyGraph::get_default_graph()->make_backward_graph(
            def, inputs, input_requires_grad, output_has_grad);
}

}  // namespace proxy_graph_detail
}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
