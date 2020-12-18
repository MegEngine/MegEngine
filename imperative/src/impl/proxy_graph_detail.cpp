/**
 * \file imperative/src/impl/proxy_graph_detail.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./proxy_graph.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/imperative/ops/autogen.h"

namespace mgb {
namespace imperative {
namespace proxy_graph_detail {

namespace {
SmallVector<Tensor*> to_raw_ptr_array(
        const SmallVector<TensorPtr>& inputs,
        bool ensure_storage=true) {
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

SmallVector<LogicalTensorDesc>
infer_output_attrs(const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto&& graph = ProxyGraph::get_default_graph();
    return graph->infer_output_attrs(def, to_raw_ptr_array(inputs));
}
} // anonymous namespace

void exec(const OpDef& def,
        const SmallVector<TensorPtr>& inputs,
        const SmallVector<TensorPtr>& outputs) {
    auto&& graph = ProxyGraph::get_default_graph();
    auto raw_inputs = to_raw_ptr_array(inputs),
         raw_outputs = to_raw_ptr_array(outputs);
    CompNode::UnorderedSet used_cns;
    for (auto&& out: raw_outputs) {
        auto cn = out->comp_node();
        if (used_cns.insert(cn).second) {
            for (auto&& in: inputs) {
                if (in->comp_node() != cn) {
                    auto&& e = in->get_or_create_event();
                    e->device_wait_by(cn);
                }
            }
        }
    }
    graph->invoke_op(def, raw_inputs, raw_outputs);
    for (auto&& cn: used_cns) {
        for (auto&& in: inputs) {
            if (in->comp_node() != cn) {
                in->add_release_callback(cn);
            }
        }
    }
}

SmallVector<TensorPtr>
apply_on_physical_tensor(const OpDef& def,
        SmallVector<TensorPtr> inputs) {
    auto output_descs = infer_output_attrs(def, inputs);
    SmallVector<TensorPtr> outputs(output_descs.size(), {});
    for (size_t i = 0; i < outputs.size(); i++) {
        auto& output = outputs[i];
        auto& output_desc = output_descs[i];
        if (def.same_type<Elemwise>()) {
            for (size_t j = 0; j < inputs.size(); j++) {
                // TODO: reindex inputs to support inplace exprs like 'y = x op x'.
                auto& input = inputs[j];
                // Because we pass inputs by value, if input and input->blob() are all unique,
                // their ownerships are on the stack, thus we can reuse them safely.
                // @see: interpreter::intl::ChannelImpl::process_one_task
                if (input.unique() && input->blob().unique() && input->blob()->storage().unique() &&
                    input->layout().dtype == output_desc.layout.dtype &&
                    input->layout().eq_layout(output_desc.layout) &&
                    input->comp_node() == output_desc.comp_node) {
                    static std::atomic_llong inplace_count = 0;
                    mgb_log_debug("do inplace for elemwise, layout: %s, count: %lld",
                            output_desc.layout.to_string().c_str(), ++inplace_count);
                    output = Tensor::make(input->blob(), input->layout(), input->offset());
                    break;
                }
            }
        }
        if (!output) {
            output = Tensor::make(output_desc.layout, output_desc.comp_node);
        }
    }
    exec(def, inputs, outputs);
    return outputs;
}

// std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const OpDef& def,
//         const SmallVector<LogicalTensorDesc>& inputs) {
//     auto&& graph = ProxyGraph::get_default_graph();
//     return graph->infer_output_attrs_fallible(def, inputs);
// }

namespace {

size_t get_backward_graph_hash_key(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    XXHash state;
    size_t length = 0, data[3 + 2 * inputs.size()];
    data[length ++] = def.hash();
    for (auto &&i : inputs) {
        data[length ++] = mgb::hash(i.layout.dtype.handle());
        data[length ++] = mgb::hash(i.comp_node);
    }
    data[length ++] = mgb::hash(input_requires_grad);
    data[length ++] = mgb::hash(output_has_grad);
    mgb_assert(length == 3 + 2 * inputs.size());
    state.update(data, length * sizeof(size_t));
    return state.digest();
}

struct BackwardGraphCache : std::unordered_map<size_t, BackwardGraphResult>, CompNodeDepedentObject {
    std::shared_ptr<void> on_comp_node_finalize() override {
        clear();
        return {};
    }
} backward_graph_cache;

} // anonymous namespace

BackwardGraphResult
make_backward_graph(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs,
        const SmallVector<bool>& input_requires_grad,
        const SmallVector<bool>& output_has_grad) {
    auto hash_key = get_backward_graph_hash_key(def, inputs, input_requires_grad, output_has_grad);
    auto&& iter = backward_graph_cache.find(hash_key);
    if (iter != backward_graph_cache.end()) {
        return iter->second;
    }
    auto&& graph = ProxyGraph::get_default_graph();
    auto res = graph->make_backward_graph(def, inputs, input_requires_grad, output_has_grad);
    backward_graph_cache.emplace(hash_key, res);
    return res;
}

} // namespace proxy_graph_detail
} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
