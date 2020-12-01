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

void exec(const OpDef& def,
        const SmallVector<TensorPtr>& inputs_,
        const SmallVector<TensorPtr>& outputs_) {
    auto&& graph = ProxyGraph::get_default_graph();
    auto inputs = to_raw_ptr_array(inputs_),
         outputs = to_raw_ptr_array(outputs_);
    CompNode::UnorderedSet used_cns;
    for (auto&& out: outputs) {
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
    graph->invoke_op(def, inputs, outputs);
    for (auto&& cn: used_cns) {
        for (auto&& in: inputs) {
            if (in->comp_node() != cn) {
                in->add_release_callback(cn);
            }
        }
    }
}

SmallVector<LogicalTensorDesc>
infer_output_attrs(const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto&& graph = ProxyGraph::get_default_graph();
    return graph->infer_output_attrs(def, to_raw_ptr_array(inputs));
}
} // anonymous namespace

SmallVector<TensorPtr>
apply_on_physical_tensor(const OpDef& def,
        const SmallVector<TensorPtr>& inputs) {
    auto desc = infer_output_attrs(def, inputs);
    SmallVector<TensorPtr> outputs;
    for (auto&& i : desc) {
        outputs.push_back(Tensor::make(i.layout, i.comp_node));
    }
    exec(def, inputs, outputs);
    return outputs;
}

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto&& graph = ProxyGraph::get_default_graph();
    return graph->infer_output_attrs_fallible(def, inputs);
}

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
