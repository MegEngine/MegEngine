/**
 * \file imperative/src/impl/proxy_graph/proxy_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./mini_graph.h"
#if 0
// ../proxy_graph.h is deprecated, leave here for debug purpose
// uncomment #if 0 macro to debug
#include "../proxy_graph.h"
#endif

namespace mgb::imperative::proxy_graph {
    MGB_DYN_TYPE_OBJ_FINAL_IMPL(ProxyGraph::InputPlaceholder);

    thread_local std::unique_ptr<ProxyGraphTypeI> ProxyGraphTypeI::sm_instance = {};
} // namespace mgb::imperative::proxy_graph

namespace mgb::imperative::proxy_graph_detail {
std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(const OpDef& def,
        const SmallVector<LogicalTensorDesc>& inputs) {
    auto ret = proxy_graph::ProxyGraphTypeI::inst().infer_output_attrs_fallible(def, inputs);
#if 0
    // delete me after the new implementation is stable
    auto ref = ProxyGraph::get_default_graph()->infer_output_attrs_fallible(def, inputs);
    auto& [a, _1] = ret;
    auto& [b, _2] = ref;
    if (a.size() != b.size()) mgb_trap();
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].layout.dtype != b[i].layout.dtype) mgb_trap();
        if (a[i].comp_node != b[i].comp_node) mgb_trap();
        if (!a[i].layout.eq_shape(b[i].layout)) mgb_trap();
    }
#endif
    return ret;
}

} // namespace mgb::imperative::proxy_graph_detail
