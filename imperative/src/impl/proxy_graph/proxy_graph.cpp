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

namespace mgb::imperative::proxy_graph {
MGB_DYN_TYPE_OBJ_FINAL_IMPL(ProxyGraph::InputPlaceholder);

thread_local std::unique_ptr<ProxyGraphTypeI> ProxyGraphTypeI::sm_instance = {};
}  // namespace mgb::imperative::proxy_graph

namespace mgb::imperative::proxy_graph_detail {

std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
        const OpDef& def, const SmallVector<LogicalTensorDesc>& inputs) {
    auto ret = proxy_graph::ProxyGraphTypeI::inst().infer_output_attrs_fallible(
            def, inputs);
    return ret;
}

SmallVector<TensorPtr> apply_on_physical_tensor(
        const OpDef& def, SmallVector<TensorPtr> inputs) {
    auto ret =
            proxy_graph::ProxyGraphTypeI::inst().apply_on_physical_tensor(def, inputs);
    return ret;
}

}  // namespace mgb::imperative::proxy_graph_detail
