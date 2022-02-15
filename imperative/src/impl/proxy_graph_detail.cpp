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
