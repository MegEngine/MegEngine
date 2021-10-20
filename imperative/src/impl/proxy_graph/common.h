/**
 * \file imperative/src/impl/proxy_graph/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

namespace mgb::imperative::proxy_graph {

// a "namespace" struct to simplify friend declaration,
// e.g. friend class mgb::imperative::proxy_graph::ProxyGraph
struct ProxyGraph {
    struct InputPlaceholder;
    class MiniGraph;
};

}  // namespace mgb::imperative::proxy_graph
