/**
 * \file imperative/src/include/megbrain/imperative/ops/backward_graph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative/op_def.h"

namespace mgb {
namespace imperative {

// a special OpDef used for taking gradient on physical tensor
struct BackwardGraph final : public OpDefImplBase<BackwardGraph> {
    MGB_DYN_TYPE_OBJ_FINAL_DECL;
public:
    struct InternalGraph {
        // op, inputs, outputs
        using Expr = std::tuple<std::shared_ptr<OpDef>,
                std::vector<size_t>, std::vector<size_t>>;
        std::vector<Expr> exprs;

        // index array of input nodes
        std::vector<size_t> inputs;

        // index array of output nodes
        std::vector<size_t> outputs;

        // pair of (node index, correspending constant)
        std::vector<std::pair<size_t, TensorPtr>> constants;

        SmallVector<TensorPtr>
        apply(const SmallVector<TensorPtr>& inputs) const;

        SmallVector<LogicalTensorDesc>
        infer_attrs(const SmallVector<LogicalTensorDesc>& inputs) const;
    };

    const InternalGraph& graph() const {
        return m_graph;
    }

    InternalGraph& graph() {
        return m_graph;
    }

private:
    InternalGraph m_graph;
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
