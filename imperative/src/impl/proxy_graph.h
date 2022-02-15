/**
 * \file imperative/src/impl/proxy_graph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"
#include "megbrain/graph/cg.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/imperative.h"

#include "megbrain/imperative/ops/backward_graph.h"

namespace mgb {
namespace imperative {

class ProxyGraph : public NonCopyableObj {
public:
    static ProxyGraph* get_default_graph();
    static std::unique_ptr<MegBrainError> get_async_error() {
        return std::move(tm_async_error);
    }

    /********************** Physical Tensor API **********************/

    EncodedSubgraph make_backward_graph(
            const OpDef& opdef, const SmallVector<LogicalTensorDesc>& input_descs,
            const SmallVector<bool>& input_requires_grad,
            const SmallVector<bool>& output_has_grad);

private:
    ProxyGraph();

    class ProxyGraphImpl;
    class StaticInferManager;
    class SeqCompNodeOptimizer;
    class InputPlaceholder;
    struct ProxyGraphInst;
    class CurOprGuard;

    void reset();

    /********************** Physical Tensor Helper **********************/

    void cleanup();

    /********************** Logical Tensor Helper **********************/

    cg::VarNodeArray make_input_place_holders(
            const SmallVector<LogicalTensorDesc>& inputs);

    /********************** Common Helper **********************/

    TensorPtr as_tensor(cg::OperatorNodeBase* opr, bool share = true);

    cg::OperatorNodeBase* m_cur_opr = nullptr;
    std::unique_ptr<ProxyGraphImpl> m_graph;
    size_t m_max_op_cnt = 100;
    std::unique_ptr<StaticInferManager> m_static_infer_manager;
    std::unique_ptr<SeqCompNodeOptimizer> m_seq_comp_node_optimizer;

    static thread_local std::unique_ptr<MegBrainError> tm_async_error;
};

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
