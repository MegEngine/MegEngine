/**
 * \file imperative/src/impl/proxy_graph.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/imperative.h"
#include "megbrain/graph/cg.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/comp_node.h"

#include "megbrain/imperative/ops/backward_graph.h"

namespace mgb {
namespace imperative {

class ProxyGraph : public NonCopyableObj {
public:
    static ProxyGraph* get_default_graph();

    /********************** Physical Tensor API **********************/

    SmallVector<LogicalTensorDesc> infer_output_attrs(
            const OpDef& opdef,
            const SmallVector<Tensor*>& inputs);

    void invoke_op(
            const OpDef& opdef,
            const SmallVector<Tensor*>& inputs,
            const SmallVector<Tensor*>& outputs);

    BackwardGraphResult make_backward_graph(
            const OpDef& opdef,
            const SmallVector<LogicalTensorDesc>& input_descs,
            const SmallVector<bool>& input_requires_grad,
            const SmallVector<bool>& output_has_grad);

    /********************** Logical Tensor API **********************/

    size_t get_opr_output_size(
            const OpDef& opdef,
            const SmallVector<LogicalTensorDesc>& inputs);

    std::tuple<SmallVector<LogicalTensorDesc>, bool> infer_output_attrs_fallible(
            const OpDef& opdef,
            const SmallVector<LogicalTensorDesc>& inputs);

private:
    ProxyGraph();

    class ProxyGraphImpl;
    class ExecEnv;
    class StaticInferManager;
    class SeqCompNodeOptimizer;
    class InputPlaceholder;
    struct ProxyGraphInst;
    struct GradGraph;
    struct CurOprGuard;

    void reset();

    /********************** Physical Tensor Helper **********************/

    void cleanup();

    void init_output_tensor(
            const SmallVector<Tensor*>& outputs);

    cg::OperatorNodeBase* get_proxy_opr(
            const OpDef& opdef,
            const SmallVector<Tensor*>& inputs);

    /********************** Logical Tensor Helper **********************/

    cg::OperatorNodeBase* get_proxy_opr(
            const OpDef& opdef,
            const SmallVector<LogicalTensorDesc>& inputs);

    cg::VarNodeArray make_input_place_holders(
            const SmallVector<LogicalTensorDesc>& inputs);

    /********************** Common Helper **********************/

    bool do_shape_infer(bool sync_value);

    TensorPtr as_tensor(cg::OperatorNodeBase* opr, bool share=true);

    cg::OperatorNodeBase* m_cur_opr = nullptr;
    std::unique_ptr<ProxyGraphImpl> m_graph;
    size_t m_max_op_cnt = 100;
    std::unique_ptr<ExecEnv> m_env;
    std::unique_ptr<StaticInferManager> m_static_infer_manager;
    std::unique_ptr<SeqCompNodeOptimizer> m_seq_comp_node_optimizer;
};

} // namespace imperative
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
