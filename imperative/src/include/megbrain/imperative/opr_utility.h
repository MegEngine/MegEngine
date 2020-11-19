/**
 * \file imperative/src/include/megbrain/imperative/opr_utility.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"
#include "megbrain/serialization/sereg.h"

#include "megdnn/oprs/utils.h"

namespace mgb {
namespace opr {
/*
 * InputCallback, OutputCallback, NopCallback
 * Intended for runtime data exchange with Python.
 */

MGB_DEFINE_OPR_CLASS(InputCallback, cg::SingleCNOperatorNodeBase) // {
public:
    using callback_t = thin_function<DeviceTensorND(void)>;
    InputCallback(cg::ComputingGraph& graph,
                  callback_t callback,
                  const VarNodeArray& inputs,
                  const TensorShape& output_shape,
                  const OperatorNodeConfig &config,
                  bool use_static_shape);
    static SymbolVarArray make(cg::ComputingGraph& graph,
                               callback_t callback,
                               CompNode comp_node,
                               DType dtype,
                               const TensorShape& shape,
                               const SymbolVarArray& inputs = {},
                               bool use_static_shape = false);
    static cg::OperatorNodeBase* shallow_copy(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config);
protected:
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
private:
    TensorShape m_output_shape;
    callback_t m_callback;
    bool m_use_static_shape;
};

MGB_DEFINE_OPR_CLASS(OutputCallback, cg::SingleCNOperatorNodeBase) // {
public:
    using callback_t = thin_function<void(DeviceTensorND)>;
    struct Param {
        callback_t callback;
        bool borrow = false; // do not obtain shared ownership on DeviceTensorND
        bool prefer_host_value = false; // use host value when possible
        bool require_contiguous = true;
    };
    OutputCallback(Param param,
                   const VarNodeArray& inputs,
                   const OperatorNodeConfig &config);
    static SymbolVar make(Param param,
                          const SymbolVarArray& inputs);
    static SymbolVar make(Param param,
                          SymbolVar input) {
        return make(std::move(param), SymbolVarArray{input});
    }
    static cg::OperatorNodeBase* shallow_copy(
            const serialization::OprShallowCopyContext &ctx,
            const cg::OperatorNodeBase &opr_, const VarNodeArray &inputs,
            const OperatorNodeConfig &config);
protected:
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
    void add_input_layout_constraint() override;
private:
    Param m_param;
    mutable bool m_use_host_value;
};

MGB_DEFINE_OPR_CLASS(NopCallback, cg::OperatorNodeBase) // {
public:
    using callback_t = thin_function<void(void)>;
    NopCallback(cg::ComputingGraph& graph,
                callback_t callback,
                const VarNodeArray& inputs,
                const OperatorNodeConfig &config);
    static SymbolVar make(cg::ComputingGraph& graph,
                          callback_t callback,
                          CompNode comp_node,
                          const SymbolVarArray& inputs = {});
protected:
    void do_execute(ExecEnv &env) override;
    void init_output_static_infer_desc() override;
    void init_output_comp_node() override;
    void on_output_comp_node_stream_changed() override;
    NodeProp* do_make_node_prop() const override;
private:
    callback_t m_callback;
};
} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
