/**
 * \file src/core/include/megbrain/opr_utility.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain/graph.h"
#include "megbrain/graph/event.h"
#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/internal/param_tag_defs.h"
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/param_defs.h"

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
                  const OperatorNodeConfig &config);
    static SymbolVarArray make(cg::ComputingGraph& graph,
                               callback_t callback,
                               CompNode comp_node,
                               DType dtype,
                               const SymbolVarArray& inputs = {});
protected:
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
private:
    callback_t m_callback;
};

MGB_DEFINE_OPR_CLASS(OutputCallback, cg::SingleCNOperatorNodeBase) // {
public:
    using callback_t = thin_function<void(DeviceTensorND)>;
    struct Param {
        callback_t callback;
        bool borrow = false;
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
protected:
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
private:
    Param m_param;
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
