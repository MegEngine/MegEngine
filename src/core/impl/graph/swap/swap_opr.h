/**
 * \file src/core/impl/graph/swap/swap_opr.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./swap_helper.h"

#include "megbrain/graph.h"
#include "megdnn/opr_param_defs.h"

#include "megbrain/opr/internal/identical_fwd.h"

#if MGB_ENABLE_MEMORY_SWAP

namespace mgb {
namespace swap {
namespace opr {

MGB_DEFINE_OPR_CLASS(WaitSwapInMS, cg::SingleCNOperatorNodeBase,
                           mgb::opr::mixin::ForwardInputToOutput) // {
public:
    WaitSwapInMS(ComputingGraph& grpah, const VarNodeArray& inputs,
                 const OperatorNodeConfig& config);
    static SymbolVar make(ComputingGraph& graph, const SymbolVarArray& inputs,
                          const OperatorNodeConfig& config = {});
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
    void init_output_static_infer_desc() override;
};  // namespace opr

MGB_DEFINE_OPR_CLASS(SwapOutMS, cg::SingleCNOperatorNodeBase) // {
public:
    struct Param {
        std::shared_ptr<SwapVarRecorder> swap_var_recorder_ptr;
        Param(std::shared_ptr<SwapVarRecorder> htp = nullptr)
                : swap_var_recorder_ptr{htp} {};
    };
    SwapOutMS(ComputingGraph& graph, VarNode* inp, const Param& param,
              const OperatorNodeConfig& config);
    static SymbolVar make(ComputingGraph& graph, SymbolVar inp,
                          const Param& param,
                          const OperatorNodeConfig& config = {});
    static SymbolVar make(ComputingGraph& graph, SymbolVar inp,
                          const OperatorNodeConfig& config = {}) {
        return make(graph, inp, {}, config);
    }

    void set_recorder(std::shared_ptr<SwapVarRecorder>& svr_ptr) {
        m_recorder = svr_ptr;
    }

    const std::shared_ptr<SwapVarRecorder>& recorder() const {
        return m_recorder;
    }

    Param param() const { return m_param; }

private:
    std::shared_ptr<SwapVarRecorder> m_recorder;
    const Param m_param;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
};

MGB_DEFINE_OPR_CLASS(SwapInMS, cg::SingleCNOperatorNodeBase) // {
public:
    struct Param {
        std::shared_ptr<SwapVarRecorder> swap_var_recorder_ptr;
        Param(std::shared_ptr<SwapVarRecorder> htp = nullptr)
                : swap_var_recorder_ptr{htp} {};
    };
    SwapInMS(ComputingGraph& graph, VarNode* swap_out_var, VarNode* dep_var,
             const Param& param, const OperatorNodeConfig& config);
    static SymbolVar make(ComputingGraph& graph, SymbolVar inp, SymbolVar d,
                          const Param& param,
                          const OperatorNodeConfig& config = {});
    static SymbolVar make(ComputingGraph& graph, SymbolVar inp, SymbolVar d,
                          const OperatorNodeConfig& config = {}) {
        return make(graph, inp, d, {}, config);
    }

    void set_recorder(std::shared_ptr<SwapVarRecorder>& svr_ptr) {
        m_recorder = svr_ptr;
    }

    const std::shared_ptr<SwapVarRecorder>& recorder() const {
        return m_recorder;
    }

    Param param() const { return m_param; }

    void wait_bucket_copy(const DeviceTensorND* x) {
        m_recorder->wait_mission_finish(x);
    }

private:
    std::shared_ptr<SwapVarRecorder> m_recorder;
    const Param m_param;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
};

MGB_DEFINE_OPR_CLASS(SwapIn, cg::SingleCNOperatorNodeBase) // {
public:
    SwapIn(ComputingGraph& graph, const SymbolVarArray inputs,
           const std::shared_ptr<HostTensorND>& host_data,
           const OperatorNodeConfig& config);

    static SymbolVar make(ComputingGraph& graph, const SymbolVarArray inputs,
                          const std::shared_ptr<HostTensorND>& host_data,
                          const OperatorNodeConfig& config);

    const std::shared_ptr<HostTensorND>& host_data() const {
        return m_host_data;
    }

private:
    std::shared_ptr<HostTensorND> m_host_data;
    void init_output_static_infer_desc() override;
    void init_output_mem_plan(bool dynamic) override final;
    void scn_do_execute() override;
    NodeProp* do_make_node_prop() const override;
};

MGB_DEFINE_OPR_CLASS(SwapOut, cg::SingleCNOperatorNodeBase) // {
public:
    struct Param {
        std::shared_ptr<HostTensorND> host_tensor_ptr;
        //! a shared ptr to the host tensor
        Param(std::shared_ptr<HostTensorND> htp = nullptr)
                : host_tensor_ptr{htp} {};
    };
    SwapOut(ComputingGraph& graph, VarNode* inp, const Param& param,
            const OperatorNodeConfig& config);
    static SymbolVar make(ComputingGraph& graph, SymbolVar inp,
                          const Param& param,
                          const OperatorNodeConfig& config = {});
    static SymbolVar make(ComputingGraph& graph, SymbolVar inp,
                          const OperatorNodeConfig& config = {}) {
        return make(graph, inp, {}, config);
    }

    const std::shared_ptr<HostTensorND>& host_data() const {
        return m_host_data;
    }
    Param param() const { return m_param; }

private:
    std::shared_ptr<HostTensorND> m_host_data;
    const Param m_param;
    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
};

}  // namespace opr
}  // namespace swap
}  // namespace mgb

#endif  // MGB_ENABLE_MEMORY_SWAP

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
