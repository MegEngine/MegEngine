/**
 * \file src/opr-mm/include/megbrain/opr/collective_comm.h
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
#include "megbrain/opr/param_defs.h"
#include "megbrain/opr/group_manager.h"
#include "megray.h"

namespace mgb {
namespace opr {

//! collective communication between multiple CompNode on localhost
MGB_DEFINE_OPR_CLASS(CollectiveComm, cg::OutshapePureByInshapeOpr<>) // {
public:
    class ModeTrait;

    using Param = megdnn::param::CollectiveComm;

    CollectiveComm(
            VarNodeArray inputs, ComputingGraph* const graph,
            const std::string& key, const size_t nr_devices, const bool is_root,
            const int rank, const bool local_grad,
            std::shared_ptr<GroupClient> group_client, const Param& param,
            const DType& dtype, const std::string& backend,
            const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffer_arr,
            const OperatorNodeConfig& config,
            const std::shared_ptr<DTypeScalar>& disable);

    static SymbolVarArray make(
            const SymbolVarArray& inputs, ComputingGraph* const graph,
            const std::string& key, const size_t nr_devices, const bool is_root,
            const int rank, const bool local_grad,
            std::shared_ptr<GroupClient> group_client,
            const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffer_arr,
            const Param& param, const DType& dtype = {},
            const std::string& backend = "nccl",
            const OperatorNodeConfig& config = {},
            const std::shared_ptr<DTypeScalar>& disable =
                    std::make_shared<DTypeScalar>(0));

    static SymbolVarArray make(const SymbolVarArray& inputs,
                               ComputingGraph* const graph,
                               const std::string& key, const size_t nr_devices,
                               const bool is_root, const int rank,
                               const bool local_grad,
                               std::shared_ptr<GroupClient> group_client,
                               const Param& param, const DType& dtype = {},
                               const std::string& backend = "nccl",
                               const OperatorNodeConfig& config = {},
                               const std::shared_ptr<DTypeScalar>& disable =
                                       std::make_shared<DTypeScalar>(0));

    const Param& param() const { return m_param; }
    const DType& dtype() const { return m_dtype; }
    const std::string& backend() const { return m_backend; }

    //! total number of devices within the clique
    size_t nr_devices() const { return m_nr_devices; }

    //! output buffers
    const SmallVector<std::shared_ptr<DeviceTensorND>>& dev_buffers() const {
        return m_dev_buffers;
    }

    int rank() const { return m_rank; }
    int root() const { return m_root; }
    bool is_root() const { return m_is_root; }
    bool local_grad() const { return m_local_grad; }

    //! The key that identifies an NCCL clique.
    //! Operators with same keys belong to the same clique.
    const std::string& key() const { return m_key; }

    std::shared_ptr<GroupClient> group_client() const {
        return m_group_client;
    }

    void set_pack_hash(uint64_t hash) { m_pack_hash = hash; }

    uint64_t pack_hash() const { return m_pack_hash; }

    std::shared_ptr<MegRay::Context> megray_ctx() const {
        return m_megray_ctx;
    }

    VarNode* grad(VarNode* out_grad) const;

private:
    Barrier m_exec_barrier;

    const Param m_param;
    const DType m_dtype;
    const std::string m_backend;
    void mem_plan_fwd_in2out_writable() override;
    void add_input_layout_constraint() override;
    void get_output_var_shape(const TensorShapeArray& inp_shape,
                              TensorShapeArray& out_shape) const override;
    void init_output_comp_node() override;
    void do_execute(ExecEnv& env) override;
    NodeProp* do_make_node_prop() const override;
    void on_output_comp_node_stream_changed() override;
    void init_output_dtype() override;
    void init_output_static_infer_desc() override;
    void init_output_mem_plan(bool dynamic) override;

    //! init nccl communicators
    void opr_register();

    std::shared_ptr<GroupClient> m_group_client;
    size_t m_nr_devices = 0;
    bool m_is_root;
    int m_rank;
    bool m_local_grad;
    std::string m_key;
    //! XXHash generated from m_key
    size_t m_hash;
    //! root of BROADCAST and REDUCE operation
    int m_root;
    //! rank of root of BROADCAST and REDUCE operation
    Maybe<TensorShape> m_output_shape = None;
    // Whether shape infer is enabled.
    // This is only used by BROADCAST and SCATTER operation,
    // whose shape infer should be disabled *during* static infer phase.
    bool m_enable_shape_infer = false;

    //! set in PackAllReduceScanPass and used in PackAllReduceReplacePass
    uint64_t m_pack_hash = 0;

    std::shared_ptr<MegRay::Context> m_megray_ctx;
    std::shared_ptr<MegRay::Communicator> m_megray_comm;
    bool m_init = false;
    bool m_debug_mode = false;

    //! dev buffers for each outputs
    SmallVector<std::shared_ptr<DeviceTensorND>> m_dev_buffers;
    //! disable flag
    std::shared_ptr<DTypeScalar> m_disable;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
