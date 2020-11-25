/**
 * \file src/opr-mm/include/megbrain/opr/io_remote.h
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
#include "megbrain/opr/internal/mixin_base.h"
#include "megbrain/opr/group_manager.h"

#include "megray.h"

namespace mgb {
namespace opr {

/*!
 * \brief base class for remote I/O nodes
 */
MGB_DEFINE_CLS_WITH_SUPER(RemoteIOBase, cg::SingleCNOperatorNodeBase) // {
    public:
        const std::string& key() const { return m_key; }

        std::shared_ptr<GroupClient> group_client() const {
            return m_group_client;
        }

    protected:
        std::string m_key;
        std::shared_ptr<GroupClient> m_group_client;
        std::shared_ptr<MegRay::Communicator> m_megray_comm;
        std::shared_ptr<MegRay::Context> m_megray_ctx;
        bool m_init = false;
        using Super::Super;
};

/*!
 * \brief send a variable to remote address; a virtual output is produced
 *        for expressing dependency
 */
MGB_DEFINE_OPR_CLASS(RemoteSend, RemoteIOBase) // {
    public:
        RemoteSend(const std::string& key, VarNode* var,
                   std::shared_ptr<GroupClient> group_client,
                   bool is_grad, const OperatorNodeConfig& config);

        static SymbolVar make(
                const std::string& key, SymbolVar var,
                std::shared_ptr<GroupClient> group_client,
                bool is_grad, const OperatorNodeConfig& config = {});

        bool is_grad() const { return m_is_grad; }

    private:
        HostTensorND m_output_val;
        bool m_is_grad;

        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        NodeProp* do_make_node_prop() const override;
};

/*!
 * \brief receive a variable from remote address; target computing node
 *        of the var must be specified in config
 */
MGB_DEFINE_OPR_CLASS(RemoteRecv, RemoteIOBase) // {
    public:
        RemoteRecv(const std::string& key, cg::ComputingGraph& graph,
                   std::shared_ptr<GroupClient> group_client,
                   const OperatorNodeConfig& config, const TensorShape& shape,
                   DType dtype);

        RemoteRecv(const std::string& key, VarNode* var, cg::ComputingGraph& graph,
                   std::shared_ptr<GroupClient> group_client,
                   const OperatorNodeConfig& config, const TensorShape& shape,
                   DType dtype);

        static SymbolVar make(
                const std::string& key, cg::ComputingGraph& graph,
                std::shared_ptr<GroupClient> group_client,
                const OperatorNodeConfig& config, const TensorShape& shape,
                DType dtype);

        static SymbolVar make(
                const std::string& key, SymbolVar var, cg::ComputingGraph& graph,
                std::shared_ptr<GroupClient> group_client,
                const OperatorNodeConfig& config, const TensorShape& shape,
                DType dtype);

        const TensorShape& shape() const { return m_shape; }
        const DType& dtype() const { return m_dtype; }

    private:
        const TensorShape m_shape;
        const DType m_dtype;
        const CompNode m_comp_node;
        DeviceTensorND m_dev_buffer;

        void scn_do_execute() override;
        void init_output_static_infer_desc() override;
        NodeProp* do_make_node_prop() const override;
};

} // namespace opr
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
