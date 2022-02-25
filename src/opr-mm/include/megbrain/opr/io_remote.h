#pragma once

#include "megbrain/graph.h"
#include "megbrain/opr/group_manager.h"
#include "megbrain/opr/internal/mixin_base.h"

#include "megray.h"

namespace mgb {
namespace opr {

/*!
 * \brief base class for remote I/O nodes
 */
MGB_DEFINE_CLS_WITH_SUPER(RemoteIOBase, cg::SingleCNOperatorNodeBase) // {
public:
    const std::string& key() const { return m_key; }

    std::shared_ptr<GroupClient> group_client() const { return m_group_client; }

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
    RemoteSend(
            const std::string& key, VarNode* var,
            std::shared_ptr<GroupClient> group_client, bool is_grad,
            std::string backend, const OperatorNodeConfig& config);

    static SymbolVar make(
            const std::string& key, SymbolVar var,
            std::shared_ptr<GroupClient> group_client, bool is_grad,
            std::string backend, const OperatorNodeConfig& config = {});

    const std::string& backend() const { return m_backend; }
    bool is_grad() const { return m_is_grad; }

private:
    HostTensorND m_output_val;
    std::string m_backend;
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
    RemoteRecv(
            const std::string& key, cg::ComputingGraph& graph,
            std::shared_ptr<GroupClient> group_client, const OperatorNodeConfig& config,
            const TensorShape& shape, DType dtype, std::string backend);

    RemoteRecv(
            const std::string& key, VarNode* var, cg::ComputingGraph& graph,
            std::shared_ptr<GroupClient> group_client, const OperatorNodeConfig& config,
            const TensorShape& shape, DType dtype, std::string backend);

    static SymbolVar make(
            const std::string& key, cg::ComputingGraph& graph,
            std::shared_ptr<GroupClient> group_client, const OperatorNodeConfig& config,
            const TensorShape& shape, DType dtype, std::string backend);

    static SymbolVar make(
            const std::string& key, SymbolVar var, cg::ComputingGraph& graph,
            std::shared_ptr<GroupClient> group_client, const OperatorNodeConfig& config,
            const TensorShape& shape, DType dtype, std::string backend);

    const TensorShape& shape() const { return m_shape; }
    const DType& dtype() const { return m_dtype; }
    const std::string& backend() const { return m_backend; }

private:
    const TensorShape m_shape;
    const DType m_dtype;
    const std::string m_backend;
    const CompNode m_comp_node;
    DeviceTensorND m_dev_buffer;

    void scn_do_execute() override;
    void init_output_static_infer_desc() override;
    NodeProp* do_make_node_prop() const override;
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
