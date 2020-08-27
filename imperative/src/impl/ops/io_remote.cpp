/**
 * \file src/core/include/megbrain/imperative.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 *
 */
#include "megbrain_build_config.h"

#if MGB_ENABLE_OPR_MM
#include "../op_trait.h"
#include "../proxy_graph_detail.h"
#include "megbrain/opr/io_remote.h"
#include "megbrain/opr/mm_handler.h"
#endif // MGB_ENABLE_OPR_MM

#include "megbrain/imperative/ops/io_remote.h"

namespace mgb {
namespace imperative {

#if MGB_ENABLE_OPR_MM
namespace {
cg::OperatorNodeBase* apply_on_var_node_remote_send(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& send = def.cast_final_safe<RemoteSend>();
    auto group_client = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", send.addr.data(), send.port));
    auto&& graph = inputs[0]->owner_graph();

    cg::OperatorNodeConfig config;
    cg::OperatorNodeBase* opr =
            graph->insert_opr(std::make_unique<mgb::opr::RemoteSend>(
                    send.key, inputs[0], group_client, true, config));
    return opr;
}

cg::OperatorNodeBase* apply_on_var_node_remote_recv(
        const OpDef& def, const VarNodeArray& inputs) {
    auto&& recv = def.cast_final_safe<RemoteRecv>();
    auto group_client = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", recv.addr.data(), recv.port));
    auto&& graph = inputs[0]->owner_graph();
    return graph->insert_opr(std::make_unique<mgb::opr::RemoteRecv>(
            recv.key, *graph, group_client, OperatorNodeConfig{recv.cn},
            recv.shape, recv.dtype));
}

OP_TRAIT_REG(RemoteSend, RemoteSend, mgb::opr::RemoteSend)
        .apply_on_var_node(apply_on_var_node_remote_send)
        .fallback();

OP_TRAIT_REG(RemoteRecv, RemoteRecv, mgb::opr::RemoteRecv)
        .apply_on_var_node(apply_on_var_node_remote_recv)
        .fallback();
}  // anonymous namespace
#endif // MGB_ENABLE_OPR_MM

MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemoteSend);
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RemoteRecv);

}  // namespace imperative
}  // namespace mgb
