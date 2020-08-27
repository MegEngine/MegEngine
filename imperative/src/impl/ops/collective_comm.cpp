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
#include "megbrain/opr/mm_handler.h"
#endif // MGB_ENABLE_OPR_MM

#include "megbrain/imperative/ops/collective_comm.h"

namespace mgb {
namespace imperative {

#if MGB_ENABLE_OPR_MM
namespace {
cg::OperatorNodeBase* apply_on_var_node(
        const OpDef& def,
        const VarNodeArray& inputs) {
    auto&& comm = def.cast_final_safe<CollectiveComm>();
    auto group_client = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", comm.addr.data(), comm.port));
    SmallVector<std::shared_ptr<mgb::DeviceTensorND>> dev_buffer_arr(1, nullptr);
    auto disable = std::make_shared<DTypeScalar>();
    disable->set(0);

    cg::OperatorNodeConfig config;
    if (comm.comp_node.size() > 0) {
        config.comp_node(CompNode::load(comm.comp_node));
    }

    mgb_assert(inputs.size() == 1, "exactly one input expected");
    auto&& graph = inputs[0]->owner_graph();

    return graph->insert_opr(std::make_unique<opr::CollectiveComm>(
            inputs, graph, comm.key, comm.nr_devices, comm.is_root, comm.rank,
            comm.local_grad, group_client, comm.mode, comm.dtype, comm.backend,
            dev_buffer_arr, config, disable));
}

OP_TRAIT_REG(CollectiveComm, CollectiveComm, opr::CollectiveComm)
    .apply_on_var_node(apply_on_var_node)
    .fallback();
} // anonymous namespace
#endif // MGB_ENABLE_OPR_MM

MGB_DYN_TYPE_OBJ_FINAL_IMPL(CollectiveComm);

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
