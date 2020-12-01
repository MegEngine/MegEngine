/**
 * \file imperative/src/impl/ops/collective_comm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain_build_config.h"

#if MGB_ENABLE_OPR_MM
#include "../op_trait.h"
#include "megbrain/imperative/proxy_graph_detail.h"
#include "megbrain/opr/mm_handler.h"
#include "megbrain/utils/hash.h"
#endif // MGB_ENABLE_OPR_MM

#include "megbrain/imperative/ops/autogen.h"

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

std::tuple<std::string, std::string> split_address(const std::string& address_and_port){
    auto index = address_and_port.find_last_of(':');
    mgb_assert(index != std::string::npos, "missing ':' in server address");
    return {address_and_port.substr(0, index), address_and_port.substr(index+1)};
}

std::shared_ptr<OpDef> make_from_op_node(cg::OperatorNodeBase* node) {
    auto&& comm = node->cast_final_safe<opr::CollectiveComm>();
    auto&& group_client = comm.group_client();
    auto [addr, port] = split_address(group_client->get_addr());
    auto comp_node = node->config().get_single_comp_node().to_string_logical();
    return std::make_shared<CollectiveComm>(
            comm.param().mode, comm.key(), comm.nr_devices(), comm.rank(),
            comm.is_root(), comm.local_grad(), addr, std::stoi(port),
            comm.dtype(), comm.backend(), comp_node);
}

OP_TRAIT_REG(CollectiveComm, CollectiveComm, opr::CollectiveComm)
    .apply_on_var_node(apply_on_var_node)
    .make_from_op_node(make_from_op_node)
    .fallback();
} // anonymous namespace
#endif // MGB_ENABLE_OPR_MM

}  // namespace imperative
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
