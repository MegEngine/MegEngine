/**
 * \file python_module/src/cpp/mm_handler.h
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#pragma once

#include "megbrain_build_config.h"

#if MGB_ENABLE_OPR_MM

#include "megbrain/opr/collective_comm.h"
#include "megbrain/opr/group_manager.h"

using namespace mgb;
using namespace opr;

/*!
 * Comm MM Client Proxy.
 * proxy the call by using zmqrpc client interact with zmqrpc server.
 */
class GroupClientProxy
        : public std::enable_shared_from_this<GroupClientProxy>,
          public opr::GroupClient {
public:
    virtual ~GroupClientProxy() = default;

    GroupClientProxy(const std::string& server_addr);

    //! graph registration, assign graph_id to worker.
    GroupManager::RegisterInfo opr_register(const std::string& key,
                                            size_t nr_devices, bool is_root,
                                            int rank,
                                            uint64_t comp_node_hash) override;

    void bcast_addr(std::string& master_ip, int& port, const std::string& key,
            uint32_t size, uint32_t rank, uint32_t root) override;

    void set_output_shape(const std::string& key,
                                  const TensorShape& shape) override;

    TensorShape get_output_shape(const std::string& key) override;

    uint32_t group_barrier(uint32_t size, uint32_t rank) override;

    const std::string& get_addr() const override {
        return m_addr;
    }

private:
    const std::string m_addr;
    void* m_stub;
};


/* ======================== ZmqRpcServerMgr ========================== */

int create_zmqrpc_server(const std::string& server_addr, int port);

#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
