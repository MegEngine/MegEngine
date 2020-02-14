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

#if MGB_CUDA

#include "zmq_rpc.h"

#include "megbrain/opr/collective_comm.h"
#include "mm_handler.pb.h"

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

    GroupClientProxy(const std::string& server_addr)
            : m_addr(server_addr),
              m_stub{ZmqRpc::ZmqRpcClient::get_client("tcp://" + server_addr)} {
    }

    //! graph registration, assign graph_id to worker.
    uint64_t opr_register(const std::string& key, size_t nr_devices, uint32_t rank,
        uintptr_t stream) override;

    std::vector<std::string> gather_uid(const std::string& uid,
            const std::string& key, uint32_t size, uint32_t rank) override;

    void set_output_shape(const std::string& key,
                                  const TensorShape& shape) override;

    TensorShape get_output_shape(const std::string& key) override;

    uint32_t group_barrier(uint32_t size, uint32_t rank) override;

    //! thread safe to create handler with address
    static GroupClientProxy* get_handler(const std::string& addr) {
        static std::unordered_map<std::string,
                                  std::unique_ptr<GroupClientProxy>>
                addr2handler;
        static std::mutex mtx;
        MGB_LOCK_GUARD(mtx);
        auto it = addr2handler.emplace(addr, nullptr);
        if (!it.second) {
            mgb_assert(it.first->second->m_addr == addr);
            return it.first->second.get();
        } else {
            auto handler = std::make_unique<GroupClientProxy>(addr);
            auto handler_ptr = handler.get();
            it.first->second = std::move(handler);
            return handler_ptr;
        }
    }

    const std::string& get_addr() const {
        return m_addr;
    }

private:
    const std::string m_addr;
    ZmqRpc::ZmqRpcClient* m_stub;
};
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
