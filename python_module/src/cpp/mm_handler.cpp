/**
 * \file python_module/src/cpp/mm_handler.cpp
 *
 * This file is part of MegBrain, a deep learning framework developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 */

#include "mm_handler.h"

#include "megbrain/exception.h"
#include "megbrain_config.h"

#if MGB_CUDA
#include "zmq_rpc.h"
#include <future>

/* ======================== GroupServerProxy ========================== */
/*!
 * A proxy that receives zmqrpc call, direct call to NCCL Manager
 */

#define RUNSERVER(rpc_name)                                   \
    if (std::strcmp(describe, #rpc_name) == 0) {              \
        std::string output;                                   \
        rpc_name(input_ptr, input_len, &output);              \
        reply.rebuild(output.length());                       \
        memcpy(reply.data(), output.data(), output.length()); \
        return;                                               \
    }

class GroupServerProxy final : public ZmqRpc::ZmqRpcServerImpl {
public:
    void solve_request(zmq::message_t& request,
                       zmq::message_t& reply) override {
        char* describe = (char*)request.data();
        void* input_ptr = (char*)request.data() + strlen(describe) + 1;
        size_t input_len = request.size() - strlen(describe) - 1;
        RUNSERVER(opr_register);
        RUNSERVER(set_output_shape);
        RUNSERVER(get_output_shape);
        RUNSERVER(gather_uid);
        RUNSERVER(group_barrier);
        mgb_assert(false, "invalid rpc request");
    }
private:
    void opr_register(void* input_ptr, size_t input_len, std::string *output);
    void set_output_shape(void* input_ptr, size_t input_len, std::string *output);
    void get_output_shape(void* input_ptr, size_t input_len, std::string *output);
    void gather_uid(void* input_ptr, size_t input_len, std::string *output);
    void group_barrier(void* input_ptr, size_t input_len, std::string *output);

private:
    GroupManager m_mgr;
};

#undef RUNSERVER

#define INFO_INIT(space, name)                               \
    using Request = space::name##Request;   \
    using Response = space::name##Response; \
    Request req;                                      \
    Response rsp;                                     \
    req.ParseFromArray(input_ptr, input_len);

void GroupServerProxy::opr_register(void* input_ptr, size_t input_len,
        std::string *output) {
    INFO_INIT(mm_handler, OprRegister);
    uint64_t hash = m_mgr.opr_register(req.key(), req.nr_expected_devices(),
        req.rank(), req.stream());
    rsp.set_hash(hash);
    rsp.SerializeToString(output);
}

void GroupServerProxy::set_output_shape(void* input_ptr, size_t input_len,
        std::string *output) {
    INFO_INIT(mm_handler, SetOutputShape);
    auto&& shape_proto = req.shape();
    TensorShape shape;
    shape.ndim = shape_proto.ndim();
    for (size_t i = 0; i < shape.ndim; ++i) {
        shape.shape[i] = shape_proto.shape(i);
    }
    m_mgr.set_output_shape(req.key(), shape);
    rsp.SerializeToString(output);
}

void GroupServerProxy::get_output_shape(void* input_ptr, size_t input_len,
        std::string *output) {
    INFO_INIT(mm_handler, GetOutputShape);
    auto shape = m_mgr.get_output_shape(req.key());
    auto&& shape_proto = *rsp.mutable_shape();
    shape_proto.set_ndim(shape.ndim);
    for (size_t i = 0; i < shape.ndim; ++i) {
        shape_proto.add_shape(shape[i]);
    }
    rsp.SerializeToString(output);
}

void GroupServerProxy::gather_uid(void* input_ptr, size_t input_len,
        std::string *output) {
    INFO_INIT(mm_handler, GatherUid);
    auto uid = req.uid();
    auto uids = m_mgr.gather_uid(uid, req.key(), req.size(), req.rank());
    for (size_t i = 0;i < uids.size();i++) {
        rsp.add_uids();
        rsp.set_uids(i, uids[i].data(), uids[i].size());
    }
    rsp.SerializeToString(output);
}

void GroupServerProxy::group_barrier(void* input_ptr, size_t input_len,
        std::string *output) {
    INFO_INIT(mm_handler, GroupBarrier);
    uint32_t rsp_size = m_mgr.group_barrier(req.size(), req.rank());
    rsp.set_size(rsp_size);
    rsp.SerializeToString(output);
}
#undef INFO_INIT

/* ======================== GroupClientProxy ========================== */

#define INFO_INIT(space, f_name, name)                       \
    using Request = space::name##Request;   \
    using Response = space::name##Response; \
    std::string func_name = #f_name;                  \
    Request req;                                      \
    Response rsp;

#define SOLVE_REQUEST(name, req, rsp)                              \
    std::string req_str;                                           \
    mgb_assert(req.SerializeToString(&req_str));                   \
    zmq::message_t send(req_str.length() + name.length() + 1);     \
    zmq::message_t recv;                                           \
    memcpy(send.data(), name.data(), name.length() + 1);           \
    memcpy((char*)send.data() + name.length() + 1, req_str.data(), \
           req_str.length());                                      \
    m_stub->request(send, recv);                                   \
    mgb_assert(rsp.ParseFromArray(recv.data(), recv.size()));

uint64_t GroupClientProxy::opr_register(const std::string& key, size_t nr_devices,
    uint32_t rank, uintptr_t stream) {
    INFO_INIT(mm_handler, opr_register, OprRegister)
    req.set_key(key);
    req.set_rank(rank);
    req.set_stream(stream);
    req.set_nr_expected_devices(nr_devices);
    SOLVE_REQUEST(func_name, req, rsp);
    return rsp.hash();
}

void GroupClientProxy::set_output_shape(const std::string& key,
                                          const TensorShape& shape) {
    INFO_INIT(mm_handler, set_output_shape, SetOutputShape)
    req.set_key(key);
    auto&& shape_proto = *req.mutable_shape();
    shape_proto.set_ndim(shape.ndim);
    for (size_t i = 0; i < shape.ndim; ++i) {
        shape_proto.add_shape(shape[i]);
    }
    SOLVE_REQUEST(func_name, req, rsp);
}

TensorShape GroupClientProxy::get_output_shape(const std::string& key) {
    INFO_INIT(mm_handler, get_output_shape, GetOutputShape)
    req.set_key(key);
    SOLVE_REQUEST(func_name, req, rsp);
    TensorShape shape;
    shape.ndim = rsp.shape().ndim();
    for (size_t i = 0; i < shape.ndim; ++i) {
        shape[i] = rsp.shape().shape(i);
    }
    return shape;
}
std::vector<std::string> GroupClientProxy::gather_uid(const std::string& uid,
        const std::string& key, uint32_t size, uint32_t rank) {
    INFO_INIT(mm_handler, gather_uid, GatherUid);
    req.set_uid(uid.data(), uid.size());
    req.set_key(key.data(), key.size());
    req.set_size(size);
    req.set_rank(rank);
    SOLVE_REQUEST(func_name, req, rsp);
    std::vector<std::string> rst;
    for (size_t i = 0;i < size;i++) {
        rst.push_back(rsp.uids(i));
    }
    return rst;
}

uint32_t GroupClientProxy::group_barrier(uint32_t size, uint32_t rank) {
    INFO_INIT(mm_handler, group_barrier, GroupBarrier);
    req.set_size(size);
    req.set_rank(rank);
    SOLVE_REQUEST(func_name, req, rsp);
    return rsp.size();
}

#undef INFO_INIT
#undef SOLVE_REQUEST

/* ======================== ZmqRpcServerMgr ========================== */

class ZmqRpcServerMgr {
    struct ServerInfo {
        std::unique_ptr<ZmqRpc::ZmqRpcServer> server;
    };

public:
    int create_zmqrpc_server(const std::string& server_addr, int port,
                           std::unique_ptr<ZmqRpc::ZmqRpcServerImpl> service) {
        MGB_LOCK_GUARD(m_mtx);
        auto server =
                std::make_unique<ZmqRpc::ZmqRpcServer>("tcp://" + server_addr, port,
                                                       std::move(service));
        port = server->port();
        auto full_srv_addr = ssprintf("%s:%d", server_addr.c_str(), port);
        server->run();
        auto ins = m_addr2server.emplace(
                full_srv_addr, ServerInfo{std::move(server)});
        mgb_assert(ins.second);

        return port;
    }

    static ZmqRpcServerMgr* get_zmqrpc_server_mgr() {
        static ZmqRpcServerMgr mgr;
        return &mgr;
    }

private:
    std::unordered_map<std::string, ServerInfo> m_addr2server;
    std::mutex m_mtx;
};

/*! see definition : src/cpp/megbrain_config.h.
 * Create mm server. port 0 is permitted, leave zmqrpc to decide which port
 * should be used.
 */
int _config::create_mm_server(const std::string& server_addr, int port) {
    return ZmqRpcServerMgr::get_zmqrpc_server_mgr()->create_zmqrpc_server(
            server_addr, port, std::make_unique<GroupServerProxy>());
}

#else

int _config::create_mm_server(const std::string& server_addr, int port) {
    mgb_throw(mgb::MegBrainError, "CUDA suppport disable at compile time");
    return 0;
}

#endif

/* ======================== Group Barrier ========================== */

void _config::group_barrier(const std::string& server_addr,
        int port, uint32_t size, uint32_t rank) {
    mgb_assert(rank < size, "invalid rank %d", rank);
    auto group_mgr = std::make_shared<GroupClientProxy>(
            ssprintf("%s:%d", server_addr.c_str(), port));
    uint32_t rsp = group_mgr->group_barrier(size, rank);
    mgb_assert(rsp != 0, "rank already registered: %d", rank);
    mgb_assert(size == rsp, "inconsistent size: %d, expect %d", size, rsp);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
