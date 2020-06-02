#pragma once

#include "megbrain_build_config.h"

#if MGB_CUDA
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>

namespace ZmqRpc {

class ZmqRpcServerImpl {
public:
    virtual void solve_request(zmq::message_t& request,
                               zmq::message_t& reply) = 0;
    virtual ~ZmqRpcServerImpl() = default;
};

class ZmqRpcWorker {
public:
    ZmqRpcWorker() = delete;
    ZmqRpcWorker(zmq::context_t* context, ZmqRpcServerImpl* impl);
    void run();
    void close();

protected:
    void work(std::string uid);
    void add_worker();

private:
    std::vector<std::thread> m_worker_threads;
    std::mutex m_mtx;
    zmq::context_t* m_ctx;
    int m_runable;
    ZmqRpcServerImpl* m_impl;
    bool m_stop = false;
};

class ZmqRpcServer {
public:
    ZmqRpcServer() = delete;
    ZmqRpcServer(std::string address, int port,
                 std::unique_ptr<ZmqRpcServerImpl> impl);
    ~ZmqRpcServer() { close(); }
    void run();
    void close();
    int port() { return m_port; }

protected:
    void work();

private:
    zmq::context_t m_ctx;
    std::unique_ptr<ZmqRpcServerImpl> m_impl;
    std::string m_address;
    int m_port;
    zmq::socket_t m_frontend, m_backend;
    ZmqRpcWorker m_workers;
    std::unique_ptr<std::thread> m_main_thread;
    bool m_stop = false;
};

class ZmqRpcClient {
public:
    ZmqRpcClient() = delete;
    ZmqRpcClient(std::string address);
    void request(zmq::message_t& request, zmq::message_t& reply);
    static ZmqRpcClient* get_client(std::string addr) {
        static std::unordered_map<std::string, std::unique_ptr<ZmqRpcClient>>
                addr2handler;
        static std::mutex mtx;
        std::unique_lock<std::mutex> lk{mtx};
        auto it = addr2handler.emplace(addr, nullptr);
        if (!it.second) {
            assert(it.first->second->m_address == addr);
            return it.first->second.get();
        } else {
            auto handler = std::make_unique<ZmqRpcClient>(addr);
            auto handler_ptr = handler.get();
            it.first->second = std::move(handler);
            return handler_ptr;
        }
    }

private:
    zmq::socket_t* new_socket();
    zmq::socket_t* get_socket();
    void add_socket(zmq::socket_t* socket);
    std::mutex m_queue_mtx;
    std::string m_address;
    zmq::context_t m_ctx;
    std::queue<zmq::socket_t*> m_avaliable_sockets;
    std::vector<std::shared_ptr<zmq::socket_t>> m_own_sockets;
};
}  // namespace ZmqRpc
#endif
