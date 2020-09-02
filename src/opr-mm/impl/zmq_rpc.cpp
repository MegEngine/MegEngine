#include "megbrain_build_config.h"

#if MGB_CUDA
#include "megbrain/opr/zmq_rpc.h"
#include "megbrain/common.h"
#include "megbrain/exception.h"

#include <unistd.h>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

using namespace std;
using namespace zmq;
using namespace ZmqRpc;

#define DISCARD_RETVAL MGB_MARK_USED_VAR

ZmqRpcWorker::ZmqRpcWorker(context_t* context, ZmqRpcServerImpl* impl)
        : m_ctx(context), m_runable(0), m_impl(impl) {}

void ZmqRpcWorker::run() {
    add_worker();
}

void ZmqRpcWorker::close() {
    m_stop = true;
    for (auto& thread : m_worker_threads) {
        thread.join();
    }
}

void ZmqRpcWorker::work(string uid) {
    // req work pattern: send recv send recv ...
    zmq::socket_t socket(*m_ctx, ZMQ_REQ);
    socket.setsockopt(ZMQ_IDENTITY, uid.data(), uid.size());
    socket.connect("inproc://workers");

    // send READY to notify server that worker is ready
    zmq::message_t ready(6);
    memcpy(ready.data(), "READY", 6);
    socket.send(ready, send_flags::dontwait);
    while (!m_stop) {
        //  Wait for next request from client
        //  request should be like [address, empty, msg]
        message_t address;
        recv_result_t ret_code;
        while (!m_stop) {
            ret_code = socket.recv(address, recv_flags::dontwait);
            if (ret_code.has_value() && ret_code.value() > 0)
                break;
            // retry after 10 usec
            usleep(10);
        }
        if (m_stop)
            break;
        message_t empty;
        DISCARD_RETVAL(socket.recv(empty));
        assert(empty.size() == 0);
        message_t request;
        DISCARD_RETVAL(socket.recv(request));

        m_mtx.lock();
        if (--m_runable <= 0) {
            add_worker();
        }
        m_mtx.unlock();

        //  Send reply back to client
        //  reply should be like [address, empty, msg]
        zmq::message_t reply;
        m_impl->solve_request(request, reply);

        socket.send(address, send_flags::sndmore);
        socket.send(empty, send_flags::sndmore);
        socket.send(reply, send_flags::dontwait);
        m_mtx.lock();
        ++m_runable;
        m_mtx.unlock();
    }

    socket.close();
}

void ZmqRpcWorker::add_worker() {
    int size = m_worker_threads.size();
    m_worker_threads.emplace_back(
            [this, size] { this->work(to_string(size)); });
    ++m_runable;
}

ZmqRpcServer::ZmqRpcServer(string address, int port,
                           unique_ptr<ZmqRpcServerImpl> impl)
        : m_ctx(1),
          m_impl(std::move(impl)),
          m_address(address),
          m_port(port),
          m_frontend(m_ctx, ZMQ_ROUTER),
          m_backend(m_ctx, ZMQ_ROUTER),
          m_workers(&m_ctx, m_impl.get()) {
    try {
        char full_addr[100];
        size_t size = sizeof(full_addr);
        sprintf(full_addr, "%s:%d", m_address.c_str(), m_port);
        m_frontend.bind(full_addr);
        m_frontend.getsockopt(ZMQ_LAST_ENDPOINT, &full_addr, &size);
        m_port = 0;
        int pow = 1, len = strlen(full_addr);
        for (int i = len - 1; i >= 0; i--) {
            if (full_addr[i] == ':') break;
            m_port += (full_addr[i] - '0') * pow;
            pow *= 10;
        }
    } catch(...) {
        m_port = -1;
    }
    m_backend.bind("inproc://workers");
}

void ZmqRpcServer::run() {
    if(m_port == -1) return;
    m_main_thread = make_unique<thread>([this] { this->work(); });
}

void ZmqRpcServer::close() {
    if(m_port == -1) return;
    m_stop = true;
    if (m_main_thread->joinable())
        m_main_thread->join();
    m_ctx.close();
}

void ZmqRpcServer::work() {
    m_workers.run();
    queue<string> worker_queue;
    while (!m_stop) {
        zmq_pollitem_t items[] = {{m_backend, 0, ZMQ_POLLIN, 0},
                                  {m_frontend, 0, ZMQ_POLLIN, 0}};
        int ret_code = zmq_poll(items, !worker_queue.empty() ? 2 : 1, 10);
        if (ret_code == -1)
            continue;
        if (items[0].revents & ZMQ_POLLIN) {
            message_t address;

            DISCARD_RETVAL(m_backend.recv(address));
            worker_queue.push({(char*)address.data(), address.size()});

            message_t empty;
            DISCARD_RETVAL(m_backend.recv(empty));
            assert(empty.size() == 0);

	    // the third frame is READY or a client address
            message_t client_address;
            DISCARD_RETVAL(m_backend.recv(client_address));
            string tmp((char*)client_address.data(), client_address.size());
            if (strcmp(tmp.c_str(), "READY") != 0) {
                empty.rebuild();
                DISCARD_RETVAL(m_backend.recv(empty));
                assert(empty.size() == 0);

                message_t respones;
                DISCARD_RETVAL(m_backend.recv(respones));
                m_frontend.send(client_address, send_flags::sndmore);
                m_frontend.send(empty, send_flags::sndmore);
                m_frontend.send(respones, send_flags::dontwait);
            }
        }
        if (items[1].revents & ZMQ_POLLIN) {
            message_t address;
            DISCARD_RETVAL(m_frontend.recv(address));

            message_t empty;
            DISCARD_RETVAL(m_frontend.recv(empty));
            assert(empty.size() == 0);

            message_t request;
            DISCARD_RETVAL(m_frontend.recv(request));

            string worker_uid = worker_queue.front();
            worker_queue.pop();

            message_t uid(worker_uid.data(), worker_uid.length());
            m_backend.send(uid, send_flags::sndmore);
            m_backend.send(empty, send_flags::sndmore);
            m_backend.send(address, send_flags::sndmore);
            m_backend.send(empty, send_flags::sndmore);
            m_backend.send(request, send_flags::dontwait);
        }
    }
    m_workers.close();
    m_frontend.close();
    m_backend.close();
}

ZmqRpcClient::ZmqRpcClient(string address) : m_address(address), m_ctx(1) {}

socket_t* ZmqRpcClient::new_socket() {
    m_own_sockets.emplace_back(make_unique<socket_t>(m_ctx, ZMQ_REQ));
    socket_t* ptr = m_own_sockets.back().get();
    ptr->connect(m_address);
    return ptr;
}

socket_t* ZmqRpcClient::get_socket() {
    unique_lock<mutex> lk{m_queue_mtx};
    if (m_avaliable_sockets.empty()) {
        return new_socket();
    }
    socket_t* ptr = m_avaliable_sockets.front();
    m_avaliable_sockets.pop();
    return ptr;
}

void ZmqRpcClient::add_socket(socket_t* socket) {
    unique_lock<mutex> lk{m_queue_mtx};
    m_avaliable_sockets.push(socket);
}

void ZmqRpcClient::request(message_t& request, message_t& reply) {
    socket_t* client = get_socket();
    client->send(request, send_flags::dontwait);
    DISCARD_RETVAL(client->recv(reply));
    add_socket(client);
}
#endif // MGB_CUDA
