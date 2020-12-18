/**
 * \file imperative/src/test/io_remote.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/opr/mm_handler.h"

using namespace mgb;
using namespace imperative;

TEST(TestImperative, IORemote) {
    REQUIRE_GPU(2);
    const char* server_addr = "127.0.0.1";
    uint32_t port = 4567;
    mgb_assert(create_zmqrpc_server(server_addr, port) > 0);
    HostTensorGenerator<> gen;
    CompNode cn0 = CompNode::load("gpu0"), cn1 = CompNode::load("gpu1");

    size_t vector_size = 233;
    auto host_x = gen({vector_size}, cn0), host_y = gen({vector_size}, cn1);

    auto expect = gen({vector_size});
    for (size_t i = 0; i < vector_size; ++i) {
        expect->ptr<float>()[i] = host_x->ptr<float>()[i];
    }

    auto run_send = [&](std::shared_ptr<HostTensorND> hnd) {
        auto def = imperative::RemoteSend::make(
            "io_remote_test", server_addr, port, 1);
        auto inp = Tensor::make(*hnd);
        auto oup = OpDef::apply_on_physical_tensor(*def, {inp});
    };

    auto run_recv = [&](std::shared_ptr<HostTensorND> hnd) {
        auto def = imperative::RemoteRecv::make(
            "io_remote_test", server_addr, port, 0,
            CompNode::load("gpu1"), TensorShape{vector_size},
            dtype::Float32());
        auto inp = Tensor::make(*hnd);
        auto oup = OpDef::apply_on_physical_tensor(*def, {inp});
        HostTensorND host_v;
        host_v.copy_from(oup[0]->dev_tensor()).sync();
        MGB_ASSERT_TENSOR_NEAR(*expect, host_v, 1e-6);
    };

    std::thread t0(std::bind(run_send, host_x));
    std::thread t1(std::bind(run_recv, host_y));

    t0.join();
    t1.join();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

// ./imperative_test --gtest_filter TestIORemote
