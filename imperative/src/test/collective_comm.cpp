/**
 * \file imperative/src/test/collective_comm.cpp
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

TEST(TestImperative, AllReduceBasic) {
    REQUIRE_GPU(2);
    const char* server_addr = "127.0.0.1";
    uint32_t port = 3456;
    mgb_assert(create_zmqrpc_server(server_addr, port) > 0);
    HostTensorGenerator<> gen;
    CompNode cn0 = CompNode::load("gpu0"),
             cn1 = CompNode::load("gpu1");

    auto host_x = gen({233}, cn0), host_y = gen({233}, cn1);
    auto expect = gen({233});
    for (size_t i = 0; i < 233; ++ i) {
        expect->ptr<float>()[i] = host_x->ptr<float>()[i] + host_y->ptr<float>()[i];
    }

    auto run = [&](std::shared_ptr<HostTensorND> hnd, uint32_t idx) {
        auto def =
            imperative::CollectiveComm::make(
                megdnn::param::CollectiveComm::Mode::ALL_REDUCE_SUM,
                "all_reduce", 2, idx, idx==0, false, server_addr, port,
                dtype::Float32(), "nccl", "");
        auto inp = Tensor::make(*hnd);
        auto oup = OpDef::apply_on_physical_tensor(*def, {inp});
        HostTensorND host_v;
        host_v.copy_from(oup[0]->dev_tensor()).sync();
        MGB_ASSERT_TENSOR_NEAR(*expect, host_v, 1e-6);
    };

    std::thread t0(std::bind(run, host_x, 0));
    std::thread t1(std::bind(run, host_y, 1));

    t0.join();
    t1.join();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
