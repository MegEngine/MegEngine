/**
 * \file src/opr-mm/test/io_remote.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io_remote.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/system.h"
#include "megbrain/test/helper.h"
#include "mock_client.h"

#include <thread>

using namespace mgb;

TEST(TestOprIORemote, Identity) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x = gen({28, 28});
    HostTensorND host_y;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x = opr::Host2DeviceCopy::make(*graph, host_x, cn0);
    auto xr = opr::RemoteSend::make("x", x, client, false);
    auto y = opr::RemoteRecv::make("x", *graph.get(),
                                   client, {cn1}, host_x->shape(),
                                   host_x->dtype());

    auto func = graph->compile({{xr, {}}, make_callback_copy(y, host_y)});

    func->execute();

    MGB_ASSERT_TENSOR_EQ(*host_x, host_y);
}

TEST(TestOprIORemote, IdentityMultiThread) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}, cns[1]);
    HostTensorND host_x_get;
    auto client = std::make_shared<test::MockGroupClient>();

    auto sender = [&]() {
        auto graph = ComputingGraph::make();
        sys::set_thread_name("sender");
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             xr = opr::RemoteSend::make("x", x, client, false);
        auto func = graph->compile({{xr, {}}});
        func->execute();
    };

    auto receiver = [&]() {
        sys::set_thread_name("receiver");
        auto graph = ComputingGraph::make();
        auto x = opr::RemoteRecv::make("x", *graph.get(),
                                       client, {cns[0]}, host_x->shape(),
                                       host_x->dtype());
        auto func = graph->compile({make_callback_copy(x, host_x_get)});
        func->execute();
    };

    std::thread th_send(sender), th_recv(receiver);
    th_send.join();
    th_recv.join();

    MGB_ASSERT_TENSOR_EQ(*host_x, host_x_get);
}

TEST(TestOprIORemote, IdentityWithGopt) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}, cns[1]);
    HostTensorND host_x_get;
    auto client = std::make_shared<test::MockGroupClient>();

    auto sender = [&]() {
        sys::set_thread_name("sender");
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x) * 2 + 1,
             xr = opr::RemoteSend::make("x", x, client, false);
        auto func = graph->compile({{xr, {}}});
        func->execute();
    };

    auto receiver = [&]() {
        sys::set_thread_name("receiver");
        auto graph = ComputingGraph::make();
        auto x = opr::RemoteRecv::make("x", *graph.get(),
                                       client, {cns[0]}, host_x->shape(),
                                       host_x->dtype());
        auto func =
                graph->compile({make_callback_copy((x - 1) / 2, host_x_get)});
        func->execute();
    };

    std::thread th_send(sender), th_recv(receiver);
    th_send.join();
    th_recv.join();

    MGB_ASSERT_TENSOR_EQ(*host_x, host_x_get);
}

TEST(TestOprIORemote, APlusB) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({5, 7}, cns[0]), host_y = gen({5, 1}, cns[0]);
    HostTensorND host_z;
    auto client = std::make_shared<test::MockGroupClient>();

    auto sender = [&]() {
        auto graph = ComputingGraph::make();
        auto z = opr::RemoteRecv::make("z", *graph.get(),
                                       client, {cns[0]}, host_x->shape(),
                                       host_x->dtype());
        auto x = opr::Host2DeviceCopy::make(*graph, host_x).rename("x"),
             y = opr::Host2DeviceCopy::make(*graph, host_y).rename("y"),
             xr = opr::RemoteSend::make("x", x, client, false)
                          .rename("xr"),
             yr = opr::RemoteSend::make("y", y, client, false)
                          .rename("yr");
        auto func = graph->compile(
                {{xr, {}}, {yr, {}}, make_callback_copy(z, host_z)});
        func->to_json()->writeto_fpath(
                output_file("TestOprIORemote.APlusB.json"));
        func->execute();
    };

    auto receiver = [&]() {
        auto graph = ComputingGraph::make();
        auto x = opr::RemoteRecv::make("x", *graph.get(),
                                       client, {cns[1]}, host_x->shape(),
                                       host_x->dtype()),
             y = opr::RemoteRecv::make("y", *graph.get(),
                                       client, {cns[1]}, host_y->shape(),
                                       host_y->dtype()),
             z = x + y,
             zr = opr::RemoteSend::make("z", z, client, false);
        auto func = graph->compile({{zr, {}}});
        func->execute();
    };

    std::thread th_send(sender), th_recv(receiver);
    th_send.join();
    th_recv.join();

    ASSERT_EQ(host_x->shape(), host_z.shape());
    auto px = host_x->ptr<float>(), py = host_y->ptr<float>(),
         pz = host_z.ptr<float>();
    for (size_t i = 0; i < host_x->shape().total_nr_elems(); ++i) {
        ASSERT_FLOAT_EQ(px[i] + py[i / host_x->shape(1)], pz[i]);
    }
}

TEST(TestOprIORemote, SendGrad) {
    auto cns = load_multiple_xpus(2);
    HostTensorGenerator<> gen;
    auto host_x = gen({2, 3}, cns[0]);
    HostTensorND host_gx, host_loss;
    auto client = std::make_shared<test::MockGroupClient>();

    auto sender = [&]() {
        sys::set_thread_name("sender");
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             loss = opr::RemoteSend::make("loss", x, client, false);
        ASSERT_TRUE(!loss.shape().ndim &&
                    loss.node()->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
        loss = opr::RemoteSend::make("loss", x, client, true);
        auto gx = cg::grad(loss, x);
        set_priority(loss, 0);
        set_priority(gx, 1);
        auto func = graph->compile({make_callback_copy(gx, host_gx),
                                    make_callback_copy(loss, host_loss)});
        auto on_opr = [&](cg::OperatorNodeBase* opr) {
            mgb_log_warn("%s", opr->name().c_str());
            return true;
        };
        func->iter_opr_seq(on_opr);
        func->execute();
    };

    auto receiver = [&]() {
        sys::set_thread_name("receiver");
        auto graph = ComputingGraph::make();
        auto x = opr::RemoteRecv::make("loss", *graph.get(),
                                       client, {cns[1]}, host_x->shape(),
                                       host_x->dtype());
        auto y = opr::RemoteSend::make("loss:grad", x + 1, client, false);
        auto func = graph->compile({{y, {}}});
        func->execute();
    };

    std::thread th_send(sender), th_recv(receiver);
    th_send.join();
    th_recv.join();

    ASSERT_EQ(host_x->shape(), host_gx.shape());
    ASSERT_EQ(TensorShape{1}, host_loss.shape());
    ASSERT_FLOAT_EQ(0.f, host_loss.ptr<float>()[0]);

    auto px = host_x->ptr<float>(), pgx = host_gx.ptr<float>();
    for (size_t i = 0; i < 6; ++i) {
        MGB_ASSERT_FLOAT_EQ(px[i] + 1.f, pgx[i]);
    }
}
