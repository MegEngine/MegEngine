/**
 * \file src/opr-mm/test/collective_comm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/collective_comm.h"
#include "megbrain/graph.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/helper.h"
#include "mock_client.h"

using namespace mgb;

using Mode = opr::CollectiveComm::Param::Mode;

SymbolVar make_all_reduce_output(const Mode mode,
                                 const SymbolVarArray& inputs) {
    if (mode == Mode::ALL_REDUCE_MAX)
        return opr::Elemwise::make(inputs, opr::Elemwise::Mode::MAX);
    if (mode == Mode::ALL_REDUCE_MIN)
        return opr::Elemwise::make(inputs, opr::Elemwise::Mode::MIN);
    if (mode == Mode::ALL_REDUCE_SUM)
        return opr::Elemwise::make(inputs, opr::Elemwise::Mode::ADD);
    mgb_assert(false);
}

SymbolVarArray make_reduce_scatter_sum_output(const SymbolVarArray& inputs) {
    auto rdc = opr::Elemwise::make(inputs, opr::Elemwise::Mode::ADD);
    return opr::Split::make(
            rdc, opr::Split::Options::make_average(0, inputs.size()));
}

TEST(TestOprCollectiveComm, AllReduce) {
    REQUIRE_GPU(2);

    auto run_mode = [](const Mode mode) {
        auto cn0 = CompNode::load("gpu0");
        auto cn1 = CompNode::load("gpu1");

        HostTensorGenerator<> gen;
        auto host_x0 = gen({28, 28});
        auto host_x1 = gen({28, 28});
        HostTensorND host_y0, host_y1, host_y_expect;

        auto client = std::make_shared<test::MockGroupClient>();
        auto graph = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1, cn0);
        auto x1c = opr::Copy::make(x1, cn1);

        auto y0 = opr::CollectiveComm::make({x0}, graph.get(), "all_reduce", 2,
                                            false, 0, false, client, {mode},
                                            dtype::Float32(), "nccl")[0];
        auto y1 = opr::CollectiveComm::make({x1c}, graph.get(), "all_reduce", 2,
                                            false, 1, false, client, {mode},
                                            dtype::Float32(), "nccl")[0];
        auto y_expect = make_all_reduce_output(mode, {x0, x1});

        auto func =
                graph->compile({make_callback_copy(y0, host_y0),
                                make_callback_copy(y1, host_y1),
                                make_callback_copy(y_expect, host_y_expect)});
        func->execute();

        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
    };

    run_mode(Mode::ALL_REDUCE_MAX);
    run_mode(Mode::ALL_REDUCE_MIN);
    run_mode(Mode::ALL_REDUCE_SUM);
}

TEST(TestOprCollectiveComm, AllReduceMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    auto run_mode = [&](const Mode mode) {
        HostTensorGenerator<> gen;
        auto host_x0 = gen({28, 28});
        auto host_x1 = gen({28, 28});
        HostTensorND host_y0, host_y1, host_y_expect;

        auto client = std::make_shared<test::MockGroupClient>();

        auto run_0 = [&]() {
            auto graph0 = ComputingGraph::make();
            auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0);
            auto y0 = opr::CollectiveComm::make(
                    {x0}, graph0.get(), "all_reduce", 2, false, 0, false,
                    client, {mode}, dtype::Float32(), "nccl")[0];
            auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
            func0->execute();
        };

        auto run_1 = [&]() {
            auto graph1 = ComputingGraph::make();
            auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
            auto y1 = opr::CollectiveComm::make(
                    {x1}, graph1.get(), "all_reduce", 2, false, 1, false,
                    client, {mode}, dtype::Float32(), "nccl")[0];
            auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
            func1->execute();
        };

        auto run_2 = [&]() {
            auto graph2 = ComputingGraph::make();
            auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
            auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
            auto y_expect = make_all_reduce_output(mode, {x0, x1});
            auto func2 = graph2->compile(
                    {make_callback_copy(y_expect, host_y_expect)});
            func2->execute();
        };

        std::thread t0(run_0);
        std::thread t1(run_1);
        std::thread t2(run_2);

        t0.join();
        t1.join();
        t2.join();

        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
        MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
    };

    run_mode(Mode::ALL_REDUCE_MAX);
    run_mode(Mode::ALL_REDUCE_MIN);
    run_mode(Mode::ALL_REDUCE_SUM);
}

TEST(TestOprCollectiveComm, AllReduceWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({10});
    auto host_x0 = gen(shape);
    auto host_x1 = gen(shape);
    auto host_grad0 = gen(shape);
    auto host_grad1 = gen(shape);

    HostTensorND host_y0, host_y1, host_y_expect;
    HostTensorND host_out_grad0, host_out_grad1, host_out_grad_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "all_reduce", 2, false, 0, false, client,
                {Mode::ALL_REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "all_reduce", 2, false, 1, false, client,
                {Mode::ALL_REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1),
                                      make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = make_all_reduce_output(Mode::ALL_REDUCE_SUM, {x0, x1});

        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect =
                make_all_reduce_output(Mode::ALL_REDUCE_SUM, {grad0, grad1});

        auto func2 = graph2->compile(
                {make_callback_copy(y_expect, host_y_expect),
                 make_callback_copy(out_grad_expect, host_out_grad_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect, host_out_grad1);
}

TEST(TestOprCollectiveComm, AllReduceWithGradThisNodeOnly) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({10});
    auto host_x0 = gen(shape);
    auto host_x1 = gen(shape);
    auto host_grad0 = gen(shape);
    auto host_grad1 = gen(shape);

    HostTensorND host_y0, host_y1, host_y_expect;
    HostTensorND host_out_grad0, host_out_grad1, host_out_grad_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "all_reduce", 2, false, 0, true, client,
                {Mode::ALL_REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "all_reduce", 2, false, 1, true, client,
                {Mode::ALL_REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1),
                                      make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = make_all_reduce_output(Mode::ALL_REDUCE_SUM, {x0, x1});

        auto func2 =
                graph2->compile({make_callback_copy(y_expect, host_y_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
    MGB_ASSERT_TENSOR_EQ(*host_grad0, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(*host_grad1, host_out_grad1);
}

TEST(TestOprCollectiveComm, AllGather) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
    auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1, cn0);
    auto x1c = opr::Copy::make(x1, cn1);

    auto y0 = opr::CollectiveComm::make(
            {x0}, graph.get(), "all_gather", 2, false, 0, false, client,
            {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
    auto y1 = opr::CollectiveComm::make(
            {x1c}, graph.get(), "all_gather", 2, false, 1, false, client,
            {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
    auto y_expect = opr::Concat::make({x0, x1}, 0);

    auto func = graph->compile({make_callback_copy(y0, host_y0),
                                make_callback_copy(y1, host_y1),
                                make_callback_copy(y_expect, host_y_expect)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
}

TEST(TestOprCollectiveComm, AllGatherMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "all_gather", 2, false, 0, false, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "all_gather", 2, false, 1, false, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = opr::Concat::make({x0, x1}, 0);
        auto func2 =
                graph2->compile({make_callback_copy(y_expect, host_y_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
}

TEST(TestOprCollectiveComm, AllGatherWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({10});
    auto host_x1 = gen({10});
    auto host_grad0 = gen({20});
    auto host_grad1 = gen({20});

    HostTensorND host_y0, host_y1, host_y_expect;
    HostTensorND host_out_grad0, host_out_grad1;
    HostTensorND host_out_grad0_expect, host_out_grad1_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "all_gather", 2, false, 0, false, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "all_gather", 2, false, 1, false, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1),
                                      make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = opr::Concat::make({x0, x1}, 0);

        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect = make_reduce_scatter_sum_output({grad0, grad1});

        auto func2 = graph2->compile(
                {make_callback_copy(y_expect, host_y_expect),
                 make_callback_copy(out_grad_expect[0], host_out_grad0_expect),
                 make_callback_copy(out_grad_expect[1],
                                    host_out_grad1_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad0_expect, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(host_out_grad1_expect, host_out_grad1);
}

TEST(TestOprCollectiveComm, AllGatherWithGradThisNodeOnly) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({10});
    auto host_x1 = gen({10});
    auto host_grad0 = gen({20});
    auto host_grad1 = gen({20});

    HostTensorND host_y0, host_y1, host_y_expect;
    HostTensorND host_out_grad0, host_out_grad1;
    HostTensorND host_out_grad0_expect, host_out_grad1_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "all_gather", 2, false, 0, true, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "all_gather", 2, false, 1, true, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1),
                                      make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = opr::Concat::make({x0, x1}, 0);

        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);

        opr::Subtensor::IndexDesc axis0;
        auto shape0 = opr::GetVarShape::make(grad0, 0);
        axis0.push_back({0, 0, shape0 / 2});
        auto out_grad0_expect = opr::Subtensor::make(grad0, axis0);

        opr::Subtensor::IndexDesc axis1;
        axis1.push_back({0, shape0 / 2});
        auto out_grad1_expect = opr::Subtensor::make(grad1, axis1);

        auto func2 = graph2->compile(
                {make_callback_copy(y_expect, host_y_expect),
                 make_callback_copy(out_grad0_expect, host_out_grad0_expect),
                 make_callback_copy(out_grad1_expect, host_out_grad1_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad0_expect, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(host_out_grad1_expect, host_out_grad1);
}

TEST(TestOprCollectiveComm, ReduceScatterSum) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y0_expect, host_y1_expect;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
    auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1, cn0);
    auto x1c = opr::Copy::make(x1, cn1);

    auto y0 = opr::CollectiveComm::make(
            {x0}, graph.get(), "reduce_scatter_sum", 2, false, 0, false, client,
            {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(), "nccl")[0];
    auto y1 = opr::CollectiveComm::make(
            {x1c}, graph.get(), "reduce_scatter_sum", 2, false, 1, false,
            client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(), "nccl")[0];
    auto y_expect = make_reduce_scatter_sum_output({x0, x1});

    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1),
             make_callback_copy(y_expect[0], host_y0_expect),
             make_callback_copy(y_expect[1], host_y1_expect)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_y0_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y1_expect, host_y1);
}

TEST(TestOprCollectiveComm, ReduceScatterSumMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({8});
    auto host_x1 = gen({8});
    HostTensorND host_y0, host_y1, host_y0_expect, host_y1_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "reduce_scatter_sum", 2, false, 0, false,
                client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(),
                "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "reduce_scatter_sum", 2, false, 1, false,
                client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(),
                "nccl")[0];
        auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = make_reduce_scatter_sum_output({x0, x1});
        auto func = graph2->compile(
                {make_callback_copy(y_expect[0], host_y0_expect),
                 make_callback_copy(y_expect[1], host_y1_expect)});
        func->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y0_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y1_expect, host_y1);
}

TEST(TestOprCollectiveComm, ReduceScatterSumWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({20});
    auto host_x1 = gen({20});
    auto host_grad0 = gen({10});
    auto host_grad1 = gen({10});

    HostTensorND host_y0, host_y1, host_y0_expect, host_y1_expect;
    HostTensorND host_out_grad0, host_out_grad1, host_out_grad_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "reduce_scatter_sum", 2, false, 0, false,
                client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(),
                "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "reduce_scatter_sum", 2, false, 1, false,
                client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(),
                "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1),
                                      make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = make_reduce_scatter_sum_output({x0, x1});

        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect = opr::Concat::make({grad0, grad1}, 0);

        auto func2 = graph2->compile(
                {make_callback_copy(y_expect[0], host_y0_expect),
                 make_callback_copy(y_expect[1], host_y1_expect),
                 make_callback_copy(out_grad_expect, host_out_grad_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y0_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y1_expect, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect, host_out_grad1);
}

TEST(TestOprCollectiveComm, ReduceScatterSumWithGradThisNodeOnly) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    HostTensorGenerator<> zeros(0, 0);
    auto host_x0 = gen({20});
    auto host_x1 = gen({20});
    auto host_grad0 = gen({10});
    auto host_grad1 = gen({10});
    auto host_zero_grad = zeros({10});

    HostTensorND host_y0, host_y1, host_y0_expect, host_y1_expect;
    HostTensorND host_out_grad0, host_out_grad1, host_out_grad_expect0,
            host_out_grad_expect1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "reduce_scatter_sum", 2, false, 0, true,
                client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(),
                "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "reduce_scatter_sum", 2, false, 1, true,
                client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(),
                "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1),
                                      make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = make_reduce_scatter_sum_output({x0, x1});

        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto zero_grad =
                opr::Host2DeviceCopy::make(*graph2, host_zero_grad, cn0);
        auto out_grad_expect0 = opr::Concat::make({grad0, zero_grad}, 0);
        auto out_grad_expect1 = opr::Concat::make({zero_grad, grad1}, 0);

        auto func2 = graph2->compile(
                {make_callback_copy(y_expect[0], host_y0_expect),
                 make_callback_copy(y_expect[1], host_y1_expect),
                 make_callback_copy(out_grad_expect0, host_out_grad_expect0),
                 make_callback_copy(out_grad_expect1, host_out_grad_expect1)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y0_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_y1_expect, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect0, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect1, host_out_grad1);
}

TEST(TestOprCollectiveComm, ReduceSum) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
    auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1, cn0);
    auto x1c = opr::Copy::make(x1, cn1);

    auto y0 = opr::CollectiveComm::make(
            {x0}, graph.get(), "reduce_sum", 2, true, 0, false, client,
            {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
    auto y1 = opr::CollectiveComm::make(
            {x1c}, graph.get(), "reduce_sum", 2, false, 1, false, client,
            {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
    auto y_expect = x0 + x1;

    auto func = graph->compile({make_callback_copy(y0, host_y0),
                                make_callback_copy(y1, host_y1),
                                make_callback_copy(y_expect, host_y_expect)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
}

TEST(TestOprCollectiveComm, ReduceSumMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "reduce", 2, true, 0, false, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "reduce", 2, false, 1, false, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile({{y1, nullptr}});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = x0 + x1;
        auto func2 =
                graph2->compile({make_callback_copy(y_expect, host_y_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
}

TEST(TestOprCollectiveComm, ReduceSumWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({28, 28});
    auto host_x0 = gen(shape);
    auto host_x1 = gen(shape);
    auto host_grad = gen(shape);

    HostTensorND host_y0, host_y0_expect, host_out_grad0, host_out_grad1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "reduce", 2, true, 0, false, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad = opr::Host2DeviceCopy::make(*graph0, host_grad, cn0);
        auto loss = opr::Dot::make(y0, grad);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "reduce", 2, false, 1, false, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad = opr::Host2DeviceCopy::make(*graph1, gen({1}), cn1);
        auto loss = opr::Dot::make(y1, grad);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile(
                {{y1, nullptr}, make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y0_expect = x0 + x1;
        auto func2 = graph2->compile(
                {make_callback_copy(y0_expect, host_y0_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y0_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_grad, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(*host_grad, host_out_grad1);
}

TEST(TestOprCollectiveComm, Gather) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
    auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1, cn0);
    auto x1c = opr::Copy::make(x1, cn1);

    auto y0 = opr::CollectiveComm::make({x0}, graph.get(), "gather", 2, true, 0,
                                        false, client, {Mode::GATHER},
                                        dtype::Float32(), "nccl")[0];
    auto y1 = opr::CollectiveComm::make({x1c}, graph.get(), "gather", 2, false,
                                        1, false, client, {Mode::GATHER},
                                        dtype::Float32(), "nccl")[0];
    auto y_expect = opr::Concat::make({x0, x1}, 0);

    auto func = graph->compile({make_callback_copy(y0, host_y0),
                                make_callback_copy(y1, host_y1),
                                make_callback_copy(y_expect, host_y_expect)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
}

TEST(TestOprCollectiveComm, GatherMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "gather", 2, true, 0, false, client,
                {Mode::GATHER}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "gather", 2, false, 1, false, client,
                {Mode::GATHER}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile({{y1, nullptr}});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = opr::Concat::make({x0, x1}, 0);
        auto func2 =
                graph2->compile({make_callback_copy(y_expect, host_y_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y_expect, host_y0);
}

TEST(TestOprCollectiveComm, GatherWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({28, 28});
    auto host_x0 = gen(shape);
    auto host_x1 = gen(shape);
    auto host_grad0 = gen(shape);
    auto host_grad1 = gen(shape);

    HostTensorND host_y0, host_y0_expect, host_out_grad0, host_out_grad1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "gather", 2, true, 0, false, client,
                {Mode::GATHER}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph0, host_grad1, cn0);
        auto grad = opr::Concat::make({grad0, grad1}, 0);
        auto loss = opr::Dot::make(y0, grad);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "gather", 2, false, 1, false, client,
                {Mode::GATHER}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad = opr::Host2DeviceCopy::make(*graph1, gen({1}), cn1);
        auto loss = opr::Dot::make(y1, grad);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile(
                {{y1, nullptr}, make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y0_expect = opr::Concat::make({x0, x1}, 0);
        auto func2 = graph2->compile(
                {make_callback_copy(y0_expect, host_y0_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_y0_expect, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_grad0, host_out_grad0);
    MGB_ASSERT_TENSOR_EQ(*host_grad1, host_out_grad1);
}

TEST(TestOprCollectiveComm, Broadcast) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y_expect;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
    auto y0 = opr::CollectiveComm::make({x0}, graph.get(), "broadcast", 2, true,
                                        0, false, client, {Mode::BROADCAST},
                                        dtype::Float32(), "nccl")[0];
    auto y_dev =
            std::make_shared<DeviceTensorND>(DeviceTensorND()
                                                     .comp_node(cn1)
                                                     .dtype(dtype::Float32())
                                                     .resize(host_x0->shape()));
    auto y1 = opr::CollectiveComm::make(
            {}, graph.get(), "broadcast", 2, false, 1, false, client, {y_dev},
            {Mode::BROADCAST}, dtype::Float32(), "nccl", {cn1})[0];

    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y1);
}

TEST(TestOprCollectiveComm, BroadcastMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    HostTensorND host_y0, host_y1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "broadcast", 2, true, 0, false, client,
                {Mode::BROADCAST}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto y_dev = std::make_shared<DeviceTensorND>(
                DeviceTensorND()
                        .comp_node(cn1)
                        .dtype(dtype::Float32())
                        .resize(host_x0->shape()));
        auto y1 = opr::CollectiveComm::make(
                {}, graph1.get(), "broadcast", 2, false, 1, false, client,
                {y_dev}, {Mode::BROADCAST}, dtype::Float32(), "nccl", {cn1})[0];
        auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
        func1->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);

    t0.join();
    t1.join();

    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y1);
}

TEST(TestOprCollectiveComm, BroadcastWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({28, 28});
    auto host_x0 = gen(shape);
    auto host_grad0 = gen(shape);
    auto host_grad1 = gen(shape);

    HostTensorND host_y0, host_y1, host_out_grad, host_out_grad_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "broadcast", 2, true, 0, false, client,
                {Mode::BROADCAST}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto y1 = opr::CollectiveComm::make(
                {}, graph1.get(), "broadcast", 2, false, 1, false, client,
                {Mode::BROADCAST}, dtype::Float32(), "nccl", {cn1})[0];

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto g = opr::CollectiveComm::make(
                {grad1}, graph1.get(), "broadcast:grad", 2, false, 1, false,
                client, Mode::REDUCE_SUM, dtype::Float32(), "nccl")[0];
        g.node()->owner_opr()->node_prop().attribute().priority = 1;

        auto func1 = graph1->compile(
                {make_callback_copy(y1, host_y1), {g, nullptr}});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect = grad0 + grad1;
        auto func2 = graph2->compile(
                {make_callback_copy(out_grad_expect, host_out_grad_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect, host_out_grad);
}

TEST(TestOprCollectiveComm, Scatter) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x0 = opr::Host2DeviceCopy::make(*graph, host_x0, cn0);
    auto x1 = opr::Host2DeviceCopy::make(*graph, host_x1, cn0);
    auto x = opr::Concat::make({x0, x1}, 0);
    auto y0 = opr::CollectiveComm::make({x}, graph.get(), "scatter", 2, true, 0,
                                        false, client, {Mode::SCATTER},
                                        dtype::Float32(), "nccl")[0];
    auto y1 = opr::CollectiveComm::make({}, graph.get(), "scatter", 2, false, 1,
                                        false, client, {Mode::SCATTER},
                                        dtype::Float32(), "nccl", {cn1})[0];

    auto func = graph->compile(
            {make_callback_copy(y0, host_y0), make_callback_copy(y1, host_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x1, host_y1);
}

TEST(TestOprCollectiveComm, ScatterMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph0, host_x1, cn0);
        auto x = opr::Concat::make({x0, x1}, 0);
        auto y0 = opr::CollectiveComm::make(
                {x}, graph0.get(), "scatter", 2, true, 0, false, client,
                {Mode::SCATTER}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto y1 = opr::CollectiveComm::make(
                {}, graph1.get(), "scatter", 2, false, 1, false, client,
                {Mode::SCATTER}, dtype::Float32(), "nccl", {cn1})[0];
        auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
        func1->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);

    t0.join();
    t1.join();

    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x1, host_y1);
}

TEST(TestOprCollectiveComm, ScatterWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({28, 28});
    auto host_x0 = gen(shape);
    auto host_x1 = gen(shape);
    auto host_grad0 = gen(shape);
    auto host_grad1 = gen(shape);

    HostTensorND host_y0, host_y1, host_out_grad, host_out_grad_expect;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph0, host_x1, cn0);
        auto x = opr::Concat::make({x0, x1}, 0);
        auto y0 = opr::CollectiveComm::make(
                {x}, graph0.get(), "scatter", 2, true, 0, false, client,
                {Mode::SCATTER}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x);

        auto func0 = graph0->compile({make_callback_copy(y0, host_y0),
                                      make_callback_copy(g, host_out_grad)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto y1 = opr::CollectiveComm::make(
                {}, graph1.get(), "scatter", 2, false, 1, false, client,
                {Mode::SCATTER}, dtype::Float32(), "nccl", {cn1})[0];

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto g = opr::CollectiveComm::make(
                {grad1}, graph1.get(), "scatter:grad", 2, false, 1, false,
                client, Mode::GATHER, dtype::Float32(), "nccl")[0];
        g.node()->owner_opr()->node_prop().attribute().priority = 1;

        auto func1 = graph1->compile(
                {make_callback_copy(y1, host_y1), {g, nullptr}});
        func1->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect = opr::Concat::make({grad0, grad1}, 0);
        auto func2 = graph2->compile(
                {make_callback_copy(out_grad_expect, host_out_grad_expect)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(*host_x0, host_y0);
    MGB_ASSERT_TENSOR_EQ(*host_x1, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_out_grad_expect, host_out_grad);
}

TEST(TestOprCollectiveComm, AllToAll) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({10});
    auto host_x00 = gen(shape);
    auto host_x01 = gen(shape);
    auto host_x10 = gen(shape);
    auto host_x11 = gen(shape);
    HostTensorND host_y0, host_y1, host_expect_y0, host_expect_y1;

    auto client = std::make_shared<test::MockGroupClient>();
    auto graph = ComputingGraph::make();

    auto x00 = opr::Host2DeviceCopy::make(*graph, host_x00, cn0);
    auto x01 = opr::Host2DeviceCopy::make(*graph, host_x01, cn0);
    auto x0 = opr::Concat::make({x00, x01}, 0);
    auto x10 = opr::Host2DeviceCopy::make(*graph, host_x10, cn1);
    auto x11 = opr::Host2DeviceCopy::make(*graph, host_x11, cn1);
    auto x1 = opr::Concat::make({x10, x11}, 0);

    auto x01c = opr::Copy::make(x01, {cn1});
    auto x10c = opr::Copy::make(x10, {cn0});

    auto expect_y0 = opr::Concat::make({x00, x10c}, 0);
    auto expect_y1 = opr::Concat::make({x01c, x11}, 0);

    auto y0 = opr::CollectiveComm::make({x0}, graph.get(), "alltoall", 2, false,
                                        0, false, client, {Mode::ALL_TO_ALL},
                                        dtype::Float32(), "nccl")[0];
    auto y1 = opr::CollectiveComm::make({x1}, graph.get(), "alltoall", 2, false,
                                        1, false, client, {Mode::ALL_TO_ALL},
                                        dtype::Float32(), "nccl")[0];

    auto func = graph->compile({make_callback_copy(y0, host_y0),
                                make_callback_copy(y1, host_y1),
                                make_callback_copy(expect_y0, host_expect_y0),
                                make_callback_copy(expect_y1, host_expect_y1)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_expect_y0, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_expect_y1, host_y1);
}

TEST(TestOprCollectiveComm, AllToAllMultiThread) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({10});
    auto host_x00 = gen(shape);
    auto host_x01 = gen(shape);
    auto host_x10 = gen(shape);
    auto host_x11 = gen(shape);
    HostTensorND host_y0, host_y1, host_expect_y0, host_expect_y1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        auto x00 = opr::Host2DeviceCopy::make(*graph0, host_x00, cn0);
        auto x01 = opr::Host2DeviceCopy::make(*graph0, host_x01, cn0);
        auto x10 = opr::Host2DeviceCopy::make(*graph0, host_x10, cn0);
        auto x0 = opr::Concat::make({x00, x01}, 0);
        auto expect_y0 = opr::Concat::make({x00, x10}, 0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "alltoall", 2, false, 0, false, client,
                {Mode::ALL_TO_ALL}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile(
                {make_callback_copy(y0, host_y0),
                 make_callback_copy(expect_y0, host_expect_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        auto x10 = opr::Host2DeviceCopy::make(*graph1, host_x10, cn1);
        auto x11 = opr::Host2DeviceCopy::make(*graph1, host_x11, cn1);
        auto x01 = opr::Host2DeviceCopy::make(*graph1, host_x01, cn1);
        auto x1 = opr::Concat::make({x10, x11}, 0);
        auto expect_y1 = opr::Concat::make({x01, x11}, 0);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "alltoall", 2, false, 1, false, client,
                {Mode::ALL_TO_ALL}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile(
                {make_callback_copy(y1, host_y1),
                 make_callback_copy(expect_y1, host_expect_y1)});
        func1->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);

    t0.join();
    t1.join();

    MGB_ASSERT_TENSOR_EQ(host_expect_y0, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_expect_y1, host_y1);
}

TEST(TestOprCollectiveComm, AllToAllWithGrad) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    TensorShape shape({10});
    auto host_x00 = gen(shape);
    auto host_x01 = gen(shape);
    auto host_x10 = gen(shape);
    auto host_x11 = gen(shape);
    auto host_grad00 = gen(shape);
    auto host_grad01 = gen(shape);
    auto host_grad10 = gen(shape);
    auto host_grad11 = gen(shape);

    HostTensorND host_y0, host_y1, host_expect_y0, host_expect_y1, host_grad0,
            host_grad1, host_expect_grad0, host_expect_grad1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x00 = opr::Host2DeviceCopy::make(*graph0, host_x00, cn0);
        auto x01 = opr::Host2DeviceCopy::make(*graph0, host_x01, cn0);
        auto x10 = opr::Host2DeviceCopy::make(*graph0, host_x10, cn0);
        auto x0 = opr::Concat::make({x00, x01}, 0);
        auto expect_y0 = opr::Concat::make({x00, x10}, 0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "alltoall", 2, false, 0, false, client,
                {Mode::ALL_TO_ALL}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad00 = opr::Host2DeviceCopy::make(*graph0, host_grad00, cn0);
        auto grad10 = opr::Host2DeviceCopy::make(*graph0, host_grad10, cn0);
        auto grad_y0 = opr::Concat::make({grad00, grad10}, 0);
        auto loss = opr::Dot::make(y0, grad_y0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
                {make_callback_copy(y0, host_y0),
                 make_callback_copy(g, host_grad0),
                 make_callback_copy(expect_y0, host_expect_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x10 = opr::Host2DeviceCopy::make(*graph1, host_x10, cn1);
        auto x11 = opr::Host2DeviceCopy::make(*graph1, host_x11, cn1);
        auto x01 = opr::Host2DeviceCopy::make(*graph1, host_x01, cn1);
        auto x1 = opr::Concat::make({x10, x11}, 0);
        auto expect_y1 = opr::Concat::make({x01, x11}, 0);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "alltoall", 2, false, 1, false, client,
                {Mode::ALL_TO_ALL}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad01 = opr::Host2DeviceCopy::make(*graph1, host_grad01, cn1);
        auto grad11 = opr::Host2DeviceCopy::make(*graph1, host_grad11, cn1);
        auto grad_y1 = opr::Concat::make({grad01, grad11}, 0);
        auto loss = opr::Dot::make(y1, grad_y1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func0 = graph1->compile(
                {make_callback_copy(y1, host_y1),
                 make_callback_copy(g, host_grad1),
                 make_callback_copy(expect_y1, host_expect_y1)});
        func0->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto grad00 = opr::Host2DeviceCopy::make(*graph2, host_grad00, cn0);
        auto grad01 = opr::Host2DeviceCopy::make(*graph2, host_grad01, cn0);
        auto grad10 = opr::Host2DeviceCopy::make(*graph2, host_grad10, cn0);
        auto grad11 = opr::Host2DeviceCopy::make(*graph2, host_grad11, cn0);
        auto out_grad0_expect = opr::Concat::make({grad00, grad01}, 0);
        auto out_grad1_expect = opr::Concat::make({grad10, grad11}, 0);
        auto func2 = graph2->compile(
                {make_callback_copy(out_grad0_expect, host_expect_grad0),
                 make_callback_copy(out_grad1_expect, host_expect_grad1)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_expect_y0, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_expect_y1, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_expect_grad0, host_grad0);
    MGB_ASSERT_TENSOR_EQ(host_expect_grad1, host_grad1);
}

TEST(TestOprCollectiveComm, AllToAllWithGradThisNodeOnly) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    HostTensorGenerator<> zeros(0, 0);
    TensorShape shape({10});
    auto host_x00 = gen(shape);
    auto host_x01 = gen(shape);
    auto host_x10 = gen(shape);
    auto host_x11 = gen(shape);
    auto host_grad00 = gen(shape);
    auto host_grad01 = gen(shape);
    auto host_grad10 = gen(shape);
    auto host_grad11 = gen(shape);
    auto host_zero_grad = zeros(shape);

    HostTensorND host_y0, host_y1, host_expect_y0, host_expect_y1, host_grad0,
            host_grad1, host_expect_grad0, host_expect_grad1;

    auto client = std::make_shared<test::MockGroupClient>();

    auto run_0 = [&]() {  // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x00 = opr::Host2DeviceCopy::make(*graph0, host_x00, cn0);
        auto x01 = opr::Host2DeviceCopy::make(*graph0, host_x01, cn0);
        auto x10 = opr::Host2DeviceCopy::make(*graph0, host_x10, cn0);
        auto x0 = opr::Concat::make({x00, x01}, 0);
        auto expect_y0 = opr::Concat::make({x00, x10}, 0);
        auto y0 = opr::CollectiveComm::make(
                {x0}, graph0.get(), "alltoall", 2, false, 0, true, client,
                {Mode::ALL_TO_ALL}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad00 = opr::Host2DeviceCopy::make(*graph0, host_grad00, cn0);
        auto grad10 = opr::Host2DeviceCopy::make(*graph0, host_grad10, cn0);
        auto grad_y0 = opr::Concat::make({grad00, grad10}, 0);
        auto loss = opr::Dot::make(y0, grad_y0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
                {make_callback_copy(y0, host_y0),
                 make_callback_copy(g, host_grad0),
                 make_callback_copy(expect_y0, host_expect_y0)});
        func0->execute();
    };

    auto run_1 = [&]() {  // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x10 = opr::Host2DeviceCopy::make(*graph1, host_x10, cn1);
        auto x11 = opr::Host2DeviceCopy::make(*graph1, host_x11, cn1);
        auto x01 = opr::Host2DeviceCopy::make(*graph1, host_x01, cn1);
        auto x1 = opr::Concat::make({x10, x11}, 0);
        auto expect_y1 = opr::Concat::make({x01, x11}, 0);
        auto y1 = opr::CollectiveComm::make(
                {x1}, graph1.get(), "alltoall", 2, false, 1, true, client,
                {Mode::ALL_TO_ALL}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad01 = opr::Host2DeviceCopy::make(*graph1, host_grad01, cn1);
        auto grad11 = opr::Host2DeviceCopy::make(*graph1, host_grad11, cn1);
        auto grad_y1 = opr::Concat::make({grad01, grad11}, 0);
        auto loss = opr::Dot::make(y1, grad_y1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func0 = graph1->compile(
                {make_callback_copy(y1, host_y1),
                 make_callback_copy(g, host_grad1),
                 make_callback_copy(expect_y1, host_expect_y1)});
        func0->execute();
    };

    auto run_2 = [&]() {  // check
        auto graph2 = ComputingGraph::make();
        auto grad00 = opr::Host2DeviceCopy::make(*graph2, host_grad00, cn0);
        auto grad11 = opr::Host2DeviceCopy::make(*graph2, host_grad11, cn0);
        auto zero_grad =
                opr::Host2DeviceCopy::make(*graph2, host_zero_grad, cn0);
        auto out_grad0_expect = opr::Concat::make({grad00, zero_grad}, 0);
        auto out_grad1_expect = opr::Concat::make({zero_grad, grad11}, 0);
        auto func2 = graph2->compile(
                {make_callback_copy(out_grad0_expect, host_expect_grad0),
                 make_callback_copy(out_grad1_expect, host_expect_grad1)});
        func2->execute();
    };

    std::thread t0(run_0);
    std::thread t1(run_1);
    std::thread t2(run_2);

    t0.join();
    t1.join();
    t2.join();

    MGB_ASSERT_TENSOR_EQ(host_expect_y0, host_y0);
    MGB_ASSERT_TENSOR_EQ(host_expect_y1, host_y1);
    MGB_ASSERT_TENSOR_EQ(host_expect_grad0, host_grad0);
    MGB_ASSERT_TENSOR_EQ(host_expect_grad1, host_grad1);
}
