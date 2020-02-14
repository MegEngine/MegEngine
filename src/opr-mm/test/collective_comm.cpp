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
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/test/helper.h"
#include "megbrain/graph.h"

using namespace mgb;

namespace {

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

class MockGroupClient final : public opr::GroupClient {
    public:
        ~MockGroupClient() override = default;

        uint64_t opr_register(const std::string& key, size_t nr_devices, uint32_t rank,
                uintptr_t stream) {
            return m_mgr.opr_register(key, nr_devices, rank, stream);
        }

        std::vector<std::string> gather_uid(const std::string& uid,
                const std::string& key, uint32_t size, uint32_t rank) {
            return m_mgr.gather_uid(uid, key, size, rank);
        }

        void set_output_shape(const std::string& key,
                              const TensorShape& shape) override {
            m_mgr.set_output_shape(key, shape);
        }
    
        TensorShape get_output_shape(const std::string& key) override {
            return m_mgr.get_output_shape(key);
        }

        uint32_t group_barrier(uint32_t size, uint32_t rank) override {
            return m_mgr.group_barrier(size, rank);
        }
    
    private:
        opr::GroupManager m_mgr;
};

}  // namespace

TEST(TestOprCollectiveComm, AllReduce) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    auto run_mode = [&](const Mode mode) {
        HostTensorGenerator<> gen;
        auto host_x0 = gen({28, 28});
        auto host_x1 = gen({28, 28});
        HostTensorND host_y0, host_y1, host_y_expect;

        auto client = std::make_shared<MockGroupClient>();

        auto run_0 = [&]() {
            auto graph0 = ComputingGraph::make();
            auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0);
            auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "all_reduce",
                    2, 0, 0, client, {mode}, dtype::Float32(), "nccl")[0];
            auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
            func0->execute();
        };

        auto run_1 = [&]() {
            auto graph1 = ComputingGraph::make();
            auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
            auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "all_reduce",
                    2, 1, 0, client, {mode}, dtype::Float32(), "nccl")[0];
            auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
            func1->execute();
        };

        auto run_2 = [&]() {
            auto graph2 = ComputingGraph::make();
            auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
            auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
            auto y_expect = make_all_reduce_output(mode, {x0, x1});
            auto func2 = graph2->compile({make_callback_copy(y_expect, host_y_expect)});
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

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "all_reduce", 2, 0, 0, client,
                {Mode::ALL_REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
            {make_callback_copy(y0, host_y0),
             make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "all_reduce", 2, 1, 0, client,
                {Mode::ALL_REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile(
            {make_callback_copy(y1, host_y1),
             make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() { // check
        auto graph2 = ComputingGraph::make();

        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = make_all_reduce_output(Mode::ALL_REDUCE_SUM, {x0, x1});

        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect = make_all_reduce_output(Mode::ALL_REDUCE_SUM, {grad0, grad1});

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

TEST(TestOprCollectiveComm, AllGather) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y1, host_y_expect;

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "all_gather", 2, 0, 0, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "all_gather", 2, 1, 0, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
        func1->execute();
    };

    auto run_2 = [&]() { // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = opr::Concat::make({x0, x1}, 0);
        auto func2 = graph2->compile({make_callback_copy(y_expect, host_y_expect)});
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

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "all_gather", 2, 0, 0, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
            {make_callback_copy(y0, host_y0),
             make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "all_gather", 2, 1, 0, client,
                {Mode::ALL_GATHER}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile(
            {make_callback_copy(y1, host_y1),
             make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() { // check
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
             make_callback_copy(out_grad_expect[1], host_out_grad1_expect)});
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
    auto host_x0 = gen({8});
    auto host_x1 = gen({8});
    HostTensorND host_y0, host_y1, host_y0_expect, host_y1_expect;

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "reduce_scatter_sum",
                       2, 0, 0, client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "reduce_scatter_sum",
                       2, 1, 0, client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile({make_callback_copy(y1, host_y1)});
        func1->execute();
    };

    auto run_2 = [&]() { // check
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

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "reduce_scatter_sum",
                2, 0, 0, client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
            {make_callback_copy(y0, host_y0),
             make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "reduce_scatter_sum",
                2, 1, 0, client, {Mode::REDUCE_SCATTER_SUM}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto loss = opr::Dot::make(y1, grad1);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile(
            {make_callback_copy(y1, host_y1),
             make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() { // check
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

TEST(TestOprCollectiveComm, ReduceSum) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    auto host_x1 = gen({28, 28});
    HostTensorND host_y0, host_y_expect;

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "reduce", 2, 0, 0, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "reduce", 2, 1, 0, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        auto func1 = graph1->compile({{y1, nullptr}});
        func1->execute();
    };

    auto run_2 = [&]() { // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y_expect = x0 + x1;
        auto func2 = graph2->compile({make_callback_copy(y_expect, host_y_expect)});
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

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "reduce", 2, 0, 0, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad = opr::Host2DeviceCopy::make(*graph0, host_grad, cn0);
        auto loss = opr::Dot::make(y0, grad);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
            {make_callback_copy(y0, host_y0),
             make_callback_copy(g, host_out_grad0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto x1 = opr::Host2DeviceCopy::make(*graph1, host_x1, cn1);
        auto y1 = opr::CollectiveComm::make({x1}, graph1.get(), "reduce", 2, 1, 0, client,
                {Mode::REDUCE_SUM}, dtype::Float32(), "nccl")[0];
        y1.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad = opr::Host2DeviceCopy::make(*graph1, gen({1}), cn1);
        auto loss = opr::Dot::make(y1, grad);
        auto g = opr::VirtualGrad::make(loss, x1);

        auto func1 = graph1->compile({{y1, nullptr}, make_callback_copy(g, host_out_grad1)});
        func1->execute();
    };

    auto run_2 = [&]() { // check
        auto graph2 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph2, host_x0, cn0);
        auto x1 = opr::Host2DeviceCopy::make(*graph2, host_x1, cn0);
        auto y0_expect = x0 + x1;
        auto func2 = graph2->compile({
            make_callback_copy(y0_expect, host_y0_expect)});
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

TEST(TestOprCollectiveComm, Broadcast) {
    REQUIRE_GPU(2);
    auto cn0 = CompNode::load("gpu0");
    auto cn1 = CompNode::load("gpu1");

    HostTensorGenerator<> gen;
    auto host_x0 = gen({28, 28});
    HostTensorND host_y0, host_y1;

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "broadcast", 2, 0, 0, client,
                {Mode::BROADCAST}, dtype::Float32(), "nccl")[0];
        auto func0 = graph0->compile({make_callback_copy(y0, host_y0)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        auto y_dev = std::make_shared<DeviceTensorND>(DeviceTensorND()
                                                      .comp_node(cn1)
                                                      .dtype(dtype::Float32())
                                                      .resize(host_x0->shape()));
        auto y1 = opr::CollectiveComm::make({}, graph1.get(), "broadcast", 2, 1, 0, client,
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

    auto client = std::make_shared<MockGroupClient>();

    auto run_0 = [&]() { // rank 0
        auto graph0 = ComputingGraph::make();
        graph0->options().graph_opt_level = 0;

        auto x0 = opr::Host2DeviceCopy::make(*graph0, host_x0, cn0);
        auto y0 = opr::CollectiveComm::make({x0}, graph0.get(), "broadcast", 2, 0, 0, client,
                {Mode::BROADCAST}, dtype::Float32(), "nccl")[0];
        y0.node()->owner_opr()->node_prop().attribute().priority = -1;

        auto grad0 = opr::Host2DeviceCopy::make(*graph0, host_grad0, cn0);
        auto loss = opr::Dot::make(y0, grad0);
        auto g = opr::VirtualGrad::make(loss, x0);

        auto func0 = graph0->compile(
            {make_callback_copy(y0, host_y0),
             make_callback_copy(g, host_out_grad)});
        func0->execute();
    };

    auto run_1 = [&]() { // rank 1
        auto graph1 = ComputingGraph::make();
        graph1->options().graph_opt_level = 0;

        auto y1 = opr::CollectiveComm::make({}, graph1.get(), "broadcast", 2, 1, 0, client,
                {Mode::BROADCAST}, dtype::Float32(), "nccl", {cn1})[0];

        auto grad1 = opr::Host2DeviceCopy::make(*graph1, host_grad1, cn1);
        auto g = opr::CollectiveComm::make({grad1}, graph1.get(), "broadcast:grad", 2, 1, 0, client,
                Mode::REDUCE_SUM, dtype::Float32(), "nccl")[0];
        g.node()->owner_opr()->node_prop().attribute().priority = 1;

        auto func1 = graph1->compile({make_callback_copy(y1, host_y1), {g, nullptr}});
        func1->execute();
    };

    auto run_2 = [&]() { // check
        auto graph2 = ComputingGraph::make();
        auto grad0 = opr::Host2DeviceCopy::make(*graph2, host_grad0, cn0);
        auto grad1 = opr::Host2DeviceCopy::make(*graph2, host_grad1, cn0);
        auto out_grad_expect = grad0 + grad1;
        auto func2 = graph2->compile({
            make_callback_copy(out_grad_expect, host_out_grad_expect)});
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
