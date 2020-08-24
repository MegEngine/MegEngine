/**
 * \file src/opr/test/muxing.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/muxing.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/blas.h"
#include "megbrain/test/helper.h"
#include "megbrain/test/numerical_diff.h"
#include <random>

using namespace mgb;

namespace {

void run_all_gather(const std::vector<size_t>& axis_size, bool& success,
                    int axis, bool make_sleep = true, bool check_gx = false) {
    success = false;
    size_t SIZE0 = 34, SIZE1 = 47;
    if (check_gx) {
        SIZE0 = 3;
        SIZE1 = 4;
    }
    std::vector<double> sleep_time;
    size_t tot_axis_size = 0;
    for (size_t i = 0; i < axis_size.size(); ++ i) {
        sleep_time.push_back(i * 0.05 + 0.1);
        tot_axis_size += axis_size[i];
    }
#if __cplusplus >= 201703L
    std::default_random_engine rng_engine;
    std::shuffle(sleep_time.begin(), sleep_time.end(), rng_engine);
#else
    std::random_shuffle(sleep_time.begin(), sleep_time.end());
#endif

    auto constexpr DEVICE_TYPE = CompNode::DeviceType::CUDA;
    size_t nr_dev = std::min<size_t>(
            CompNode::get_device_count(DEVICE_TYPE), 4);
    HostTensorGenerator<> gen;
    std::vector<std::shared_ptr<HostTensorND>> host_x, host_lossp;
    for (size_t i = 0; i < axis_size.size(); ++ i) {
        // test both cases of non-overlapping and overlapping comp nodes
        int stream = axis_size.size() % 2 ? i / nr_dev : 0;
        auto cn = CompNode::load({DEVICE_TYPE,
                static_cast<int>(i % nr_dev), stream});
        host_x.push_back(gen({SIZE0, axis_size[i], SIZE1}, cn));
        host_lossp.push_back(gen({SIZE0, tot_axis_size, SIZE1}, cn));
    }

    auto graph = ComputingGraph::make();
    SymbolVarArray dev_x, dev_x_delay, dev_lossp;
    for (size_t i = 0; i < axis_size.size(); ++ i) {
        dev_x.push_back(opr::Host2DeviceCopy::make(*graph, host_x[i]));
        dev_lossp.push_back(opr::Host2DeviceCopy::make(*graph, host_lossp[i]));
        auto delay = dev_x.back();
        if (make_sleep)
            delay = opr::Sleep::make(delay, sleep_time[i]);
        dev_x_delay.push_back(delay);
    }

    auto dev_y = opr::AllGather::make(dev_x_delay, axis);

    SymbolVarArray dev_gx;

    SymbolVar loss;
    if (check_gx) {
        ASSERT_EQ(axis_size.size(), dev_y.size());
        TensorShape shp = {SIZE0 * tot_axis_size * SIZE1};
        auto cn = CompNode::load("gpu0");

        for (size_t i = 0; i < axis_size.size(); ++ i) {
            auto cur_loss = opr::Dot::make(
                    dev_y[i].reshape(shp), dev_lossp[i].reshape(shp)).rename(
                    ssprintf("loss%zd", i));
            if (cn != cur_loss.node()->comp_node()) {
                cur_loss = opr::Copy::make(cur_loss, cn);
            }
            if (loss.node())
                loss = loss + cur_loss;
            else
                loss = cur_loss;
        }

        for (auto &&i: dev_x)
            dev_gx.push_back(cg::grad(loss, i));
    }

    ComputingGraph::OutputSpec outspec;
    std::vector<HostTensorND> host_y(dev_y.size()), host_gx(host_x.size());
    for (size_t i = 0; i < axis_size.size(); ++ i) {
        outspec.push_back(make_callback_copy(dev_y[i], host_y[i]));
        if (check_gx)
            outspec.push_back(make_callback_copy(dev_gx[i], host_gx[i]));
    }

    auto func = graph->compile(outspec);
    func->execute();
    mgb_log("exec_time=%.3fms; axis_size=%zd",
            func->wait().get_prev_exec_time() * 1e3, axis_size.size());

    {
        // check y
        HostTensorND expected{CompNode::load("gpu0"), dtype::Float32()};
        {
            expected.resize({SIZE0, tot_axis_size, SIZE1});
            size_t start = 0;
            for (auto &&i: host_x) {
                auto end = start + i->shape().shape[1];
                for (size_t slice = 0; slice < SIZE0; ++ slice) {
                    memcpy(expected.ptr<float>({slice, start, 0}),
                            i->ptr<float>({slice}),
                            (end - start) * SIZE1 * sizeof(float));
                }
                start = end;
            }
        }

        for (auto &&i: host_y)
            MGB_ASSERT_TENSOR_EQ(expected, i);

    }

    if (check_gx) {
        std::vector<HostTensorND*> inp;
        for (auto &&i: host_x)
            inp.push_back(i.get());

        HostTensorND host_loss;
        auto func = graph->compile({make_callback_copy(loss, host_loss)});

        auto cost = [&]() {
            func->execute();
            return host_loss.ptr<float>()[0];
        };
        auto diff = numerical_diff_pt2(inp, cost,
                std::vector<Maybe<float>>(inp.size(), 1.f));

        for (size_t i = 0; i < axis_size.size(); ++ i)
            MGB_ASSERT_TENSOR_NEAR(diff.at(i), host_gx.at(i), 1e-4);
    }

    success = true;
}

} // anonymous namespace

TEST(TestMuxing, AllGather) {
    REQUIRE_GPU(4);
    bool success;
    run_all_gather({2}, success, 1, false, true);
    ASSERT_TRUE(success) << "failed grad 1";
    run_all_gather({2, 3, 4, 5}, success, 1, false, true);
    ASSERT_TRUE(success) << "failed grad 4";

    std::mt19937 rng;
    std::vector<size_t> sizes;
    for (size_t i = 1; i <= 8; i ++) {
        sizes.push_back(10 + rng() % 10);
        run_all_gather(sizes, success, 1);
        ASSERT_TRUE(success) << ssprintf("failed at axis_size %zd", i);
    }
    run_all_gather(sizes, success, 1, false);
};

TEST(TestMuxing, AllGatherWithNegativeAxis) {
    REQUIRE_GPU(4);
    bool success;
    run_all_gather({2}, success, -2, false, true);
    ASSERT_TRUE(success) << "failed grad 1";
    run_all_gather({2, 3, 4, 5}, success, -2, false, true);
    ASSERT_TRUE(success) << "failed grad 4";

    std::mt19937 rng;
    std::vector<size_t> sizes;
    for (size_t i = 1; i <= 8; i ++) {
        sizes.push_back(10 + rng() % 10);
        run_all_gather(sizes, success, -2);
        ASSERT_TRUE(success) << ssprintf("failed at axis_size %zd", i);
    }
    run_all_gather(sizes, success, -2, false);
};

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

