/**
 * \file src/core/test/graph/multi_thread.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/convolution.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
#include "megbrain/system.h"

#include "megbrain/test/helper.h"

#include <atomic>
#include <thread>

using namespace mgb;

namespace {
template <typename Opr>
HostTensorND eval_conv(
        const std::shared_ptr<HostTensorND>& src,
        const std::shared_ptr<HostTensorND>& filter,
        const typename Opr::Param& param = {}) {
    auto graph = ComputingGraph::make();
    graph->options().log_level = 0;
    SymbolVar x = opr::Host2DeviceCopy::make(*graph, src);
    SymbolVar y = opr::Host2DeviceCopy::make(*graph, filter);
    SymbolVar z = Opr::make(x, y, param);
    HostTensorND host_z;
    auto func = graph->compile({make_callback_copy(z, host_z)});
    func->execute();

    host_z.sync();
    return host_z;
}

template <typename Opr>
HostTensorND eval_conv_cpu(
        const HostTensorND& xv, const HostTensorND& fv,
        const typename Opr::Param& param = {}) {
    auto cn = CompNode::load("cpux");
    auto src = std::make_shared<HostTensorND>(cn, xv.layout()),
         filter = std::make_shared<HostTensorND>(cn, fv.layout());
    memcpy(src->raw_ptr(), xv.raw_ptr(), xv.layout().span().dist_byte());
    memcpy(filter->raw_ptr(), fv.raw_ptr(), fv.layout().span().dist_byte());
    return eval_conv<Opr>(src, filter, param);
}
}  // namespace

TEST(TestGraph, AsyncExecLevel) {
    REQUIRE_GPU(1);

    std::thread::id th_null, th_gpu0, th_gpu1, th_cpu0, th_cpu1;
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;

    auto make_marker = [&](std::thread::id& dest, CompNode cn) {
        auto cb = [&dest, cn](DeviceTensorND& dv) {
            dest = std::this_thread::get_id();
            mgb_assert(dv.comp_node() == cn);
        };
        auto x = opr::Host2DeviceCopy::make(*graph, gen({1}, cn));
        return opr::CallbackInjector::make(x, cb);
    };

    int casenum = -1;
    auto check = [&](int level, std::initializer_list<SymbolVar> ys,
                     std::initializer_list<int> eid_list) {
        ++casenum;
        ComputingGraph::OutputSpec spec;
        for (auto i : ys) {
            spec.push_back({i, {}});
        }
        graph->options().async_exec_level = level;
        auto func = graph->compile(spec);
        th_gpu0 = th_gpu1 = th_cpu0 = th_cpu1 = th_null;
        func->execute();
        std::thread::id get_thid[4] = {th_gpu0, th_gpu1, th_cpu0, th_cpu1},
                        eid2thid[4] = {th_null, th_null, th_null, th_null};
        int cur_get_thid = 0;
        for (auto eid : eid_list) {
            while (cur_get_thid < 4 && get_thid[cur_get_thid] == th_null) {
                ++cur_get_thid;
            }
            ASSERT_LT(cur_get_thid, 4);
            std::thread::id expect;
            if (eid == 0) {
                expect = std::this_thread::get_id();
            } else {
                ASSERT_GE(eid, 1);
                ASSERT_LT(eid, 5);
                auto&& thid = eid2thid[eid - 1];
                if (thid == th_null) {
                    thid = get_thid[cur_get_thid];
                }
                expect = thid;
                ASSERT_NE(expect, std::this_thread::get_id());
            }
            ASSERT_EQ(expect, get_thid[cur_get_thid]) << ssprintf(
                    "failed on case #%d with cur_get_thid=%d eid=%d", casenum,
                    cur_get_thid, eid);
            ++cur_get_thid;
        }
        while (cur_get_thid < 4 && get_thid[cur_get_thid] == th_null) {
            ++cur_get_thid;
        }
        ASSERT_EQ(4, cur_get_thid);
    };

    auto yg0 = make_marker(th_gpu0, CompNode::load("gpu0:0")),
         yg1 = make_marker(th_gpu1, CompNode::load("gpu0:1")),
         yc0 = make_marker(th_cpu0, CompNode::load("cpu0:0")),
         yc1 = make_marker(th_cpu1, CompNode::load("cpu0:1"));
    check(0, {yg0, yg1, yc0, yc1}, {0, 0, 0, 0});
    check(1, {yg0, yg1, yc0, yc1}, {1, 2, 3, 3});
    check(1, {yc0, yc1}, {0, 0});
    check(1, {yg0}, {0});
    check(1, {yg0, yg1}, {1, 2});
    check(0b10, {yc0, yc1}, {1, 2});
    check(0b10, {yg0, yg1}, {1, 2});
    check(0b10, {yg0, yg1, yc0, yc1}, {1, 2, 3, 4});
    check(0b100, {yg0}, {1});
    check(0b100, {yc0}, {1});
    check(0b100, {yc0, yc1}, {1, 1});
    check(0b110, {yc0, yc1}, {1, 2});
    check(0b110, {yg0, yg1, yc0, yc1}, {1, 2, 3, 4});
}

TEST(TestGraph, ParallelRun) {
    // check race conditions when graphs are executed on multple threads
    std::atomic_size_t sync_counter{0};
    constexpr size_t NR_RUN = 50;
    size_t nr_worker = std::max(4, sys::get_cpu_count() / 4);
    if (auto setting = MGB_GETENV("TestGraphParallelRun_nr_worker")) {
        nr_worker = std::stoul(setting);
    }
    mgb_log("use %zu workers", nr_worker);

    auto sync_barrier = [&sync_counter, nr_worker](size_t& cnt) {
        ++sync_counter;
        ++cnt;
        while (sync_counter < cnt * nr_worker)
            ;
    };

    auto do_worker = [&sync_barrier](size_t sync_cnt) {
        HostTensorGenerator<> gen;
        auto host_x = gen({23});
        HostTensorND host_y, y_expect;
        y_expect.copy_from(*host_x);
        {
            auto py = y_expect.ptr<float>();
            for (int i = 0; i < 23; ++i) {
                for (int j = 0; j < 5; ++j) {
                    py[i] = py[i] * 2 + 3;
                }
            }
        }

        sync_barrier(sync_cnt);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = x;
        for (int i = 0; i < 5; ++i) {
            y = y * 2 + 3;
        }

        sync_barrier(sync_cnt);
        auto func = graph->compile({make_callback_copy(y, host_y)});

        sync_barrier(sync_cnt);
        func->execute();
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
        memset(host_y.raw_ptr(), -1, 23 * sizeof(float));

        sync_barrier(sync_cnt);
        func->execute();
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
        func->wait();
    };
    auto worker = [&]() {
        size_t scnt = 0;
        for (size_t run_id = 0; run_id < NR_RUN; ++run_id) {
            do_worker(scnt);
        }
    };

    std::vector<std::thread> workers;
    for (size_t i = 0; i < nr_worker; ++i)
        workers.emplace_back(worker);

    for (auto&& i : workers)
        i.join();
}
#ifndef IOS
TEST(TestGraph, MultiThreadRecorder) {
    using ConvParam = opr::Convolution::Param;
    ConvParam param;
    param.sparse = ConvParam::Sparse::GROUP;
    HostTensorGenerator<> gen;
    auto cn = CompNode::load("cpux");
    auto host_x = gen({3, 4, 10, 8}, cn), host_y = gen({2, 3, 2, 3, 3}, cn);
    auto worker = [&](int record_level) {
        HostTensorND host_z;
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x),
             y = opr::Host2DeviceCopy::make(*graph, host_y),
             z = opr::Convolution::make(x, y, param);
        graph->options().comp_node_seq_record_level = record_level;
        graph->options().var_sanity_check_first_run = false;
        auto sync = (record_level != 1);
        auto func = graph->compile({make_callback_copy(z, host_z, sync)});
        for (int i = 0; i < 5; i++) {
            func->execute();
            if (!sync) {
                func->wait();
            }
        }
        auto expect = eval_conv_cpu<opr::Convolution>(*host_x, *host_y, param);
        MGB_ASSERT_TENSOR_NEAR(expect, host_z, 1e-3);
    };

    std::vector<std::thread> workers;
    for (size_t i = 0; i < 4; ++i)
        workers.emplace_back(worker, i % 2);

    for (auto&& i : workers)
        i.join();
}
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
