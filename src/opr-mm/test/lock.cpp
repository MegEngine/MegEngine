/**
 * \file src/opr-mm/test/lock.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/test/helper.h"
#include "megbrain/opr/lock.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"

#include <thread>
#include <atomic>

using namespace mgb;

namespace {
    using locker_t = thin_function<SymbolVar(SymbolVar var)>;
    constexpr int NR_RUN = 100, NR_WORKER = 4,
              EXPECTED_SUM = (1 + NR_RUN * NR_WORKER) * NR_RUN * NR_WORKER / 2;

    int run(locker_t lock, locker_t unlock) {
        HostTensorND host_sum{CompNode::load("xpu0"), dtype::Float32()},
                     host_adder;
        host_sum.resize({1}).ptr<float>()[0] = 0;
        host_adder.copy_from(host_sum);

        auto host_one = std::make_shared<HostTensorND>(
                host_sum.comp_node(), host_sum.dtype());
        host_one->resize({1}).ptr<float>()[0] = 1;

        auto dev_sum = std::make_shared<DeviceTensorND>(),
             dev_adder = std::make_shared<DeviceTensorND>();
        dev_sum->copy_from(host_sum);
        dev_adder->copy_from(host_adder);

        std::atomic_int nr_ready{0};

        auto worker = [&]() {
            auto graph = ComputingGraph::make();
            auto sum = opr::SharedDeviceTensor::make(*graph, dev_sum),
                 adder = opr::SharedDeviceTensor::make(*graph, dev_adder),
                 one = lock(opr::Host2DeviceCopy::make(*graph, host_one)),
                 adder_u = unlock(opr::AddUpdate::make(adder, one)),
                 sum_u = unlock(opr::AddUpdate::make(sum, adder_u));

            graph->options().var_sanity_check_first_run = false;
            auto func = graph->compile({{sum_u, {}}});
            func->execute();

            ++ nr_ready;
            while (nr_ready.load() != NR_WORKER);
            for (int i = 1; i < NR_RUN; ++ i)
                func->execute();
        };

        std::vector<std::thread> worker_th;
        for (int i = 0; i < NR_WORKER; ++ i)
            worker_th.emplace_back(worker);
        for (auto &&i: worker_th)
            i.join();

        return host_sum.copy_from(*dev_sum).sync().ptr<float>()[0];
    }
}

TEST(TestOprLock, FailWithoutLock) {
    auto empty = [](SymbolVar v) {
        return v;
    };
    ASSERT_NE(EXPECTED_SUM, run(empty, empty));
}

TEST(TestOprLock, SuccWithLock) {
    auto lock = [](SymbolVar var) {
        return opr::LockAcquire::make(var, {0, 0});
    };

    auto unlock = [](SymbolVar var) {
        return opr::LockRelease::make(var, {0, 0});
    };

    ASSERT_EQ(EXPECTED_SUM, run(lock, unlock));
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

