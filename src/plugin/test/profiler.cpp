/**
 * \file src/plugin/test/profiler.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/plugin/profiler.h"
#include "megbrain/test/helper.h"
#include <sstream>

using namespace mgb;

namespace {
    void run_test(CompNode cn, const char *fpath) {
        HostTensorGenerator<> gen;
        auto host_x = gen({1}), host_y = gen({1});
        auto graph = ComputingGraph::make();
        SymbolVar
            x = opr::Host2DeviceCopy::make(*graph, host_x, cn).rename("x"),
            y = opr::Host2DeviceCopy::make(*graph, host_y, cn).rename("y"),
            z = x + y;

        HostTensorND host_z;
        auto func = graph->compile({make_callback_copy(z, host_z)});
        auto profiler = std::make_shared<GraphProfiler>(graph.get());
        func->execute();
        float vx = host_x->ptr<float>()[0], vy = host_y->ptr<float>()[0],
        vz = host_z.sync().ptr<float>()[0];
        ASSERT_FLOAT_EQ(vx + vy, vz);

        profiler->to_json()->writeto_fpath(output_file(fpath));
    }
}

TEST(TestGraphProfiler, APlusBGPU) {
    REQUIRE_GPU(1);
    run_test(CompNode::load("gpu0"), "test_profiler_gpu.json");
}

TEST(TestGraphProfiler, APlusBCPU) {
    run_test(CompNode::load("cpu0"), "test_profiler_cpu.json");
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

