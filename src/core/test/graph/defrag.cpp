/**
 * \file src/core/test/graph/defrag.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/utility.h"
#include "megbrain/comp_node_env.h"

#include "megbrain/test/helper.h"

using namespace mgb;

#if MGB_CUDA && MGB_ENABLE_EXCEPTION
// defrag only works when exception is enabled

namespace {
void run_graph(size_t mem_reserved, bool enable_defrag) {
    CompNode::try_coalesce_all_free_memory();
    CompNode::finalize();
    auto cn = CompNode::load("gpux");
    cn.sync(); // wait for async init to finish
    size_t size = mem_reserved / (12.1 * 4);

    HostTensorND host_x{cn, dtype::Int32()};
    auto px = host_x.resize({size}).ptr<dt_int32>();
    RNGxorshf rng{next_rand_seed()};
    dt_int32 expect = 0;
    for (size_t i = 0; i < size; ++ i) {
        expect += (px[i] = rng());
    }
    expect *= 7;

    auto dev_x = std::make_shared<DeviceTensorND>();
    dev_x->copy_from(host_x);

    auto graph = ComputingGraph::make();
    graph->options().enable_var_mem_defragment = enable_defrag;
    graph->options().force_dynamic_alloc = true;
    graph->options().graph_opt_level = 0;
    graph->options().var_sanity_check_first_run = false;

    auto x0 = opr::SharedDeviceTensor::make(*graph, dev_x).rename("x0"),
         // x1 has rdonly fwd chain
         x1 = opr::Concat::make({x0, x0}, 0).add_axis(0).reshape({size*2}).rename("x1"),
         x2 = opr::Concat::make({x1, x0}, 0).rename("x2"),
         x3 = opr::Concat::make({x2, x0}, 0).rename("x3"),
         x4 = opr::Concat::make({x3, x0}, 0).rename("x4"),
         y0 = opr::reduce_sum(x1, x1.make_scalar(1)).rename("y0"),
         y1 = opr::reduce_sum(x4, x4.make_scalar(1)).rename("y1"),
         y = opr::add(y0, y1, {cn});

    set_priority(y0, 100); // y0 executes after defrag

    HostTensorND host_y;
    auto func = graph->compile({make_callback_copy(y, host_y)});
    func->execute();
    ASSERT_EQ(expect, host_y.ptr<dt_int32>()[0]);

#if 0
    auto show = [](SymbolVar var) {
        auto size = var.node()->shape().total_nr_elems() * 4;
        const void* begin = var.node()->prev_dev_ptr(),
              *end = static_cast<const dt_byte*>(begin) + size;
        return ssprintf("[%p,%p]%.2fMiB", begin, end, size / 1024.0 / 1024);
    };
    printf("x0=%s\nx1=%s\nx2=%s\nx3=%s\nx4=%s\n",
            show(x0).c_str(),
            show(x1).c_str(),
            show(x2).c_str(),
            show(x3).c_str(),
            show(x4).c_str()
            );
#endif
}
} // anonymous namespace

TEST(TestGraph, Defragment) {
    REQUIRE_GPU(1);
    CompNode::load("gpux").activate();
    size_t reserve;
    {
        size_t free, tot;
        MGB_CUDA_CHECK(cudaMemGetInfo(&free, &tot));
        reserve = free * 0.92;
    }
    auto reserve_setting = ssprintf("b:%zu", reserve);

    auto do_run = [reserve]() {
        ASSERT_THROW(run_graph(reserve, false), MemAllocError);
        run_graph(reserve, true);
    };

    // reserve memory explicitly to avoid uncontrollable factors
    constexpr const char* KEY = "MGB_CUDA_RESERVE_MEMORY";
    auto old_value = getenv(KEY);
    setenv(KEY, reserve_setting.c_str(), 1);
    MGB_TRY {
        do_run();
    } MGB_FINALLY(
        if (old_value) {
            setenv(KEY, old_value, 1);
        } else {
            unsetenv(KEY);
        }
        CompNode::try_coalesce_all_free_memory();
        CompNode::finalize();
    );
}
#endif // MGB_CUDA && MGB_ENABLE_EXCEPTION

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

