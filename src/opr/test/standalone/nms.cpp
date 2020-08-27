/**
 * \file src/opr/test/standalone/nms.cpp
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 */

#include "megbrain/opr/standalone/nms_opr.h"
#include "megbrain/test/helper.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/tensor_gen.h"
#include <random>

using namespace mgb;

namespace {

void run_on_comp_node(const char* cn_name) {
    auto cn = CompNode::load(cn_name);
    auto graph = ComputingGraph::make();
    auto host_x = std::make_shared<HostTensorND>(cn, TensorShape{1, 2, 4},
                                                 dtype::Float32{});
    auto ptr = host_x->ptr<float>();
    ptr[0] = 0.; ptr[1] = 0.;
    ptr[2] = 2.; ptr[3] = 2.;
    ptr[4] = 0.5; ptr[5] = 0.5;
    ptr[6] = 1.5; ptr[7] = 1.5;

    auto x = opr::Host2DeviceCopy::make(*graph, host_x);

    {
        auto idx = opr::standalone::NMSKeep::make(x, {0.2, 16});
        auto size = idx.node()->owner_opr()->output(1);
        HostTensorND host_idx, host_size;
        auto func = graph->compile({make_callback_copy(idx, host_idx),
                                    make_callback_copy(size, host_size)});
        func->execute().wait();
        auto idx_ptr = host_idx.ptr<int32_t>();
        auto size_ptr = host_size.ptr<int32_t>();
        ASSERT_EQ(size_ptr[0], 1);
        ASSERT_EQ(idx_ptr[0], 0);
    }
    {
        auto idx = opr::standalone::NMSKeep::make(x, {0.5, 16});
        auto size = idx.node()->owner_opr()->output(1);
        HostTensorND host_idx, host_size;
        auto func = graph->compile({make_callback_copy(idx, host_idx),
                                    make_callback_copy(size, host_size)});
        func->execute().wait();
        auto idx_ptr = host_idx.ptr<int32_t>();
        auto size_ptr = host_size.ptr<int32_t>();
        ASSERT_EQ(size_ptr[0], 2);
        ASSERT_EQ(idx_ptr[0], 0);
        ASSERT_EQ(idx_ptr[1], 1);
    }
}

}

TEST(TestOprNMS, CPU) {
    run_on_comp_node("cpu0");
}

TEST(TestOprNMS, GPU) {
    REQUIRE_GPU(1);
    run_on_comp_node("gpu0");
}

#if MGB_ENABLE_EXCEPTION
TEST(TestOprNMS, InvalidInput) {
    HostTensorGenerator<> gen;
    auto graph = ComputingGraph::make();
    auto host_x = gen({1, 9, 5});
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    ASSERT_ANY_THROW(opr::standalone::NMSKeep::make(x, {1., 1}));
}
#endif  // MGB_ENABLE_EXCEPTION
