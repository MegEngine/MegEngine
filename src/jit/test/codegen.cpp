/**
 * \file src/jit/test/codegen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./helper.h"

#include "megbrain/jit/executor_opr.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/test/helper.h"

#if MGB_JIT
using namespace mgb;
using namespace jit;

#define FOREACH_CASE(cb) cb(simple) cb(grad)

namespace {
#define def_tag(x) \
    struct x {};
FOREACH_CASE(def_tag)
#undef def_tag

#define t(n) n,
using test_types = ::testing::Types<FOREACH_CASE(t) void>;
#undef t

template <typename tag>
void run(Backend backend, CompNode cn);

template <>
void run<simple>(Backend backend, CompNode cn) {
    set_backend(backend);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x0 = gen({23, 42}, cn), host_x1 = gen({23, 1}, cn),
         host_x2 = gen({1, 42}, cn);

    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         c = opr::Host2DeviceCopy::make(*graph, host_x2);

    a = opr::TypeCvt::make(a, dtype::Float16{});

    auto y = a + b * c;
    y = opr::TypeCvt::make(y, dtype::Float16{});
    y = opr::TypeCvt::make((y + y.make_scalar_dt(1.f)), dtype::Float32{});

    VarNodeArray inputs{a.node(), b.node(), c.node()}, outputs{y.node()};
    auto ig_gen =
            std::make_unique<InternalGraphGenrator>(y.node()->owner_opr());

    for (auto i : get_rev_topo_order(y)) {
        if (!i->same_type<opr::Host2DeviceCopy>()) {
            ig_gen->add_opr(i);
        }
    }

    auto igraph = ig_gen->generate();
    auto y_jit = JITExecutor::make(igraph, ig_gen->orig_inps());

    HostTensorND host_y, host_y_jit;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_jit, host_y_jit)});
    func->execute();

    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_jit, 5e-3);
};

template <>
void run<grad>(Backend backend, CompNode cn) {
    set_backend(backend);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;
    auto host_x0 = gen({23, 42}, cn), host_x1 = gen({23, 1}, cn),
         host_x2 = gen({1, 42}, cn);

    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         c = opr::Host2DeviceCopy::make(*graph, host_x2);

    a = opr::TypeCvt::make(a, dtype::Float16{});

    auto y = opr::floor_div(a, opr::abs(b) + 0.1f) * opr::sin(c);

    VarNodeArray inputs{a.node(), b.node(), c.node()}, outputs{y.node()};
    auto ig_gen =
            std::make_unique<InternalGraphGenrator>(y.node()->owner_opr());

    for (auto i : get_rev_topo_order(y)) {
        if (!i->same_type<opr::Host2DeviceCopy>()) {
            ig_gen->add_opr(i);
        }
    }

    auto igraph = ig_gen->generate();
    auto y_jit = JITExecutor::make(igraph, ig_gen->orig_inps());

    HostTensorND host_y, host_y_jit;
    auto func = graph->compile({make_callback_copy(y, host_y),
                                make_callback_copy(y_jit, host_y_jit)});
    func->execute();

    MGB_ASSERT_TENSOR_EQ(host_y, host_y_jit);

    auto grad = [loss = opr::reduce_sum(y_jit, y_jit.make_scalar(1))](
            SymbolVar x) {
        return cg::grad(loss, x, false, false).node();
    };
    ASSERT_EQ(nullptr, grad(a));
    ASSERT_EQ(nullptr, grad(b));
    ASSERT_NE(nullptr, grad(c));
};

template <>
void run<void>(Backend, CompNode) {}
}  // anonymous namespace

#if MGB_JIT_HALIDE
template <typename tag>
class TestJITHalideCodeGenCuda : public ::testing::Test {};
TYPED_TEST_CASE(TestJITHalideCodeGenCuda, test_types);
TYPED_TEST(TestJITHalideCodeGenCuda, run) {
    REQUIRE_GPU(1);
    run<TypeParam>(Backend::HALIDE, CompNode::load("gpu0"));
}
#endif

template <typename tag>
class TestJITNvrtcCodeGen : public ::testing::Test {};
TYPED_TEST_CASE(TestJITNvrtcCodeGen, test_types);
TYPED_TEST(TestJITNvrtcCodeGen, run) {
    REQUIRE_GPU(1);
    run<TypeParam>(Backend::NVRTC, CompNode::load("gpu0"));
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
