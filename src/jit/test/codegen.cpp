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

#include <memory>
#include "./helper.h"

#include "megbrain/jit/executor_opr.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/test/helper.h"
#include "megdnn/dtype.h"

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
            std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());

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
            std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());

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

#if MGB_JIT_MLIR
void run_mlir(CompNode cn) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Float32> gen;

    auto host_x0 = gen({23, 42}, cn), host_x1 = gen({23, 42}, cn),
         host_x2 = gen({23, 42}, cn), host_x3 = gen({23, 42}, cn);

    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         c = opr::Host2DeviceCopy::make(*graph, host_x2);

    auto y = a + b * c + 0.3f;

    auto ig_gen =
            std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());

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
}

template <typename tag, int arity>
void run_mlir_mode(CompNode cn) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    float low = 0.f, high = 1.f;
    if (tag::mode == opr::Elemwise::Mode::LOG) {
        low = 0.1;
        high = 4;
    }
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen(low,
                                                                         high);

    SmallVector<std::shared_ptr<HostTensorND>> hosts;
    VarNodeArray input_vars;
    for (int i = 0; i < arity; i++) {
        hosts.push_back(gen({23, 42}, cn));
        input_vars.push_back(
                opr::Host2DeviceCopy::make(*graph, hosts[i]).node());
    }

    auto y = opr::Elemwise::make(input_vars, tag::mode);

    auto ig_gen =
            std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());

    for (auto i : get_rev_topo_order(y)) {
        if (!i->template same_type<opr::Host2DeviceCopy>()) {
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
}
#endif

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

#if MGB_JIT_MLIR
TEST(TestJITMlirCodeGen, Basic) {
    auto cn = CompNode::load("cpu0");
    run_mlir(cn);
}

TEST(TestJITMlirCodeGen, BasicGPU) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    run_mlir(cn);
}

///////////////////////// unary ///////////////////////////////
// clang-format off
#define FOREACH_UNARY_MODE(cb) \
    cb(RELU) \
    cb(ABS) \
    cb(NEGATE) \
    cb(CEIL) \
    cb(EXP) \
    cb(FLOOR) \
    cb(LOG) \
    cb(LOG1P) \
    cb(SIN) \
    cb(TANH) \
    cb(FAST_TANH) \
    cb(H_SWISH) \
    cb(SIGMOID) \
    cb(EXPM1) \
    cb(ROUND)
// clang-format on
template <typename tag>
class TestJITMlirUnaryElemwise : public ::testing::Test {};

#define def_tag(x)                                                          \
    struct x {                                                              \
        static constexpr opr::Elemwise::Mode mode = opr::Elemwise::Mode::x; \
    };
FOREACH_UNARY_MODE(def_tag)
#undef def_tag

#define t(n) n,
        using mlir_elemwise_unary_types =
                ::testing::Types<FOREACH_UNARY_MODE(t) ABS>;
#undef t
TYPED_TEST_CASE(TestJITMlirUnaryElemwise, mlir_elemwise_unary_types);
TYPED_TEST(TestJITMlirUnaryElemwise, run) {
    auto cn = CompNode::load("cpu0");
    run_mlir_mode<TypeParam, 1>(cn);
}

#define SKIP_MODE(_mode)                                 \
    if (TypeParam::mode == opr::Elemwise::Mode::_mode) { \
        printf("skip\n");                                \
        return;                                          \
    }
TYPED_TEST(TestJITMlirUnaryElemwise, runGpu) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");

    SKIP_MODE(SIN);

    run_mlir_mode<TypeParam, 1>(cn);
}

///////////////////////// binary ///////////////////////////////
// clang-format off
#define FOREACH_BINARY_MODE(cb) \
    cb(ADD) \
    cb(FLOOR_DIV) \
    cb(MUL) \
    cb(MAX) \
    cb(MIN) \
    cb(MOD) \
    cb(SUB) \
    cb(TRUE_DIV) \
    cb(ABS_GRAD) \
    cb(SIGMOID_GRAD) \
    cb(SWITCH_GT0) \
    cb(TANH_GRAD) \
    cb(LT) \
    cb(LEQ) \
    cb(EQ) \
    cb(FUSE_ADD_RELU) \
    cb(LOG_SUM_EXP) \
    cb(FUSE_ADD_TANH) \
    cb(FAST_TANH_GRAD) \
    cb(FUSE_ADD_SIGMOID) \
    cb(H_SWISH_GRAD) \
    cb(FUSE_ADD_H_SWISH)
// clang-format on
template <typename tag>
class TestJITMlirBinaryElemwise : public ::testing::Test {};

#define def_tag(x)                                                          \
    struct x {                                                              \
        static constexpr opr::Elemwise::Mode mode = opr::Elemwise::Mode::x; \
    };
FOREACH_BINARY_MODE(def_tag)
#undef def_tag

#define t(n) n,
        using mlir_elemwise_binary_types =
                ::testing::Types<FOREACH_BINARY_MODE(t) ADD>;
#undef t
TYPED_TEST_CASE(TestJITMlirBinaryElemwise, mlir_elemwise_binary_types);
TYPED_TEST(TestJITMlirBinaryElemwise, run) {
    auto cn = CompNode::load("cpu0");
    run_mlir_mode<TypeParam, 2>(cn);
}

TYPED_TEST(TestJITMlirBinaryElemwise, runGpu) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    run_mlir_mode<TypeParam, 2>(cn);
}

///////////////////////// ternary ///////////////////////////////
// clang-format off
#define FOREACH_TERNARY_MODE(cb) \
    cb(COND_LEQ_MOV) \
    cb(FUSE_MUL_ADD3) \
// clang-format on
template <typename tag>
class TestJITMlirTernaryElemwise : public ::testing::Test {};

#define def_tag(x)                                                          \
    struct x {                                                              \
        static constexpr opr::Elemwise::Mode mode = opr::Elemwise::Mode::x; \
    };
FOREACH_TERNARY_MODE(def_tag)
#undef def_tag

#define t(n) n,
        using mlir_elemwise_ternary_types =
                ::testing::Types<FOREACH_TERNARY_MODE(t) COND_LEQ_MOV>;
#undef t
TYPED_TEST_CASE(TestJITMlirTernaryElemwise, mlir_elemwise_ternary_types);
TYPED_TEST(TestJITMlirTernaryElemwise, run) {
    auto cn = CompNode::load("cpu0");
    run_mlir_mode<TypeParam, 3>(cn);
}

TYPED_TEST(TestJITMlirTernaryElemwise, runGpu) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    run_mlir_mode<TypeParam, 3>(cn);
}

#undef SKIP_MODE

#endif

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
