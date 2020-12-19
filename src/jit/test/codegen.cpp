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
#include "megbrain/opr/tensor_manip.h"
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

    auto host_x0 = gen({23, 42}, cn), host_x1 = gen({23, 1}, cn),
         host_x2 = gen({23, 42}, cn);

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

void run_mlir_broadcast(CompNode cn) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Float32> gen;

    auto host_x0 = gen({10, 20, 5, 6}, cn), host_x1 = gen({1, 20, 1, 1}, cn),
         host_x2 = gen({10, 1, 5, 1}, cn), host_x3 = gen({10, 1, 1, 1}, cn);

    auto a = opr::Host2DeviceCopy::make(*graph, host_x0),
         b = opr::Host2DeviceCopy::make(*graph, host_x1),
         c = opr::Host2DeviceCopy::make(*graph, host_x2),
         d = opr::Host2DeviceCopy::make(*graph, host_x3);

    auto y =
            opr::Elemwise::make({a, b, c}, opr::Elemwise::Mode::FUSE_MUL_ADD3) +
            opr::Elemwise::make({d}, opr::Elemwise::Mode::ABS) - 0.3f;

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

void run_mlir_different_shape(CompNode cn) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<dtype::Float32> gen;

    auto run = [&](TensorShape tshp) {
        auto host_x = gen(tshp, cn);
        auto x = opr::Host2DeviceCopy::make(*graph, host_x);
        auto y = x * 2;
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
    };

    run({23, 42});
    run({16, 31});
    run({32, 56});
    run({10});
}

struct MlirTestOpt {
    float low;
    float high;
    float maxerr;
};

struct MlirTestOpt get_mode_opt(opr::Elemwise::Mode mode) {
    struct MlirTestOpt opt = {0, 1, 1e-6};
    if (mode == opr::Elemwise::Mode::ABS) {
        opt.low = -10;
        opt.high = 10;
    } else if (mode == opr::Elemwise::Mode::LOG) {
        opt.low = 0.1;
        opt.high = 4;
    } else if (mode == opr::Elemwise::Mode::ERF or
               mode == opr::Elemwise::Mode::ERFC) {
        opt.low = -5;
        opt.high = 5;
    } else if (mode == opr::Elemwise::Mode::ERFINV) {
        opt.low = -0.999;
        opt.high = 0.999;
        opt.maxerr = 1e-4;
    } else if (mode == opr::Elemwise::Mode::ERFCINV) {
        opt.low = 0.001;
        opt.high = 1.999;
        opt.maxerr = 1e-4;
    }
    return opt;
}

template <typename tag, int arity>
void run_mlir_mode(CompNode cn) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    auto opt = get_mode_opt(tag::mode);
    HostTensorGenerator<dtype::Float32, RandomDistribution::UNIFORM> gen(opt.low,
                                                                         opt.high);

    SmallVector<std::shared_ptr<HostTensorND>> hosts;
    VarNodeArray input_vars;
    for (int i = 0; i < arity; i++) {
        hosts.push_back(gen({2323, 4242}, cn));
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

    MGB_ASSERT_TENSOR_NEAR(host_y, host_y_jit, opt.maxerr);
}
#endif

}  // anonymous namespace

/* ===================== TestJITHalideCodeGenCude ===================== */

#if MGB_JIT_HALIDE
template <typename tag>
class TestJITHalideCodeGenCuda : public ::testing::Test {};
TYPED_TEST_CASE(TestJITHalideCodeGenCuda, test_types);
TYPED_TEST(TestJITHalideCodeGenCuda, run) {
    REQUIRE_GPU(1);
    run<TypeParam>(Backend::HALIDE, CompNode::load("gpu0"));
}
#endif

/* ===================== TestJITNvrtcCodeGen ===================== */

template <typename tag>
class TestJITNvrtcCodeGen : public ::testing::Test {};
TYPED_TEST_CASE(TestJITNvrtcCodeGen, test_types);
TYPED_TEST(TestJITNvrtcCodeGen, run) {
    REQUIRE_GPU(1);
    run<TypeParam>(Backend::NVRTC, CompNode::load("gpu0"));
}

/* ===================== TestJITMlirCodeGen ===================== */

#if MGB_JIT_MLIR
TEST(TestJITMlirCodeGen, Basic) {
    auto cn = CompNode::load("cpu0");
    run_mlir(cn);
    run_mlir_broadcast(cn);
    run_mlir_different_shape(cn);
}

TEST(TestJITMlirCodeGen, BasicGPU) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");
    run_mlir(cn);
    run_mlir_broadcast(cn);
    run_mlir_different_shape(cn);
}

/* ===================== TestJITMlirUnaryElemwise ===================== */

// clang-format off
#define FOREACH_UNARY_MODE(cb) \
    cb(RELU) \
    cb(ABS) \
    cb(NEGATE) \
    cb(ACOS) \
    cb(ASIN) \
    cb(CEIL) \
    cb(EXP) \
    cb(FLOOR) \
    cb(LOG) \
    cb(LOG1P) \
    cb(SIN) \
    cb(COS) \
    cb(TANH) \
    cb(FAST_TANH) \
    cb(H_SWISH) \
    cb(SIGMOID) \
    cb(EXPM1) \
    cb(ROUND) \
    cb(ERF) \
    cb(ERFINV) \
    cb(ERFC) \
    cb(ERFCINV)
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

#define SKIP_MODE(_mode)                                 \
    if (TypeParam::mode == opr::Elemwise::Mode::_mode) { \
        printf("skip\n");                                \
        return;                                          \
    }

TYPED_TEST(TestJITMlirUnaryElemwise, run) {
    auto cn = CompNode::load("cpu0");

    SKIP_MODE(ROUND);

    run_mlir_mode<TypeParam, 1>(cn);
}

TYPED_TEST(TestJITMlirUnaryElemwise, runGpu) {
    REQUIRE_GPU(1);
    auto cn = CompNode::load("gpu0");

    SKIP_MODE(ROUND);

    run_mlir_mode<TypeParam, 1>(cn);
}

/* ===================== TestJITMlirBinaryElemwise ===================== */

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
    cb(POW) \
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
    cb(FUSE_ADD_H_SWISH) \
    cb(ATAN2)
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

    SKIP_MODE(MOD);

    run_mlir_mode<TypeParam, 2>(cn);
}

/* ===================== TestJITMlirTenaryElemwise ===================== */

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


/* ===================== TestJITMlirTypeCvt ===================== */

template <typename itype, typename otype>
void run_typecvt(CompNode cn) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<itype, RandomDistribution::UNIFORM> gen(-10, 10);

    auto host_x = gen({23, 42}, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto y = opr::TypeCvt::make(x, otype());

    auto ig_gen = std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());

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
};

#define add_typecvt_gtest(itype, otype) \
    TEST(TestJITMlirTypeCvt, itype##_to_##otype) { \
        run_typecvt<dtype::itype, dtype::otype>(CompNode::load("cpu0")); \
    } \
    TEST(TestJITMlirTypeCvt, itype##_to_##otype##_GPU) { \
        REQUIRE_GPU(1); \
        run_typecvt<dtype::itype, dtype::otype>(CompNode::load("gpu0")); \
    }

#if !MEGDNN_DISABLE_FLOAT16

// TODO: the support for f16 and bf16 is currently not complete in mlir

// FPExtOp
// add_typecvt_gtest(Float16, Float32);
// add_typecvt_gtest(BFloat16, Float32);
// add_typecvt_gtest(Float16, BFloat16);

// FPTruncOp
// add_typecvt_gtest(Float32, Float16);
// add_typecvt_gtest(Float32, BFloat16);
// add_typecvt_gtest(Float16, BFloat16);

#endif

// FPToSIOp
add_typecvt_gtest(Float32, Int8);
add_typecvt_gtest(Float32, Int16);
add_typecvt_gtest(Float32, Int32);

// FPToUIOp
add_typecvt_gtest(Float32, Uint8);

// SIToFPOp
add_typecvt_gtest(Int8, Float32);
add_typecvt_gtest(Int16, Float32);
add_typecvt_gtest(Int32, Float32);

// UIToFPOp
add_typecvt_gtest(Uint8, Float32);

#undef add_typecvt_gtest

/* ===================== TestJITMlirDimshuffle ===================== */

void run_dimshuffle(CompNode cn, TensorShape ishape,
                    const std::vector<int>& pattern) {
    set_backend(Backend::MLIR);
    auto graph = ComputingGraph::make();
    HostTensorGenerator<> gen;

    auto host_x = gen(ishape, cn);
    auto x = opr::Host2DeviceCopy::make(*graph, host_x);
    auto y = opr::Dimshuffle::make(x, pattern);

    auto ig_gen = std::make_unique<InternalGraphGenerator>(y.node()->owner_opr());

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

void run_dimshuffle_cases(CompNode cn) {
    run_dimshuffle(cn, {3, 4, 5}, {2, 0, 1});
    run_dimshuffle(cn, {3, 4, 5}, {1, -1, 0, 2});
}

TEST(TestJITMlirDimshuffle, Basic) {
    run_dimshuffle_cases(CompNode::load("cpu0"));
}

TEST(TestJITMlirDimshuffle, BasicGPU) {
    REQUIRE_GPU(1);
    run_dimshuffle_cases(CompNode::load("gpu0"));
}

#endif  // MGB_JIT_MLIR

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
