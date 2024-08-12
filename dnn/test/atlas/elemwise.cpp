#include "test/common/elemwise.h"
#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, ELEMWISE_UNARY) {
    Checker<ElemwiseForward> checker(handle_atlas());
    using Mode = ElemwiseForward::Param::Mode;
    UniformFloatRNG ui_rng{.1f, 1000.f};
    UniformFloatRNG ui_rng2{.1f, 10.f};
    auto run_unary = [&](size_t N, Mode mode, DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype);
        checker.execs({{N}, {}});
    };

#define RUN_UNARY_UINT16(_dt)              \
    run_unary(100, Mode::RELU, _dt);       \
    run_unary(100, Mode::NEGATE, _dt);     \
    run_unary(100, Mode::LOG, _dt);        \
    run_unary(100, Mode::LOGSIGMOID, _dt); \
    run_unary(100, Mode::SIGMOID, _dt);    \
    run_unary(100, Mode::SQRT, _dt);       \
    run_unary(100, Mode::ABS, _dt);        \
    run_unary(100, Mode::SIGN, _dt);       \
    run_unary(100, Mode::SIGN, _dt);       \
    run_unary(100, Mode::TANH, _dt);

    checker.set_rng(0, &ui_rng);
    RUN_UNARY_UINT16(dtype::Float32());

    checker.set_rng(0, &ui_rng2);
    run_unary(10, Mode::EXP, dtype::Float32());

    BoolRNG bool_rng(0);
    checker.set_rng(0, &bool_rng);
    run_unary(100, Mode::NOT, dtype::Bool());
}

TEST_F(ATLAS, ELEMWISE) {
    Checker<ElemwiseForward> checker(handle_atlas());
    using Mode = ElemwiseForward::Param::Mode;
    UniformFloatRNG ui_rng{.1f, 1000.f};

    checker.set_rng(0, &ui_rng).set_rng(1, &ui_rng);
    auto run_binary = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                          DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);

        checker.execs({{5}, {5}, {}});
        checker.execs({{4}, {4}, {}});
        checker.execs({{4}, {1}, {}});
        checker.execs({{N, C / 4, H, W, 4}, {N, C / 4, H, W, 4}, {}});
        checker.execs({{N, C / 4, H, W, 4}, {1, C / 4, 1, 1, 4}, {}});
        checker.execs({{N, C / 32, H, W, 32}, {N, C / 32, H, W, 32}, {}});
        checker.execs({{N, C / 32, H, W, 32}, {1, C / 32, 1, 1, 32}, {}});
        checker.execs({{3, 5, 7}, {3, 5, 7}, {}});
    };

#define RUN_BINARY_UINT16(_dt)                          \
    run_binary(4, 32, 10, 10, Mode::ADD, _dt);          \
    run_binary(4, 32, 10, 10, Mode::MUL, _dt);          \
    run_binary(4, 32, 10, 10, Mode::SUB, _dt);          \
    run_binary(4, 32, 10, 10, Mode::TRUE_DIV, _dt);     \
    run_binary(4, 32, 10, 10, Mode::MAX, _dt);          \
    run_binary(4, 32, 10, 10, Mode::MIN, _dt);          \
    run_binary(4, 32, 10, 10, Mode::SWITCH_GT0, _dt);   \
    run_binary(4, 32, 10, 10, Mode::SIGMOID_GRAD, _dt); \
    run_binary(4, 32, 10, 10, Mode::FLOOR_DIV, _dt);    \
    run_binary(4, 32, 10, 10, Mode::TANH_GRAD, _dt);    \
    run_binary(4, 32, 10, 10, Mode::MOD, _dt);

    RUN_BINARY_UINT16(dtype::Float32());

    UniformFloatRNG ui_rng2{.1f, 10.f};
    checker.set_rng(0, &ui_rng2).set_rng(1, &ui_rng2);
    run_binary(4, 32, 10, 10, Mode::POW, dtype::Float32());

    UniformFloatRNG ui_rng3{-100.f, 100.f};
    checker.set_rng(0, &ui_rng3).set_rng(1, &ui_rng3);
    run_binary(4, 32, 10, 10, Mode::ABS_GRAD, dtype::Float32());

    checker.set_rng(0, &ui_rng2).set_rng(1, &ui_rng2);
    run_binary(4, 32, 10, 10, Mode::SOFTPLUS_GRAD, dtype::Float32());

    BoolRNG bool_rng(0);
    checker.set_rng(0, &bool_rng).set_rng(1, &bool_rng);
    run_binary(4, 32, 10, 10, Mode::AND, dtype::Bool());

#undef RUN_BINARY_UINT16
}

TEST_F(ATLAS, ELEMWISE_TERNARY) {
    Checker<ElemwiseForward> checker(handle_atlas());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_ternary = [&](Mode mode, DType dtype) {
        checker.set_param(mode)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype);
        checker.execs({{3, 4, 5, 6}, {3, 4, 5, 6}, {3, 4, 5, 6}, {}});
        checker.execs({{10, 4, 5, 6}, {10, 4, 5, 6}, {10, 4, 5, 6}, {}});
    };
#define RUN_TERNARY(_dt)                 \
    run_ternary(Mode::COND_LT_MOV, _dt); \
    run_ternary(Mode::COND_LEQ_MOV, _dt);

    checker.set_epsilon(1e-3);
    RUN_TERNARY(dtype::Float32());

    checker.set_epsilon(1e-2);
    RUN_TERNARY(dtype::Float16());

#undef RUN_TERNARY
}

TEST_F(ATLAS, ELEMWISE_CLIP) {
    Checker<ElemwiseForward> checker(handle_atlas());
    using Mode = ElemwiseForward::Param::Mode;
    auto run = [&](DType dtype) {
        checker.set_param(Mode::CLIP)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype);
        checker.execs({{1, 1, 9}, {1}, {1}, {1, 1, 9}});
    };

    UniformFloatRNG rng_min(1e-3, 10e1);
    UniformFloatRNG rng_max(10e1, 20e1);
    UniformFloatRNG rng_mid(1e-3, 20e1);
    checker.set_rng(0, &rng_mid);
    checker.set_rng(1, &rng_min);
    checker.set_rng(2, &rng_max);
    run(dtype::Float32());
    run(dtype::Float16());
}
