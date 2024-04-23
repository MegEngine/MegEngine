#include "test/common/elemwise.h"
#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"

using namespace megdnn;
using namespace test;

TEST_F(ATLAS, ELEMWISE) {
    Checker<ElemwiseForward> checker(handle_atlas());
    using Mode = ElemwiseForward::Param::Mode;
    UniformFloatRNG ui_rng{.1f, 1000.f};
    UniformFloatRNG ui_rng2{.1f, 10.f};
    auto run_unary = [&](size_t N, Mode mode, DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype);
        checker.execs({{N}, {}});
    };

#define RUN_UNARY_UINT16(_dt)          \
    run_unary(100, Mode::RELU, _dt);   \
    run_unary(100, Mode::NEGATE, _dt); \
    run_unary(100, Mode::LOG, _dt);

    checker.set_rng(0, &ui_rng);
    RUN_UNARY_UINT16(dtype::Float32());

    checker.set_rng(0, &ui_rng2);
    run_unary(10, Mode::EXP, dtype::Float32());
    checker.set_rng(0, &ui_rng);
#undef RUN_UNARY_UINT16
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
#define RUN_BINARY_UINT16(_dt)                                     \
    run_binary(4, 32, 10, 10, Mode::ADD, _dt);                     \
    run_binary(4, 32, 10, 10, Mode::MUL, _dt);                     \
    run_binary(4, 32, 10, 10, Mode::SUB, _dt);                     \
    run_binary(4, 32, 10, 10, Mode::SWITCH_GT0, dtype::Float32()); \
    run_binary(4, 32, 10, 10, Mode::TRUE_DIV, _dt);                \
    checker.set_rng(0, &ui_rng).set_rng(1, &ui_rng);

    RUN_BINARY_UINT16(dtype::Float32());
#undef RUN_BINARY_UINT16
}