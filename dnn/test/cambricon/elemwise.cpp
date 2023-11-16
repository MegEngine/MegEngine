#include "test/common/elemwise.h"
#include "megdnn/oprs.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"

using namespace megdnn;
using namespace test;

namespace {

class BoolRNG final : public RNG {
    std::mt19937_64 m_rng;

public:
    BoolRNG(size_t seed) : m_rng(seed) {}

    void gen(const TensorND& tensor) override {
        std::uniform_int_distribution<int> dist(0, 1);
        auto ptr = tensor.ptr<bool>() + tensor.layout.span().low_elem;
        for (size_t i = 0; i < tensor.layout.span().dist_elem(); ++i)
            ptr[i] = (dist(m_rng) == 1);
    }
};
}  // namespace

TEST_F(CAMBRICON, ELEMWISE_UNARY) {
    Checker<ElemwiseForward> checker(handle_cambricon());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_activate = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                            DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        checker.execs({{N, C, H, W}, {}});
    };
    auto run_unary = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                         DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        checker.execs({{N, C, H, W}, {}});
    };

#define RUN_ACTIVATE(_dt)                            \
    run_activate(4, 32, 10, 10, Mode::RELU, _dt);    \
    run_activate(4, 32, 10, 10, Mode::SIGMOID, _dt); \
    run_activate(4, 32, 10, 10, Mode::LOGSIGMOID, _dt);
    RUN_ACTIVATE(dtype::Float32());
    RUN_ACTIVATE(dtype::Float16());
#undef RUN_ACTIVATE

#define RUN_UNARY(_dt)                           \
    run_unary(4, 32, 10, 10, Mode::NEGATE, _dt); \
    run_unary(4, 32, 10, 10, Mode::EXP, _dt);

    RUN_UNARY(dtype::Float32());
    checker.set_epsilon(1e-2);
    RUN_UNARY(dtype::Float16());
#undef RUN_UNARY

#define RUN_UNARY_POS(_dt)                    \
    run_unary(4, 32, 10, 10, Mode::LOG, _dt); \
    run_unary(4, 32, 10, 10, Mode::SQRT, _dt);

    UniformFloatRNG pos_rng_f32{.1f, 1000.f};
    checker.set_rng(0, &pos_rng_f32);
    RUN_UNARY_POS(dtype::Float32());
    checker.set_epsilon(1e-2);
    RUN_UNARY_POS(dtype::Float16());
#undef RUN_UNARY_POS

    BoolRNG bool_rng(0);
    checker.set_rng(0, &bool_rng);
    run_unary(4, 32, 10, 10, Mode::NOT, dtype::Bool());
}

TEST_F(CAMBRICON, ELEMWISE_BINARY) {
    Checker<ElemwiseForward> checker(handle_cambricon());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_binary = [&](Mode mode, DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(
                2, dtype);
        checker.execs({{3, 7, 4}, {3, 7, 4}, {}});
        checker.execs({{3, 4, 7}, {1, 4, 1}, {}});
        checker.execs({{1, 4, 1}, {3, 4, 7}, {}});
        checker.execs({{1, 2, 2}, {1, 1, 1}, {}});
        checker.execs({{1, 1, 1}, {1, 2, 2}, {}});
        checker.execl({{{100, 1}, {4, 4}, dtype}, {{1}, {1}, dtype}, {}});
    };
    checker.set_epsilon(1e-3);
#define RUN_BINARY(_dt)                   \
    run_binary(Mode::ADD, _dt);           \
    run_binary(Mode::SUB, _dt);           \
    run_binary(Mode::MUL, _dt);           \
    run_binary(Mode::TRUE_DIV, _dt);      \
    run_binary(Mode::SWITCH_GT0, _dt);    \
    run_binary(Mode::MIN, _dt);           \
    run_binary(Mode::MAX, _dt);           \
    run_binary(Mode::SIGMOID_GRAD, _dt);  \
    run_binary(Mode::SOFTPLUS_GRAD, _dt); \
    run_binary(Mode::FLOOR_DIV, _dt);     \
    run_binary(Mode::MOD, _dt);

    RUN_BINARY(dtype::Float32());
    checker.set_epsilon(1e-2);
    RUN_BINARY(dtype::Float16());
#undef RUN_BINARY

    UniformFloatRNG rng_fp32(1e-5, 7e1);
    UniformFloatRNG rng_fp16(1e-3, 3e1);
    checker.set_epsilon(1e-3);
    checker.set_rng(0, &rng_fp32);
    run_binary(Mode::POW, dtype::Float32());
    checker.set_rng(0, &rng_fp16);
    run_binary(Mode::POW, dtype::Float16());

    BoolRNG bool_rng(0);
    checker.set_rng(0, &bool_rng);
    checker.set_rng(1, &bool_rng);
    run_binary(Mode::AND, dtype::Bool());
}

TEST_F(CAMBRICON, ELEMWISE_MOD) {
    Checker<ElemwiseForward> checker(handle_cambricon());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_binary = [&](Mode mode, DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(
                2, dtype);
        checker.execs({{3, 4, 7}, {3, 4, 7}, {}});
        checker.execs({{3, 4, 5, 7}, {1, 4, 1, 1}, {}});
        checker.execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {}});
        checker.execs({{3, 4, 7}, {1, 4, 1}, {}});
        checker.execs({{1, 4, 1}, {3, 4, 7}, {}});
        checker.execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}});
        checker.execs({{1, 1, 1, 1}, {3, 4, 5, 7}, {}});
        checker.execs({{1, 7}, {1, 7}, {}});
        checker.execs({{1, 2, 2}, {1, 2, 1}, {}});
        checker.execs({{1, 2, 1}, {1, 2, 2}, {}});
        checker.execs({{1, 2, 2}, {1, 1, 1}, {}});
        checker.execs({{1, 1, 1}, {1, 2, 2}, {}});
        checker.execs({{3, 4, 1}, {3, 4, 1}, {}});
        checker.execl({{{100, 1}, {4, 4}, dtype}, {{1}, {1}, dtype}, {}});
    };
    UniformFloatNonZeroRNG nonzero_rng{1, 100};
    checker.set_rng(1, &nonzero_rng);
    run_binary(Mode::MOD, dtype::Int32());

    run_binary(Mode::MOD, dtype::Float32());
}

TEST_F(CAMBRICON, ELEMWISE_TERNARY) {
    Checker<ElemwiseForward> checker(handle_cambricon());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_ternary = [&](Mode mode, DType dtype) {
        checker.set_param(mode)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype);
        checker.execs({{2, 3, 4}, {2, 3, 4}, {2, 3, 4}, {2, 3, 4}});
        checker.execs({{1, 3, 4}, {2, 1, 4}, {2, 3, 1}, {2, 3, 4}});
        checker.execs({{2, 1, 1, 5}, {4, 5}, {3, 1, 1}, {2, 3, 4, 5}});
        checker.execs({{3, 1, 1}, {5}, {4, 1}, {3, 4, 5}});
        checker.execl(
                {{{100, 1}, {4, 4}, dtype}, {{1}, {1}, dtype}, {{1}, {1}, dtype}, {}});
    };
    checker.set_epsilon(1e-3);
#define RUN_TERNARY(_dt)                  \
    run_ternary(Mode::COND_LEQ_MOV, _dt); \
    run_ternary(Mode::COND_LT_MOV, _dt);

    RUN_TERNARY(dtype::Float32());
    checker.set_epsilon(1e-2);
    RUN_TERNARY(dtype::Float16());

#undef RUN_TERNARY
}

TEST_F(CAMBRICON, ELEMWISE_CLIP) {
    Checker<ElemwiseForward> checker(handle_cambricon());
    using Mode = ElemwiseForward::Param::Mode;
    auto run = [&](DType dtype) {
        checker.set_param(Mode::CLIP)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype);
        checker.execs({{4, 7, 9}, {1}, {1}, {4, 7, 9}});
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

TEST_F(CAMBRICON, ELEMWISE_OPTENSOR) {
    Checker<ElemwiseForward> checker(handle_cambricon());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_binary = [&](Mode mode, DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype).set_dtype(
                2, dtype);
        checker.execs({{3, 7, 4}, {3, 7, 4}, {}});
        checker.execs({{3, 4, 7}, {1, 4, 1}, {}});
        checker.execs({{1, 4, 1}, {3, 4, 7}, {}});
        checker.execs({{1, 2, 2}, {1, 1, 1}, {}});
        checker.execs({{1, 1, 1}, {1, 2, 2}, {}});
        checker.execl({{{100, 1}, {4, 4}, dtype}, {{1}, {1}, dtype}, {}});
    };
    checker.set_epsilon(1e-3);
#define RUN_BINARY(_dt)         \
    run_binary(Mode::ADD, _dt); \
    run_binary(Mode::SUB, _dt); \
    run_binary(Mode::MUL, _dt);

    RUN_BINARY(dtype::Int32());
    RUN_BINARY(dtype::Float32());
    checker.set_epsilon(1e-2);
    RUN_BINARY(dtype::Float16());
#undef RUN_BINARY
}

TEST_F(CAMBRICON, ELEMWISE_FMADD3) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle_cambricon());
    checker.set_param(Mode::FUSE_MUL_ADD3);
    auto make_shape = [](const TensorShape& s0, const TensorShape& s1,
                         const TensorShape& s2) {
        TensorShape dest;
        dest.ndim = s0.ndim;
        for (size_t i = 0; i < dest.ndim; ++i)
            dest[i] = std::max(s0[i], s1[i]);
        return TensorShapeArray{s0, s1, s2, dest};
    };
    checker.exec(make_shape({2, 1}, {2, 2}, {2, 2}));
    checker.exec(make_shape({2, 2}, {2, 1}, {2, 2}));
    checker.exec(make_shape({2, 2}, {2, 2}, {1}));
    checker.exec(make_shape({3, 1}, {1, 3}, {3, 1}));
    checker.exec(make_shape({2, 1, 2, 1, 2, 1}, {1, 2, 1, 2, 1, 2}, {1}));
    checker.exec(make_shape({1, 1, 3}, {5, 8, 1}, {5, 8, 1}));

    //! check VEC_SCALA_SCALA
    checker.execs({{1, 8, 8, 8}, {1}, {1}, {1, 8, 8, 8}});

    //! check VEC_SCALA_VEC
    checker.execs({{1, 8, 8, 8}, {1}, {1, 8, 8, 8}, {1, 8, 8, 8}});

    // not contig
    TensorLayout ly{{2, 3}, dtype::Float32()};
    ly.stride[0] = 4;
    checker.execl({ly, ly, ly, {{2, 3}, dtype::Float32()}});
}