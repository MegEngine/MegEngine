#include "test/common/matrix_mul.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/task_record_check.h"
#include "test/fallback/fixture.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, MATRIX_MUL) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_matmul_args();
    for (auto arg : args) {
        auto m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;

        if (param.transposeA)
            AS = TensorShape{k, m};
        else
            AS = TensorShape{m, k};
        if (param.transposeB)
            BS = TensorShape{n, k};
        else
            BS = TensorShape{k, n};
        CS = TensorShape{m, n};
        TensorLayout AL, BL, CL;
        AL = TensorLayout(AS, dtype::Float32());
        BL = TensorLayout(BS, dtype::Float32());
        CL = TensorLayout(CS, dtype::Float32());
        checker.set_param(param);
        checker.execl({AL, BL, CL});
    }
}

TEST_F(FALLBACK, MATRIX_MUL_MK4_GI) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "FB_GI_F32_MK4_4x8", param::MatrixMul::Format::MK4, 1);
}

TEST_F(FALLBACK, MATRIX_MULF_GI_F32_4x12) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "FB_GI_F32_4x12");
}

TEST_F(FALLBACK, MATRIX_MUL_RECORD) {
    TaskRecordChecker<MatrixMul> checker(1);
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_matmul_args();
    for (auto arg : args) {
        auto m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;

        if (param.transposeA)
            AS = TensorShape{k, m};
        else
            AS = TensorShape{m, k};
        if (param.transposeB)
            BS = TensorShape{n, k};
        else
            BS = TensorShape{k, n};
        CS = TensorShape{m, n};
        TensorLayout AL, BL, CL;
        AL = TensorLayout(AS, dtype::Float32());
        BL = TensorLayout(BS, dtype::Float32());
        CL = TensorLayout(CS, dtype::Float32());
        checker.set_param(param);
        checker.execl({AL, BL, CL});
    }
}

TEST_F(FALLBACK, MATRIX_MUL_NAIVE_MK4) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(), "FB_NAIVE",
            param::MatrixMul::Format::MK4, 1);
}

TEST_F(FALLBACK, MATRIX_MUL_NAIVE_MK8) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(), "FB_NAIVE",
            param::MatrixMul::Format::MK8, 1);
}

TEST_F(FALLBACK, MATRIX_MUL_NAIVE_MK4_DOT) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(), "FB_NAIVE",
            param::MatrixMul::Format::MK4_DOT, 1);
}

TEST_F(FALLBACK, MATRIX_MUL_NAIVE) {
    Checker<MatrixMul> checker(handle());
    checker.set_before_exec_callback(AlgoChecker<MatrixMul>("FB_NAIVE"));
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_matmul_args();
    for (auto arg : args) {
        auto m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;

        if (param.transposeA)
            AS = TensorShape{k, m};
        else
            AS = TensorShape{m, k};
        if (param.transposeB)
            BS = TensorShape{n, k};
        else
            BS = TensorShape{k, n};
        CS = TensorShape{m, n};
        TensorLayout AL, BL, CL;
        AL = TensorLayout(AS, dtype::Float32());
        BL = TensorLayout(BS, dtype::Float32());
        CL = TensorLayout(CS, dtype::Float32());
        checker.set_param(param);
        checker.execl({AL, BL, CL});
    }
}

TEST_F(FALLBACK, BATCHED_MATRIX_MUL) {
    Checker<BatchedMatrixMul> checker(handle());
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_batched_matmul_args();
    for (auto arg : args) {
        auto b = arg.b, m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;

        if (param.transposeA)
            AS = TensorShape{b, k, m};
        else
            AS = TensorShape{b, m, k};
        if (param.transposeB)
            BS = TensorShape{b, n, k};
        else
            BS = TensorShape{b, k, n};
        TensorLayout AL, BL;
        AL = TensorLayout(AS, dtype::Float32());
        BL = TensorLayout(BS, dtype::Float32());
        checker.set_param(param);
        checker.execs({AL, BL, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_MATRIX_MUL_FB_GI_F32_4x12) {
    auto args = matrix_mul::get_benchmark_matmul_args();
    matrix_mul::benchmark_single_algo(
            handle(), args, dtype::Float32{}, dtype::Float32{}, dtype::Float32{},
            "FB_GI_F32_4x12", param::MatrixMul::Format::DEFAULT);
}

#endif
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
