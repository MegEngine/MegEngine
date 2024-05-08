#include "megdnn/oprs.h"
#include "test/atlas/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ATLAS, MATRIX_MUL_FLOAT) {
    Checker<MatrixMul> checker(handle_atlas());
    checker.set_before_exec_callback(
            AlgoChecker<MatrixMulForward>("AclMatrixMulForward"));
    using Param = MatrixMul::Param;
    size_t m = 12, n = 16, k = 20;

    std::vector<DType> dtype_array;
    dtype_array.push_back(dtype::Float32());
    dtype_array.push_back(dtype::Float16());

    for (DType dtype : dtype_array) {
        unsigned mask = 0;
        // for (unsigned mask = 0; mask < 4; ++mask) {
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        DType stype = dtype;
        TensorShape A, B;
        if (param.transposeA)
            A = TensorShape{k, m};
        else
            A = TensorShape{m, k};
        if (param.transposeB)
            B = TensorShape{n, k};
        else
            B = TensorShape{k, n};
        if (dtype == dtype::Float16()) {
            param.compute_mode = param::MatrixMul::ComputeMode::FLOAT32;
        }
        checker.set_param(param)
                .set_dtype(0, stype)
                .set_dtype(1, stype)
                .set_dtype(2, dtype)
                .set_epsilon(dtype == dtype::Float16() ? 5e-2 : 5e-3)
                .execs({A, B, {}});
        // }
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
