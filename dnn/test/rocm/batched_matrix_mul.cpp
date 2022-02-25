#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, BATCHED_MATRIX_MUL) {
    Checker<BatchedMatrixMul> checker(handle_rocm());
    checker.set_epsilon(1e-2);
    using Param = MatrixMul::Param;
    size_t b = 9, m = 10, n = 11, k = 12;
    std::vector<DType> dtypes{DNN_INC_FLOAT16(dtype::Float16() MEGDNN_COMMA)
                                      dtype::Float32()};
    for (auto dtype : dtypes)
        for (unsigned mask = 0; mask < 4; ++mask) {
            Param param;
            param.transposeA = mask & 1;
            param.transposeB = mask & 2;
            TensorShape A, B;
            if (param.transposeA)
                A = TensorShape{b, k, m};
            else
                A = TensorShape{b, m, k};
            if (param.transposeB)
                B = TensorShape{b, n, k};
            else
                B = TensorShape{b, k, n};
            checker.set_param(param)
                    .set_dtype(0, dtype)
                    .set_dtype(1, dtype)
                    .set_dtype(2, dtype)
                    .execs({A, B, {}});
        }
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
