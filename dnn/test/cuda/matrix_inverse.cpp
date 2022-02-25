#include "test/cuda/fixture.h"

#include "megdnn/oprs/linalg.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, MATRIX_INVERSE) {
    InvertibleMatrixRNG rng;
    Checker<MatrixInverse>{handle_cuda()}
            .set_rng(0, &rng)
            .execs({{4, 5, 5}, {}})
            .execs({{100, 3, 3}, {}});
}

// vim: syntax=cpp.doxygen
