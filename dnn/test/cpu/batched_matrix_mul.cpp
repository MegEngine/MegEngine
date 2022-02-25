#include "test/cpu/fixture.h"

#include <chrono>
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"

using namespace megdnn;
using namespace test;

//! check batch=1 and batch_stride is arbitrarily
TEST_F(CPU, BATCHED_MATRIX_MUL_BATCH_1) {
    matrix_mul::check_batched_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(), "", 1e-3,
            std::vector<matrix_mul::TestArg>{{5, 5, 5, 0, 5, 5, 5, 1, 5, 5, 5}});
}

// vim: syntax=cpp.doxygen
