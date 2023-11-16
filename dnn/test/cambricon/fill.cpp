#include "megdnn/oprs.h"
#include "test/cambricon/fixture.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

inline void run_fill_test(Handle* handle, DType dtype) {
    Checker<Fill> checker(handle);
    for (float value : {-1.23, 0.0, 0.001, 234.0, 2021.072}) {
        checker.set_param({value});
        checker.set_dtype(0, dtype);
        checker.exec(TensorShapeArray{{1, 1}});
        checker.exec(TensorShapeArray{{2, 3, 4}});
    }
}

TEST_F(CAMBRICON, FILL_F32) {
    run_fill_test(handle_cambricon(), dtype::Float32{});
}

TEST_F(CAMBRICON, FILL_I32) {
    run_fill_test(handle_cambricon(), dtype::Int32{});
}

#if !MEGDNN_DISABLE_FLOAT16
TEST_F(CAMBRICON, FILL_F16) {
    run_fill_test(handle_cambricon(), dtype::Float16{});
}
#endif

// vim: syntax=cpp.doxygen
