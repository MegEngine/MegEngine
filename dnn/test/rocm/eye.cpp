#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

#include "test/rocm/benchmarker.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, EYE) {
    Checker<Eye> checker(handle_rocm());
    for (DType dtype : std::vector<DType>{
                 DNN_INC_FLOAT16(dtype::Float16() MEGDNN_COMMA) dtype::Int32(),
                 dtype::Float32()})
        for (int k = -20; k < 20; ++k) {
            checker.set_param({k, dtype.enumv()});
            checker.set_dtype(0, dtype);
            checker.exec(TensorShapeArray{{3, 4}});
            checker.exec(TensorShapeArray{{4, 3}});
        }
}

TEST_F(ROCM, EYE_BENCHMARK) {
    auto benchmarker = ROCMBenchmarker<Eye>(handle_rocm(), handle_naive(false));
    benchmarker.set_display(true);
    benchmarker.set_param({10240, dtype::Float32().enumv()});
    benchmarker.set_dtype(0, dtype::Float32());
    auto time_ms = benchmarker.execs({{10000, 10000}});
    float io = 10000 * 10000 * dtype::Float32().size();
    printf("io = %.3f GB, bandwidth = %.3f GB/s\n", io / 1e9, io / (1e6 * time_ms));
}

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
