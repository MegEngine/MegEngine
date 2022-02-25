#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/sliding_window_transpose.h"
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, SLIDINGWINDOWTRANSPOSE_FORWARD) {
    UniformFloatRNG rng(0, 1);
    auto args = sliding_window_transpose::get_args();
    for (auto&& arg : args) {
        Checker<SlidingWindowTransposeForward> checker(handle_cuda());
        checker.set_rng(0, &rng);
        checker.set_epsilon(1e-2);
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;
        {
            auto opr = handle_cuda()->create_operator<SlidingWindowTransposeForward>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype) {
            checker.set_dtype(0, dtype).set_dtype(1, dtype);
        };
        set_dtype(dtype::Float32());
        checker.set_param(arg.param).exec(TensorShapeArray{ilayout, olayout});
        set_dtype(dtype::Float16());
        checker.set_param(arg.param).exec(TensorShapeArray{ilayout, olayout});
    }
}

TEST_F(CUDA, SLIDINGWINDOWTRANSPOSE_BACKWARD) {
    UniformFloatRNG rng(0, 1);
    auto args = sliding_window_transpose::get_args();
    for (auto&& arg : args) {
        Checker<SlidingWindowTransposeBackward> checker(handle_cuda());
        // checker.set_epsilon(1e-2);
        checker.set_rng(0, &rng);
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;
        {
            auto opr = handle_cuda()->create_operator<SlidingWindowTranspose>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype) {
            checker.set_dtype(0, dtype).set_dtype(1, dtype);
        };
        set_dtype(dtype::Float32());
        checker.set_param(arg.param).exec(TensorShapeArray{olayout, ilayout});
        set_dtype(dtype::Float16());
        checker.set_param(arg.param).exec(TensorShapeArray{olayout, ilayout});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_SLIDINGWINDOWTRANSPOSE_FORWARD) {
    auto args = sliding_window_transpose::get_benchmark_args();
    for (auto&& arg : args) {
        CUBenchmarker<SlidingWindowTransposeForward> bencher(handle_cuda());
        bencher.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .exec(TensorShapeArray{arg.ishape, {}});
    }
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
