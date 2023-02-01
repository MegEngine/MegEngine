#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, GENERALNORM_FORWARD) {
    using Param = GeneralNormForward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-6;
    Checker<GeneralNormForward> checker(handle_cuda());
    checker.set_epsilon(1e-2);

    auto run = [&](DType d) {
        for (size_t n_slices : {10, 30})
            for (size_t slice_len : {10, 30}) {
                param.normalized_axis = 0;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{n_slices, slice_len},
                                {n_slices},
                                {n_slices},
                                {n_slices, slice_len},
                                {slice_len},
                                {slice_len}});
                param.normalized_axis = 1;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, d)
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, dtype::Float32())
                        .execs({{n_slices, slice_len},
                                {slice_len},
                                {slice_len},
                                {n_slices, slice_len},
                                {n_slices},
                                {n_slices}});
            }
    };

    run(dtype::Float32());
    run(dtype::Float16());
    run(dtype::BFloat16());
}

TEST_F(CUDA, GENERALNORM_SPEED_FP32) {
    using Param = GeneralNormForward::Param;
    auto benchmarker = Benchmarker<GeneralNormForward>(handle_cuda());
    benchmarker.set_dtype(0, dtype::Float32());
    benchmarker.set_dtype(1, dtype::Float32());
    Param param;
    param.affine = true;
    float eachTime;
    float totalTime = 0.f;

#define ITER 10
    param.normalized_axis = 0;
    for (auto i = 0; i < ITER; i++) {
        eachTime = benchmarker.set_param(param).exec({{100, 2000},
                                {100},
                                {100},
                                {},
                                {},
                                {}});
        totalTime += eachTime;
    }
    totalTime /= ITER;
    printf("PGENERALNORM_SPEED_FP32 AVG TIME: %.6fms\n", totalTime);

    totalTime = 0.f;
    param.normalized_axis = 1;
    for (auto i = 0; i < ITER; i++) {
        eachTime = benchmarker.set_param(param).exec({{2000, 100},
                                {100},
                                {100},
                                {},
                                {},
                                {}});
        totalTime += eachTime;
    }
    totalTime /= ITER;
    printf("PGENERALNORM_SPEED_FP32 AVG TIME: %.6fms\n", totalTime);
#undef ITER
}

TEST_F(CUDA, GENERALNORM_BACKWARD) {
    using Param = GeneralNormBackward::Param;
    Param param;
    param.affine = true;
    param.eps = 1e-6;
    Checker<GeneralNormBackward> checker(handle_cuda());
    checker.set_epsilon(1e-1);

    auto run = [&](DType d) {
        for (size_t n_slices : {10, 30})
            for (size_t slice_len : {10, 30}) {
                param.normalized_axis = 0;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, dtype::Float32())
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, d)
                        .set_dtype(6, d)
                        .set_dtype(7, d)
                        .execs({{n_slices, slice_len},
                                {n_slices, slice_len},
                                {n_slices},
                                {slice_len},
                                {slice_len},
                                {n_slices, slice_len},
                                {n_slices},
                                {n_slices}});
                param.normalized_axis = 1;
                checker.set_param(param)
                        .set_dtype(0, d)
                        .set_dtype(1, d)
                        .set_dtype(2, d)
                        .set_dtype(3, dtype::Float32())
                        .set_dtype(4, dtype::Float32())
                        .set_dtype(5, d)
                        .set_dtype(6, d)
                        .set_dtype(7, d)
                        .execs({{n_slices, slice_len},
                                {n_slices, slice_len},
                                {slice_len},
                                {n_slices},
                                {n_slices},
                                {n_slices, slice_len},
                                {slice_len},
                                {slice_len}});                                
            }
    };

    run(dtype::Float32());
    run(dtype::Float16());
    run(dtype::BFloat16());
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
