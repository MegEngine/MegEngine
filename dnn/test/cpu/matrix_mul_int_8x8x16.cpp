/**
 * \file dnn/test/cpu/matrix_mul_int_8x8x16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/convolution.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

namespace megdnn {
namespace test {

TEST_F(CPU, MATRIX_MUL_INT_8_8_16)
{
    Checker<MatrixMul> checker(handle());
    param::MatrixMul param;
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int16());
    checker.set_param(param);
    for (size_t b: {1, 2, 3})
    for (size_t i: {10, 20})
    for (size_t o: {11, 22})
    {
        checker.exec({{b, i}, {i, o}, {}});
    }
    for (size_t m = 16; m <= 512; m*=4)
    for (size_t n = 16; n <= 512; n*=4)
    for (size_t k = 16; k <= 512; k*=4)
    {
        checker.exec({{m, k}, {k, n}, {}});

        checker.exec({{m + 1, k}, {k, n}, {}});
        checker.exec({{m + 5, k}, {k, n}, {}});
        checker.exec({{m + 7, k}, {k, n}, {}});

        checker.exec({{m, k}, {k, n + 15}, {}});
        checker.exec({{m, k}, {k, n + 9}, {}});
        checker.exec({{m, k}, {k, n + 8}, {}});
        checker.exec({{m, k}, {k, n + 7}, {}});
        checker.exec({{m, k}, {k, n + 1}, {}});

        checker.exec({{m+1, k}, {k, n + 9}, {}});
        checker.exec({{m+7, k}, {k, n + 15}, {}});
        checker.exec({{m+7, k}, {k, n + 7}, {}});
    }
    // test transpose scenerio
    {
        for (int mask = 0; mask < 4; ++mask) {
            param::MatrixMul param;
            param.transposeA = (mask & 1);
            param.transposeB = (mask & 2);
            checker.set_param(param);
            size_t m = 100, n = 101, k = 102;
            TensorShape A = param.transposeA ?
                TensorShape({k, m}) : TensorShape({m, k});
            TensorShape B = param.transposeB ?
                TensorShape({n, k}) : TensorShape({k, n});
            checker.exec({A, B, {}});
        }
    }
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(CPU, BENCHMARK_MATRIX_MUL_INT8_INT8_INT16)
{
    bool verbose = getenv("MEGDNN_BENCH_VERBOSE");
    using Param = param::MatrixMul;
    double speedup_sum = 0, speedup_wsum = 0;
    auto run = [&](const TensorShapeArray &shapes,
            const Param& param) {
        TensorLayoutArray layouts;
        layouts.emplace_back(shapes[0], dtype::Int8());
        layouts.emplace_back(shapes[1], dtype::Int8());
        layouts.emplace_back(shapes[2], dtype::Int16());
        Benchmarker<MatrixMul>
            benchmarker_cpu(handle());
        param::MatrixMul param_int(param);
        benchmarker_cpu.set_param(param_int);
        Benchmarker<MatrixMul> benchmarker_float(handle());
        benchmarker_float.set_param(param);
        auto t2 = benchmarker_cpu.set_display(false).
            set_adaptive_benchmark(0.01).execl(layouts);
        auto t4 = benchmarker_float.set_display(false).
            set_adaptive_benchmark(0.01).exec(shapes);
        if (t2 > t4 || verbose) {
            std::cout << "MatA=" << shapes[0].to_string()
                << " MatB=" << shapes[1].to_string()
                << " float=" << t4 << "ms"
                << " int=" << t2 << "ms"
                << " speedup=" << t4/t2 << std::endl;
        }
        speedup_sum += t4 / t2;
        speedup_wsum += 1;
    };
    for (size_t m = 16; m <= 256; m*=4)
    for (size_t k = 16; k <= 256; k*=4)
    for (size_t n = 16; n <= 1024; n*=4)
    {
        Param param;
        run({{m, k}, {k, n}, {}}, param);
        run({{m, k}, {k, n + 8}, {}}, param);
        run({{m, k}, {k, n + 15}, {}}, param);

        run({{m + 5, k}, {k, n}, {}}, param);
        run({{m + 7, k}, {k, n}, {}}, param);
    }
    printf("average speedup: %.3f\n", speedup_sum / speedup_wsum);
}

#endif

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen

