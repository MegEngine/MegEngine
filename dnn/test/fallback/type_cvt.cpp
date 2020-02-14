/**
 * \file dnn/test/fallback/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/benchmarker.h"
#include "test/common/checker.h"

#include "test/fallback/fixture.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, TYPE_CVT) {
    Checker<TypeCvt> checker(handle());
    NormalRNG rng(128);
    checker.set_rng(0, &rng);

    std::vector<DType> dtypes = {
            dtype::Float32(),
            dtype::Float16(),
            dtype::Int32(),
            dtype::Int16(),
            dtype::Int8(),
            dtype::Uint8(),
            dtype::QuantizedS8(0.5f),
            dtype::QuantizedS32(0.5f),
            dtype::Quantized8Asymm(2.0f, static_cast<uint8_t>(3))
    };

    for (size_t size : {1, 7, 15, 33}) {
        for (auto sdtype : dtypes)
            for (auto ddtype : dtypes) {
                checker.set_dtype(0, sdtype).set_dtype(1, ddtype).execs(
                        {{size}, {size}});
            }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_TYPE_CVT) {
    auto handle_naive = create_cpu_handle(2);
    Benchmarker<TypeCvt> benchmarker(handle());
    Benchmarker<TypeCvt> benchmarker_naive(handle_naive.get());
    benchmarker_naive.set_display(false);
    benchmarker.set_display(false);
    constexpr size_t RUNS = 10;
    benchmarker_naive.set_times(RUNS);
    benchmarker.set_times(RUNS);
    auto run = [&](const TensorShapeArray& shapes, DType src_type,
                   DType dst_type, const char* msg) {

        benchmarker_naive.set_dtype(0, src_type).set_dtype(1, dst_type);
        benchmarker.set_dtype(0, src_type).set_dtype(1, dst_type);
        for (auto&& shape : shapes) {
            auto cur = benchmarker.execs({shape, shape}) / RUNS;
            auto naive = benchmarker_naive.execs({shape, shape}) / RUNS;
            printf("run %s %s: naive=%fms cur=%fms "
                   "speedup=%f\n",
                   shape.to_string().c_str(), msg, naive, cur, naive / cur);
        }
    };

    TensorShapeArray shapes = {{100000}, {1000000}};

    run(shapes, dtype::QuantizedS8(0.5f), dtype::QuantizedS8(0.2f),
        "QuantizedS8->QuantizedS8");
    run(shapes, dtype::QuantizedS32(0.5f),
        dtype::Quantized8Asymm(0.2f, static_cast<uint8_t>(3)),
        "QuantizedS32->Quantized8Asymm");
    run(shapes, dtype::Float32{}, dtype::Float16{}, "Float32->Float16");
    run(shapes, dtype::Float16{}, dtype::Float32{}, "Float16->Float32");
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
