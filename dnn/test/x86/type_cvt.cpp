/**
 * \file dnn/test/x86/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/benchmarker.h"
#include "test/common/checker.h"

#include "test/x86/fixture.h"

namespace megdnn {
namespace test {

TEST_F(X86, TYPE_CVT) {
    Checker<TypeCvt> checker(handle());
    NormalRNG rng(0, 127);
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
            dtype::Quantized8Asymm(2.0f, static_cast<uint8_t>(3))};

    for (size_t size : {1, 7, 15, 33}) {
        for (auto sdtype : dtypes)
            for (auto ddtype : dtypes) {
                checker.set_dtype(0, sdtype).set_dtype(1, ddtype).execs(
                        {{size}, {size}});
                TensorLayout non_contig_src(
                        {1, 10, 10, 12}, {10 * 10 * 18, 10 * 18, 18, 1}, sdtype);
                TensorLayout non_contig_dst({1, 10, 10, 12}, ddtype);
                checker.exec(TensorLayoutArray{non_contig_src, non_contig_dst});
            }
    }

    for (size_t size : {1, 7, 15, 33}) {
        checker.set_dtype(0, dtype::Uint16())
                .set_dtype(1, dtype::Float32())
                .execs({{size}, {size}});
    }
    TensorLayout non_contig_src(
            {1, 10, 10, 12}, {10 * 10 * 18, 10 * 18, 18, 1}, dtype::Uint16());
    TensorLayout non_contig_dst({1, 10, 10, 12}, dtype::Float32());
    checker.exec(TensorLayoutArray{non_contig_src, non_contig_dst});
}

TEST_F(X86, TYPE_CVT_NO_CONTIGUOUS) {
    UniformFloatRNG init(0, 100);
    Checker<TypeCvt> checker(handle());
    std::vector<DType> dtypes = {
            dtype::Float32(),
            dtype::Float16(),
            dtype::Int32(),
            dtype::Int8(),
            dtype::Uint8(),
            dtype::QuantizedS8(2.45f),
            dtype::Quantized8Asymm(4.54f, static_cast<uint8_t>(10)),
            dtype::QuantizedS32(3.23f)};
    for (auto sdtype : dtypes)
        for (auto ddtype : dtypes) {
            TensorLayout src({16, 128, 128}, {49152, 384, 3}, sdtype),
                    dst({16, 128, 128}, {16384, 128, 1}, ddtype);
            checker.set_rng(0, &init).execl({src, dst});
        }
}

TEST_F(X86, TYPE_CVT_2) {
    Checker<TypeCvt> checker(handle());
    UniformIntRNG rng{INT32_MIN >> 1, INT32_MAX >> 1};
    UniformIntRNG rng8{INT8_MIN >> 1, INT8_MAX >> 1};

    for (size_t size : {1, 7, 15, 33, 10000}) {
        checker.set_rng(0, &rng);
        checker.set_dtype(0, dtype::QuantizedS32(0.0000113264f))
                .set_dtype(
                        1, dtype::Quantized8Asymm(0.018909f, static_cast<uint8_t>(3)))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS32(0.0003f))
                .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3)))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS32(0.000815917f))
                .set_dtype(1, dtype::QuantizedS8(0.245121f))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS32(0.0003f))
                .set_dtype(1, dtype::QuantizedS8(0.2f))
                .execs({{size}, {size}});

        checker.set_rng(0, &rng8);

        //! we should not use so large random value, otherwise it may cause
        //! compute error
        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::QuantizedS8(0.245121f))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Quantized8Asymm(2.f, static_cast<uint8_t>(128)))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS32(0.0004f))
                .set_dtype(1, dtype::QuantizedS32(0.0002f))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS8(0.3f))
                .set_dtype(1, dtype::QuantizedS8(0.2f))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::Quantized8Asymm(0.3f, static_cast<uint8_t>(8)))
                .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3)))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS8(0.245121f))
                .set_dtype(1, dtype::QuantizedS32(0.000815917f))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::QuantizedS8(0.2f))
                .set_dtype(1, dtype::QuantizedS32(0.0003f))
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float16())
                .execs({{size}, {size}});

        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float32())
                .execs({{size}, {size}});
    }

    UniformIntRNG narrow_rng{-40000, 40000};
    checker.set_rng(0, &narrow_rng);
    checker.set_dtype(0, dtype::QuantizedS32(0.000163794f))
            .set_dtype(1, dtype::Quantized8Asymm(0.0479196f, static_cast<uint8_t>(144)))
            .execs({{1, 32, 24, 128}, {1, 32, 24, 128}});
}
#if MEGDNN_WITH_BENCHMARK
TEST_F(X86, BENCHMARK_TYPE_CVT) {
    auto handle_naive = create_cpu_handle(2);
    Benchmarker<TypeCvt> benchmarker(handle());
    Benchmarker<TypeCvt> benchmarker_naive(handle_naive.get());
    benchmarker_naive.set_display(false);
    benchmarker.set_display(false);
    constexpr size_t RUNS = 10;
    benchmarker_naive.set_times(RUNS);
    benchmarker.set_times(RUNS);
    auto run = [&](const TensorShapeArray& shapes, DType src_type, DType dst_type,
                   const char* msg) {
        benchmarker_naive.set_dtype(0, src_type).set_dtype(1, dst_type);
        benchmarker.set_dtype(0, src_type).set_dtype(1, dst_type);
        for (auto&& shape : shapes) {
            auto cur = benchmarker.execs({shape, shape}) / RUNS;
            auto naive = benchmarker_naive.execs({shape, shape}) / RUNS;
            const float computation = shape.total_nr_elems() * 1e-6;
            const float throughput = computation / cur;
            printf("run %s %s: naive=%fms cur=%fms "
                   "speedup=%f, throughput = %f Gops\n",
                   shape.to_string().c_str(), msg, naive, cur, naive / cur, throughput);
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
