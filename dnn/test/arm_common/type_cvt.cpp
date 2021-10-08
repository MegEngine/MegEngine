/**
 * \file dnn/test/arm_common/type_cvt.cpp
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

#include "test/arm_common/fixture.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, TYPE_CVT) {
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
                .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3)))
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

TEST_F(ARM_COMMON, TYPE_CVT_16_F32) {
    Checker<TypeCvt> checker(handle());
    UniformIntRNG rng{INT16_MIN >> 1, INT16_MAX >> 1};

    for (size_t size : {3, 7, 15, 33, 10000}) {
        checker.set_rng(0, &rng);
        checker.set_dtype(0, dtype::Int16()).execs({{size}, {size}});
        checker.set_dtype(0, dtype::Uint16()).execs({{size}, {size}});
    }
    TensorLayout src_int16{
            {1, 96, 64, 120}, {128 * 64 * 96, 128 * 64, 128, 1}, dtype::Int16()};
    TensorLayout dst_int16{{1, 96, 64, 120}, dtype::Float32()};
    checker.execl({src_int16, dst_int16});

    TensorLayout src_uint16{
            {1, 96, 64, 120}, {128 * 64 * 96, 128 * 64, 128, 1}, dtype::Uint16()};
    TensorLayout dst_uint16{{1, 96, 64, 120}, dtype::Float32()};
    checker.execl({src_uint16, dst_uint16});
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_TYPE_CVT) {
    auto run = [&](const TensorShapeArray& shapes) {
        auto handle_fallback = create_cpu_handle(1);
        Benchmarker<TypeCvt> benchmarker(handle());
        Benchmarker<TypeCvt> benchmarker_fallback(handle_fallback.get());
        benchmarker_fallback.set_display(false);
        benchmarker.set_display(false);
        constexpr size_t RUNS = 50;
        benchmarker_fallback.set_times(RUNS);
        benchmarker.set_times(RUNS);

        auto bench = [&](const char* msg) {
            for (auto&& shape : shapes) {
                auto fallback = benchmarker_fallback.execs({shape, shape}) / RUNS;
                auto cur = benchmarker.execs({shape, shape}) / RUNS;
                printf("run %s %s: fallback=%fms "
                       "cur=%fms speedup=%f\n",
                       shape.to_string().c_str(), msg, fallback, cur, fallback / cur);
            }
        };

        benchmarker_fallback.set_dtype(0, dtype::QuantizedS32(0.25f))
                .set_dtype(1, dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(3)));
        benchmarker.set_dtype(0, dtype::QuantizedS32(0.25f))
                .set_dtype(1, dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(3)));
        bench("QuantizedS32->Quantized8Asymm");

        benchmarker_fallback
                .set_dtype(0, dtype::Quantized8Asymm(0.25f, static_cast<uint8_t>(9)))
                .set_dtype(1, dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(3)));
        benchmarker.set_dtype(0, dtype::Quantized8Asymm(0.25f, static_cast<uint8_t>(9)))
                .set_dtype(1, dtype::Quantized8Asymm(1.3f, static_cast<uint8_t>(3)));
        bench("Quantized8Asymm->Quantized8Asymm");

        benchmarker_fallback.set_dtype(0, dtype::QuantizedS32(0.25f))
                .set_dtype(1, dtype::QuantizedS8(1.3f));
        benchmarker.set_dtype(0, dtype::QuantizedS32(0.25f))
                .set_dtype(1, dtype::QuantizedS8(1.3f));
        bench("QuantizedS32->QuantizedS8");

        benchmarker_fallback.set_dtype(0, dtype::QuantizedS8(1.3f))
                .set_dtype(1, dtype::QuantizedS32(0.25f));
        benchmarker.set_dtype(0, dtype::QuantizedS32(1.3f))
                .set_dtype(1, dtype::QuantizedS8(0.25f));
        bench("QuantizedS8->QuantizedS32");

        benchmarker_fallback.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float32());
        benchmarker.set_dtype(0, dtype::Float16()).set_dtype(1, dtype::Float32());
        bench("Float16->Float32");

        benchmarker_fallback.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float16());
        benchmarker.set_dtype(0, dtype::Float32()).set_dtype(1, dtype::Float16());
        bench("Float32->Float16");

        benchmarker_fallback.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::QuantizedS8(0.245121f));
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::QuantizedS8(0.245121f));
        bench("Float32->QuantizedS8");

        benchmarker_fallback.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3)));
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3)));
        bench("Float32->Quantized8Asymm");
    };

    TensorShapeArray shapes = {{100000}, {1000000}};

    run(shapes);
}
#endif

}  // namespace test
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
