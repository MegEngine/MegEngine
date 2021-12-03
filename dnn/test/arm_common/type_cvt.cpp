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
#include "test/common/task_record_check.h"

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

TEST_F(ARM_COMMON, TYPE_CVT_RECORD) {
    TaskRecordChecker<TypeCvt> checker(0);
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

TEST_F(ARM_COMMON, TYPE_CVT_NONCONTIGUOUS) {
    UniformIntRNG rng32{INT32_MIN >> 1, INT32_MAX >> 1};
    UniformIntRNG rng16{INT16_MIN >> 1, INT16_MAX >> 1};
    UniformIntRNG rng8{INT8_MIN >> 1, INT8_MAX >> 1};

    Checker<TypeCvt> checker(handle());

    size_t N = 1;
    size_t C = 96;
    size_t H = 64;
    size_t W = 120;
    TensorShape shape{N, C, H, W};
    std::vector<ptrdiff_t> stride{
            static_cast<long>(C * H * (W + 8)), static_cast<long>(H * (W + 8)),
            static_cast<long>(W + 8), 1};
    TensorLayout src, dst;

    //! float32 -> float16
    src = TensorLayout{shape, stride, dtype::Float32()};
    dst = TensorLayout{shape, dtype::Float16()};
    checker.execl({src, dst});

    //! float16 -> float32
    src = TensorLayout{shape, stride, dtype::Float16()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! float -> s8
    src = TensorLayout{shape, stride, dtype::Float32()};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.245121f)};
    checker.execl({src, dst});

    //! float -> as8
    src = TensorLayout{shape, stride, dtype::Float32()};
    dst = TensorLayout{shape, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    checker.set_rng(0, &rng32);
    //! s32 -> as8
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0000113264f)};
    dst = TensorLayout{
            shape, dtype::Quantized8Asymm(0.018909f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0003f)};
    dst = TensorLayout{shape, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    //! s32 -> s8
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.000815917f)};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.245121f)};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0003f)};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.2f)};
    checker.execl({src, dst});

    checker.set_rng(0, &rng8);
    //! s32 -> s32
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0004f)};
    dst = TensorLayout{shape, dtype::QuantizedS32(0.0002f)};
    checker.execl({src, dst});

    //! s8 -> s8
    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.3f)};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.2f)};
    checker.execl({src, dst});

    //! as8 -> as8
    src = TensorLayout{
            shape, stride, dtype::Quantized8Asymm(0.3f, static_cast<uint8_t>(8))};
    dst = TensorLayout{shape, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    //! s8 -> s32
    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.245121f)};
    dst = TensorLayout{shape, dtype::QuantizedS32(0.000815917f)};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.2f)};
    dst = TensorLayout{shape, dtype::QuantizedS32(0.0003f)};
    checker.execl({src, dst});

    //! s8 -> float
    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.3f)};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! as8 -> float
    src = TensorLayout{
            shape, stride, dtype::Quantized8Asymm(0.3f, static_cast<uint8_t>(8))};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! int8/uint8 -> float
    src = TensorLayout{shape, stride, dtype::Int8()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::Uint8()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! int16/uint16 -> float
    checker.set_rng(0, &rng16);
    for (size_t size : {3, 7, 15, 33, 10000}) {
        checker.set_dtype(0, dtype::Int16()).execs({{size}, {size}});
        checker.set_dtype(0, dtype::Uint16()).execs({{size}, {size}});
    }

    src = TensorLayout{shape, stride, dtype::Int16()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::Uint16()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    UniformIntRNG narrow_rng{-40000, 40000};
    checker.set_rng(0, &narrow_rng);
    //! s32 -> as8
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.000163794f)};
    dst = TensorLayout{
            shape, dtype::Quantized8Asymm(0.0479196f, static_cast<uint8_t>(144))};
    checker.execl({src, dst});
}

TEST_F(ARM_COMMON, TYPE_CVT_MONOTONOUS) {
    UniformIntRNG rng32{INT32_MIN >> 1, INT32_MAX >> 1};
    UniformIntRNG rng16{INT16_MIN >> 1, INT16_MAX >> 1};
    UniformIntRNG rng8{INT8_MIN >> 1, INT8_MAX >> 1};

    Checker<TypeCvt> checker(handle());

    size_t N = 1;
    size_t C = 96;
    size_t H = 64;
    size_t W = 120;
    TensorShape shape{N, C, H, W};
    std::vector<ptrdiff_t> stride{
            static_cast<long>((C + 8) * (H + 8) * (W + 8)),
            static_cast<long>((H + 8) * (W + 8)), static_cast<long>(W + 8), 1};
    TensorLayout src, dst;

    //! float32 -> float16
    src = TensorLayout{shape, stride, dtype::Float32()};
    dst = TensorLayout{shape, dtype::Float16()};
    checker.execl({src, dst});

    //! float16 -> float32
    src = TensorLayout{shape, stride, dtype::Float16()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! float -> s8
    src = TensorLayout{shape, stride, dtype::Float32()};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.245121f)};
    checker.execl({src, dst});

    //! float -> as8
    src = TensorLayout{shape, stride, dtype::Float32()};
    dst = TensorLayout{shape, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    checker.set_rng(0, &rng32);
    //! s32 -> as8
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0000113264f)};
    dst = TensorLayout{
            shape, dtype::Quantized8Asymm(0.018909f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0003f)};
    dst = TensorLayout{shape, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    //! s32 -> s8
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.000815917f)};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.245121f)};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0003f)};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.2f)};
    checker.execl({src, dst});

    checker.set_rng(0, &rng8);
    //! s32 -> s32
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.0004f)};
    dst = TensorLayout{shape, dtype::QuantizedS32(0.0002f)};
    checker.execl({src, dst});

    //! s8 -> s8
    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.3f)};
    dst = TensorLayout{shape, dtype::QuantizedS8(0.2f)};
    checker.execl({src, dst});

    //! as8 -> as8
    src = TensorLayout{
            shape, stride, dtype::Quantized8Asymm(0.3f, static_cast<uint8_t>(8))};
    dst = TensorLayout{shape, dtype::Quantized8Asymm(0.1f, static_cast<uint8_t>(3))};
    checker.execl({src, dst});

    //! s8 -> s32
    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.245121f)};
    dst = TensorLayout{shape, dtype::QuantizedS32(0.000815917f)};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.2f)};
    dst = TensorLayout{shape, dtype::QuantizedS32(0.0003f)};
    checker.execl({src, dst});

    //! s8 -> float
    src = TensorLayout{shape, stride, dtype::QuantizedS8(0.3f)};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! as8 -> float
    src = TensorLayout{
            shape, stride, dtype::Quantized8Asymm(0.3f, static_cast<uint8_t>(8))};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    //! int8/uint8 -> float
    src = TensorLayout{shape, stride, dtype::Int8()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::Uint8()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::Int16()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    src = TensorLayout{shape, stride, dtype::Uint16()};
    dst = TensorLayout{shape, dtype::Float32()};
    checker.execl({src, dst});

    UniformIntRNG narrow_rng{-40000, 40000};
    checker.set_rng(0, &narrow_rng);
    //! s32 -> as8
    src = TensorLayout{shape, stride, dtype::QuantizedS32(0.000163794f)};
    dst = TensorLayout{
            shape, dtype::Quantized8Asymm(0.0479196f, static_cast<uint8_t>(144))};
    checker.execl({src, dst});
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
