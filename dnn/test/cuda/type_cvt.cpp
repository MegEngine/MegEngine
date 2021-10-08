/**
 * \file dnn/test/cuda/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, TYPE_CVT) {
    UniformFloatRNG init(0, 20);
    std::vector<DType> dtypes = {dtype::Float32(), dtype::Float16(), dtype::Int32(),
                                 dtype::Int16(),   dtype::Int8(),    dtype::Uint8()};
    for (auto sdtype : dtypes)
        for (auto ddtype : dtypes) {
            TensorLayout src({10, 10}, sdtype), dst({10, 10}, ddtype);
            Checker<TypeCvt> checker(handle_cuda());
            checker.set_rng(0, &init).exec(TensorLayoutArray{src, dst});

            TensorLayout non_contig_src(
                    {1, 96, 64, 120}, {96 * 64 * 128, 64 * 128, 128, 1}, sdtype);
            TensorLayout non_contig_dst({1, 96, 64, 120}, ddtype);
            checker.exec(TensorLayoutArray{non_contig_src, non_contig_dst});
        }
}

TEST_F(CUDA, BENCHMARK_TYPE_CVT_LAST_NOT_CONTIG) {
    const size_t RUNS = 3;

    auto run = [&](TensorLayout src, TensorLayout dst) {
        Benchmarker<TypeCvt> benchmarker(handle_cuda());
        auto&& layout = src;
        benchmarker.set_times(RUNS);
        dst.init_contiguous_stride();
        auto used = benchmarker.execl({src, dst});
        printf("layout: %s bandwith: %f gbps/s\n", layout.to_string().c_str(),
               2 * layout.total_nr_elems() * layout.dtype.size() * RUNS / used * 1000 /
                       (1024 * 1024 * 1024));
    };

    TensorLayout src({16, 128, 128}, {49152, 384, 3}, dtype::Float32()),
            dst({16, 128, 128}, {16384, 128, 1}, dtype::Float32());
    run(src, dst);
}

TEST_F(CUDA, QUANTIZED_TYPECVT) {
    UniformIntRNG int_rng{-66, 66};
    Checker<TypeCvt> checker(handle_cuda());
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng);

    auto set_err = [&](const DType& dst_dtype) {
        if (dst_dtype.category() == DTypeCategory::FLOAT)
            checker.set_epsilon(1e-6);
        else {
            checker.set_epsilon(1);
        }
    };

    auto run = [&](const DType& src_dtype, const DType& dst_dtype) {
        set_err(dst_dtype);
        checker.set_dtype(0, src_dtype)
                .set_dtype(1, dst_dtype)
                .execs({{20, 3, 224, 224}, {20, 3, 224, 224}});
        set_err(src_dtype);
        checker.set_dtype(0, dst_dtype)
                .set_dtype(1, src_dtype)
                .execs({{20, 3, 224, 224}, {20, 3, 224, 224}});
    };

    run(dtype::Float32(), dtype::QuantizedS8(3.0f));
    run(dtype::Float16(), dtype::QuantizedS8(3.0f));
    run(dtype::Int32(), dtype::QuantizedS32(5.0f));
    run(dtype::Int8(), dtype::QuantizedS32(10.0f));

    run(dtype::Float32(), dtype::QuantizedS8(2e-3f));
    run(dtype::Float16(), dtype::QuantizedS8(1e-3f));
    run(dtype::Int32(), dtype::QuantizedS32(1e-3f));
    run(dtype::Int8(), dtype::QuantizedS32(7e-4f));

    run(dtype::QuantizedS8(3.0f), dtype::QuantizedS8(10.0f));
    run(dtype::QuantizedS32(3.0f), dtype::QuantizedS8(10.0f));
    run(dtype::QuantizedS8(3.0f), dtype::QuantizedS32(10.0f));
    run(dtype::QuantizedS32(3.0f), dtype::QuantizedS32(10.0f));

    run(dtype::QuantizedS8(1e-3f), dtype::QuantizedS8(5e-3f));
    run(dtype::QuantizedS32(2e-3f), dtype::QuantizedS8(9e-4f));
    run(dtype::QuantizedS8(9e-4f), dtype::QuantizedS32(7e-4f));
    run(dtype::QuantizedS32(5e-3f), dtype::QuantizedS32(1e-3f));

    run(dtype::Quantized8Asymm(5.0f, (uint8_t)128), dtype::Float32());
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)124), dtype::Float16());
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)30), dtype::Int8());
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)20), dtype::Int32());
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)10), dtype::QuantizedS8(10.5f));
    run(dtype::Quantized8Asymm(5.0f, (uint8_t)18), dtype::QuantizedS32(10.5f));

    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)128), dtype::Float32());
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)124), dtype::Float16());
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)30), dtype::Int8());
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)20), dtype::Int32());
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)10), dtype::QuantizedS8(2e-3f));
    run(dtype::Quantized8Asymm(1e-3f, (uint8_t)18), dtype::QuantizedS32(7e-4f));
}

TEST_F(CUDA, QUANTIZED_TYPECVT_4BIT) {
    UniformIntRNG int_rng{-8, 8};
    Checker<TypeCvt> checker(handle_cuda());
    checker.set_rng(0, &int_rng).set_rng(1, &int_rng);

    auto set_err = [&](const DType& dst_dtype) {
        if (dst_dtype.category() == DTypeCategory::FLOAT)
            checker.set_epsilon(1e-6);
        else {
            checker.set_epsilon(1e-3);
        }
    };

    auto run = [&](const DType& src_dtype, const DType& dst_dtype) {
        set_err(dst_dtype);
        checker.set_dtype(0, src_dtype)
                .set_dtype(1, dst_dtype)
                .execs({{16, 3, 224, 223}, {16, 3, 224, 223}})
                .execs({{16, 3, 224, 1}, {16, 3, 224, 1}});
        set_err(src_dtype);
        checker.set_dtype(0, dst_dtype)
                .set_dtype(1, src_dtype)
                .execs({{16, 3, 224, 223}, {16, 3, 224, 223}})
                .execs({{16, 3, 224, 1}, {16, 3, 224, 1}});
    };

    run(dtype::Quantized4Asymm{1.19990518f, 8}, dtype::Quantized8Asymm{1.f, 128});
    run(dtype::QuantizedS4{1.19990518f}, dtype::QuantizedS8{1.19990518f});
    run(dtype::QuantizedS4{1.19990518f}, dtype::Quantized4Asymm{1.19990518f, 8});
}

TEST_F(CUDA, TYPE_CVT_BFLOAT16) {
    Checker<TypeCvt> checker(handle_cuda());
    UniformFloatRNG rng(-20, 20);
    checker.set_rng(0, &rng);
    std::vector<DType> dtypes = {
            dtype::Float32(), dtype::Float16(), dtype::Int32(), dtype::Int16(),
            dtype::Int8()};
    for (auto sdtype : dtypes) {
        TensorLayout src({10, 10}, sdtype), dst({10, 10}, dtype::BFloat16());
        checker.exec(TensorLayoutArray{src, dst});
        TensorLayout src2({10, 10}, dtype::BFloat16()), dst2({10, 10}, sdtype);
        checker.exec(TensorLayoutArray{src2, dst2});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_TYPE_CVT) {
    UniformIntRNG rng{-128, 127};
    auto run = [&](TensorLayout src, TensorLayout dst) {
        Benchmarker<TypeCvt> benchmarker(handle_cuda());
        auto&& layout = src;
        size_t nr_times = 1000;
        benchmarker.set_times(nr_times);
        dst.init_contiguous_stride();
        auto used = benchmarker.set_dtype(0, src.dtype)
                            .set_dtype(1, dst.dtype)
                            .set_rng(0, &rng)
                            .execl({src, dst}) /
                    nr_times;
        printf("layout: %s time %.2fms, bandwith: %f GB/s\n",
               layout.to_string().c_str(), used,
               (1.f * src.dtype.size() * src.total_nr_elems() +
                dst.dtype.size() * dst.total_nr_elems()) /
                       (used * 1e6));
    };

    TensorLayout src({16, 128, 128}, {49152, 384, 3}, dtype::Int8()),
            dst({16, 128, 128}, {16384, 128, 1}, dtype::QuantizedS8(3.f));
    run(src, dst);
    // NCHW astype(float32)
    src = TensorLayout{{256, 256, 56, 56}, dtype::QuantizedS8(3.f)};
    dst = TensorLayout{{256, 256, 56, 56}, dtype::Float32()};
    run(src, dst);
    // NCHW astype(qint8)
    src = TensorLayout{{256, 256, 56, 56}, dtype::Float32()};
    dst = TensorLayout{{256, 256, 56, 56}, dtype::QuantizedS8(3.f)};
    run(src, dst);
    // NCHW4 astype(float32)
    src = TensorLayout{{256, 64, 56, 56, 4}, dtype::QuantizedS8(3.f)};
    dst = TensorLayout{{256, 64, 56, 56, 4}, dtype::Float32()};
    run(src, dst);
    // NCHW4 astype(qint8)
    src = TensorLayout{{256, 64, 56, 56, 4}, dtype::Float32()};
    dst = TensorLayout{{256, 64, 56, 56, 4}, dtype::QuantizedS8(3.f)};
    run(src, dst);
    // NCHW32 astype(float32)
    src = TensorLayout{{256, 8, 56, 56, 32}, dtype::QuantizedS8(3.f)};
    dst = TensorLayout{{256, 8, 56, 56, 32}, dtype::Float32()};
    run(src, dst);
    // NCHW32 astype(qint8)
    src = TensorLayout{{256, 8, 56, 56, 32}, dtype::Float32()};
    dst = TensorLayout{{256, 8, 56, 56, 32}, dtype::QuantizedS8(3.f)};
    run(src, dst);
    // quantize to quantize
    src = TensorLayout{
            {256, 8, 56, 56, 32}, dtype::Quantized8Asymm(5.f, (uint8_t)(30))};
    dst = TensorLayout{{256, 8, 56, 56, 32}, dtype::QuantizedS8(3.f)};
    run(src, dst);
    // quantize to quantize
    src = TensorLayout{{256, 8, 56, 56, 32}, dtype::QuantizedS8(3.f)};
    dst = TensorLayout{
            {256, 8, 56, 56, 32}, dtype::Quantized8Asymm(5.f, (uint8_t)(30))};
    run(src, dst);
}

TEST_F(CUDA, BENCHMARK_TYPE_CVT_Q4) {
    UniformIntRNG rng{-7, 7};
    auto run = [&](TensorLayout src, TensorLayout dst) {
        Benchmarker<TypeCvt> benchmarker(handle_cuda());
        auto&& layout = src;
        size_t nr_times = 1000;
        benchmarker.set_times(nr_times);
        dst.init_contiguous_stride();
        auto used = benchmarker.set_dtype(0, src.dtype)
                            .set_dtype(1, dst.dtype)
                            .set_rng(0, &rng)
                            .execl({src, dst}) /
                    nr_times;
        printf("layout: %s time %.2fms, bandwith: %f GB/s\n",
               layout.to_string().c_str(), used,
               (1.f * src.access_bytes() + dst.access_bytes()) / (used * 1e6));
    };

    // NCHW astype(float32)
    TensorLayout src = TensorLayout{{256, 256, 56, 56}, dtype::QuantizedS8(1.f)};
    TensorLayout dst = TensorLayout{{256, 256, 56, 56}, dtype::QuantizedS4(1.f)};

    run(src, dst);

    src = TensorLayout{{256, 4, 56, 56, 64}, dtype::QuantizedS4(1.f)};
    dst = TensorLayout{{256, 4, 56, 56, 64}, dtype::QuantizedS8(1.f)};
    run(src, dst);
}

#endif

// vim: syntax=cpp.doxygen
