/**
 * \file dnn/test/fallback/elemwise_multi_type.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/benchmarker.h"
#include "test/common/elemwise_multi_type.h"
#include "test/fallback/fixture.h"

using namespace megdnn;
using namespace test;

namespace {
template <typename tag>
class FALLBACK_ELEMWISE_MULTI_TYPE : public FALLBACK {};
TYPED_TEST_CASE(FALLBACK_ELEMWISE_MULTI_TYPE, elemwise_multi_type::test_types);
}  // anonymous namespace

TYPED_TEST(FALLBACK_ELEMWISE_MULTI_TYPE, run) {
    elemwise_multi_type::run_test<TypeParam>(this->handle());
}
#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, ELEMWISE_MULTI_TYPE_BENCHMARK_FMA3_INT16x32x32x32) {
    Benchmarker<ElemwiseMultiType> bench{handle()};
    bench.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_INT16x32x32x32});
    bench.set_dtype(0, dtype::Int16());
    bench.set_dtype(1, dtype::Int32());
    bench.set_dtype(2, dtype::Int32());
    UniformIntRNG rng{-100, 100};
    bench.set_rng(0, &rng);
    bench.set_rng(1, &rng);
    bench.set_rng(2, &rng);
    bench.set_adaptive_benchmark(0.8);
    constexpr size_t A = 32, B = 602, C = 103;
    auto time = bench.execs({{A, B, C}, {1, B, 1}, {1, B, 1}, {}}) * 1e-3;
    printf("computation: %.2fGFLOPS/s memory: %.2fGiB/s\n",
           A * B * C * 2 / time * 1e-9,
           (A * B * C * (2 + 4) + B * 8) / time / (1024.0 * 1024.0 * 1024.0));
}

TEST_F(FALLBACK, ELEMWISE_MULTI_TYPE_BENCHMARK_FMA3_IXxf32xf32xI8) {
    Benchmarker<ElemwiseMultiType> bench{handle()};
    bench.set_param({ElemwiseMultiType::Mode::FUSE_MUL_ADD3_IXxF32xF32xI8});
    std::array<DType, 3> src_types{
            {dtype::Int8{}, dtype::Int16{}, dtype::Int32{}}};
    bench.set_dtype(1, dtype::Float32());
    bench.set_dtype(2, dtype::Float32());
    UniformIntRNG rng{-100, 100};
    bench.set_rng(0, &rng);
    bench.set_adaptive_benchmark(0.8);
    constexpr size_t A = 328, B = 602;
    for (auto stype : src_types) {
        bench.set_dtype(0, stype);
        auto time = bench.execs({{A, B}, {1, B}, {1, B}, {}}) * 1e-3;
        printf("stype: %s, computation: %.2fGFLOPS/s memory: %.2fGiB/s\n",
               stype.name(), A * B * 2 / time * 1e-9,
               (A * B * (stype.size() + 1) + B * 8) / time /
                       (1024.0 * 1024.0 * 1024.0));
    }
}
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
