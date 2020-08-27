/**
 * \file dnn/test/rocm/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "test/common/elemwise.h"
#include "test/rocm/fixture.h"
#include "megdnn/oprs.h"
#include "test/common/tensor.h"
#include "test/common/rng.h"

#include "hip_header.h"
#include "src/rocm/miopen_with_check.h"

#include "test/rocm/benchmarker.h"

using namespace megdnn;
using namespace test;

namespace {
    void run_tensor_add(
            Handle *handle_rocm, 
            const TensorND &a, const TensorND &b,
            const TensorND &c) {
        auto opr = handle_rocm->create_operator<ElemwiseForward>();
        opr->param().mode = ElemwiseForward::Mode::ADD;
        hipProfilerStart();
        opr->exec({a, b}, c);
        hipProfilerStop();
    }

    using Mode = ElemwiseForward::Mode;
    template <Mode mode>
    void run_elemwise_benchmark(Handle* handle_rocm, Handle* handle_naive,
                                TensorShapeArray shapes, DType dtype) {
        auto benchmarker =
                ROCMBenchmarker<ElemwiseForward>(handle_rocm, handle_naive);
        benchmarker.set_display(true);
        ElemwiseForward::Param param;
        param.mode = mode;
        benchmarker.set_param(param);
        TensorShape dst_shp;
        ElemwiseForward::deduce_shape(shapes, dst_shp);
        shapes.push_back(dst_shp);
        for (size_t i = 0; i < shapes.size(); i++) {
            benchmarker.set_dtype(i, dtype); 
        }
        float io = 0.f;
        for (auto&& shp : shapes) {
            io += 1.f * shp.total_nr_elems() * dtype.size();
        }
        auto time_ms = benchmarker.execs(shapes);
        printf("io = %.3f GB, bandwidth = %.3f GB/s\n", io / 1e9,
               io / (1e6 * time_ms));
    }

}  // anonymous namespace

template <typename tag>
class ROCM_ELEMWISE : public ROCM {};
TYPED_TEST_CASE(ROCM_ELEMWISE, elemwise::test_types);
TYPED_TEST(ROCM_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle_rocm());
}

//! the memory of this test case is too large, sometimes will fail on tx1
TEST_F(ROCM, ELEMWISE_BENCHMARK_DENSE) {
    constexpr size_t A = 1024 * 1024 * 64,
              S0 = 64, S1 = 256, S2 = 64, S3 = 64;
    static_assert(A == S0 * S1 * S2 * S3, "bad value");
    SyncedTensor<>
        t0(handle_rocm(), {TensorShape{S0, S1, S2, S3}, dtype::Float32()}),
        t1(handle_rocm(), {TensorShape{S0, S1, S2, S3}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    run_tensor_add(handle_rocm(),
            t0.tensornd_dev(), t0.tensornd_dev(), t1.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host();
    for (size_t i = 0; i < A; ++ i) {
        ASSERT_EQ(p0[i] + p0[i], p1[i]) << "at index " << i << "/" << A;
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ROCM, ELEMWISE_BENCHMARK_BCAST_101) {
    constexpr size_t A = 511, B = 509, C0 = 23, C1 = 23, C = C0 * C1;
    SyncedTensor<>
        t0(handle_rocm(), {TensorShape{A, B, C0, C1}, dtype::Float32()}),
        t1(handle_rocm(), {TensorShape{1, B, 1, 1}, dtype::Float32()}),
        t2(handle_rocm(), {TensorShape{A, B, C0, C1}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    run_tensor_add(handle_rocm(),
            t0.tensornd_dev(), t1.tensornd_dev(), t2.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host(), p2 = t2.ptr_host();
    for (size_t i = 0; i < A; ++ i) {
        for (size_t j = 0; j < B; ++ j) {
            for (size_t k = 0; k < C; ++ k) {
                auto off = i * B * C + j * C + k;
                ASSERT_EQ(p0[off] + p1[j], p2[off]);
            }
        }
    }
}

TEST_F(ROCM, ELEMWISE_BENCHMARK_BCAST_10) {
    constexpr size_t A = 11583, B = 11587;
    SyncedTensor<> t0(handle_rocm(), {TensorShape{A, B}, dtype::Float32()}),
                   t1(handle_rocm(), {TensorShape{1, B}, dtype::Float32()}),
                   t2(handle_rocm(), {TensorShape{A, B}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    run_tensor_add(handle_rocm(),
            t0.tensornd_dev(), t1.tensornd_dev(), t2.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host(), p2 = t2.ptr_host();
    for (size_t i = 0; i < A; ++ i) {
        for (size_t j = 0; j < B; ++ j) {
            auto off = i * B + j;
            ASSERT_EQ(p0[off] + p1[j], p2[off]);
        }
    }
}

TEST_F(ROCM, ELEMWISE_BENCHMARK_BCAST_01) {
    constexpr size_t A = 11583, B = 11587;
    SyncedTensor<> t0(handle_rocm(), {TensorShape{1, A, B}, dtype::Float32()}),
                   t1(handle_rocm(), {TensorShape{1, A, 1}, dtype::Float32()}),
                   t2(handle_rocm(), {TensorShape{1, A, B}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    run_tensor_add(handle_rocm(),
            t0.tensornd_dev(), t1.tensornd_dev(), t2.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host(), p2 = t2.ptr_host();
    for (size_t i = 0; i < A; ++ i) {
        for (size_t j = 0; j < B; ++ j) {
            auto off = i * B + j;
            ASSERT_EQ(p0[off] + p1[i], p2[off]);
        }
    }
}

TEST_F(ROCM, ELEMWISE_BENCHMARK) {
    using Mode = ElemwiseForward::Mode;
    run_elemwise_benchmark<Mode::ADD>(handle_rocm(), handle_naive(false),
                                      {{32, 128, 56, 56}, {32, 128, 56, 56}},
                                      dtype::Float32());
    run_elemwise_benchmark<Mode::ADD>(handle_rocm(), handle_naive(false),
                                      {{32, 128, 56, 56}, {1, 128, 1, 1}},
                                      dtype::Float32());
    run_elemwise_benchmark<Mode::FUSE_ADD_RELU>(handle_rocm(), handle_naive(false),
                                      {{32, 128, 56, 56}, {1, 128, 1, 1}},
                                      dtype::Float32());
    run_elemwise_benchmark<Mode::FUSE_MUL_ADD3>(
            handle_rocm(), handle_naive(false),
            {{32, 128, 56, 56}, {1, 128, 1, 1}, {32, 128, 56, 56}},
            dtype::Float32());
}

TEST_F(ROCM, ELEMWISE_BENCHMARK_PEAK_BANDWIDTH) {
    using Mode = ElemwiseForward::Mode;
    run_elemwise_benchmark<Mode::FUSE_MUL_ADD4>(
            handle_rocm(), handle_naive(false),
            {{10000, 10000}, {10000, 10000}, {10000, 10000}, {10000, 10000}},
            dtype::Float32());
}
#endif

// vim: syntax=cpp.doxygen

