/**
 * \file dnn/test/cuda/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/elemwise.h"
#include "test/cuda/fixture.h"
#include "megdnn/oprs.h"
#include "test/common/tensor.h"
#include "test/common/rng.h"
#include "./utils.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"

#include <cudnn.h>
#include <cuda_profiler_api.h>

using namespace megdnn;
using namespace test;

#define cudnn_check(e) megdnn_assert((e) == CUDNN_STATUS_SUCCESS)
namespace {
__attribute__((unused)) cudnnTensorDescriptor_t make_cudnn_tensor_desc(
        const TensorLayout& ly) {
    megdnn_assert(ly.ndim && ly.ndim <= 4 && ly.is_contiguous());
    int dim[4] = {1, 1, 1, 1}, stride[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < ly.ndim; ++i) {
        dim[i] = ly.shape[i];
        stride[i] = ly.stride[i];
    }
    cudnnTensorDescriptor_t ret;
    cudnn_check(cudnnCreateTensorDescriptor(&ret));
    // cudnn requires tensors to be at-least 4D
    cudnn_check(cudnnSetTensor4dDescriptorEx(ret, CUDNN_DATA_FLOAT, dim[0],
                                             dim[1], dim[2], dim[3], stride[0],
                                             stride[1], stride[2], stride[3]));

    return ret;
}

void run_tensor_add(Handle* handle_cuda, const TensorND& a, const TensorND& b,
                    const TensorND& c) {
#if 1
    cudnnHandle_t cudnn_handle;
    cudnn_check(cudnnCreate(&cudnn_handle));
    cuda_check(cudaDeviceSynchronize());
    cuda_check(cudaMemcpy(c.raw_ptr, a.raw_ptr, a.layout.span().dist_byte(),
                          cudaMemcpyDeviceToDevice));

    auto bdesc = make_cudnn_tensor_desc(b.layout),
         cdesc = make_cudnn_tensor_desc(c.layout);

    float alpha = 1, beta = 1;
    cudaProfilerStart();
    cudnn_check(cudnnAddTensor(cudnn_handle, &alpha, bdesc, b.raw_ptr, &beta,
                               cdesc, c.raw_ptr));
    cudaProfilerStop();

    cudnn_check(cudnnDestroyTensorDescriptor(cdesc));
    cudnn_check(cudnnDestroyTensorDescriptor(bdesc));
    cudnn_check(cudnnDestroy(cudnn_handle));

    cuda_check(cudaMemset(c.raw_ptr, 0, c.layout.span().dist_byte()));
    cuda_check(cudaDeviceSynchronize());
#endif

    auto opr = handle_cuda->create_operator<ElemwiseForward>();
    opr->param().mode = ElemwiseForward::Mode::ADD;
    cudaProfilerStart();
    opr->exec({a, b}, c);
    cudaProfilerStop();
}

}  // anonymous namespace

template <typename tag>
class CUDA_ELEMWISE : public CUDA {};
TYPED_TEST_CASE(CUDA_ELEMWISE, elemwise::test_types);
TYPED_TEST(CUDA_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle_cuda());
}

TEST_F(CUDA, ELEMWISE_IBYTE) {
    Checker<ElemwiseForward> checker(handle_cuda());
    using Mode = ElemwiseForward::Param::Mode;
    UniformIntRNG i_rng{-128, 127};
    UniformIntRNG ui_rng{0, 255};
    checker.set_rng(0, &i_rng);
    auto run_unary = [&](size_t N, Mode mode, DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype);
        checker.execs({{N}, {}});
    };
#define RUN_UNARY_IBYTE(_dt)         \
    run_unary(100, Mode::RELU, _dt); \
    run_unary(100, Mode::ABS, _dt);
    RUN_UNARY_IBYTE(dtype::Int8());
    checker.set_rng(0, &i_rng);
    RUN_UNARY_IBYTE(dtype::Uint8());
#undef RUN_UNARY_IBYTE
    auto run_binary = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                          DType dtype) {
        checker.set_param(mode).set_dtype(0, dtype).set_dtype(1, dtype);
        checker.execs({{5}, {5}, {}});
        checker.execs({{4}, {4}, {}});
        checker.execs({{4}, {1}, {}});
        checker.execs({{N, C / 4, H, W, 4}, {N, C / 4, H, W, 4}, {}});
        checker.execs({{N, C / 4, H, W, 4}, {1, C / 4, 1, 1, 4}, {}});
        checker.execs({{N, C / 32, H, W, 32}, {N, C / 32, H, W, 32}, {}});
        checker.execs({{N, C / 32, H, W, 32}, {1, C / 32, 1, 1, 32}, {}});
        checker.execs({{3, 5, 7}, {3, 5, 7}, {}});
        checker.execs({{3, 5, 7}, {3, 5, 1}, {}});
        checker.execs({{3, 5, 1}, {3, 5, 7}, {}});
        checker.execs({{1}, {3, 5, 7}, {}});
        checker.execs({{3, 5, 7}, {1}, {}});
    };
#define RUN_BINARY_IBYTE(_dt)                  \
    run_binary(4, 32, 10, 10, Mode::ADD, _dt); \
    run_binary(4, 32, 10, 10, Mode::MUL, _dt); \
    run_binary(4, 32, 10, 10, Mode::MAX, _dt); \
    run_binary(4, 32, 10, 10, Mode::MIN, _dt); \
    run_binary(4, 32, 10, 10, Mode::SUB, _dt);
    checker.set_rng(0, &i_rng).set_rng(1, &i_rng);
    RUN_BINARY_IBYTE(dtype::Int8());
    checker.set_rng(0, &ui_rng).set_rng(1, &ui_rng);
    RUN_BINARY_IBYTE(dtype::Uint8());
#undef RUN_BINARY_IBYTE
    auto run_ternary = [&](size_t N, size_t C, size_t H, size_t W, Mode mode,
                           DType dtype) {
        checker.set_param(mode)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype);
        checker.execs({{5}, {5}, {5}, {}});
        checker.execs({{4}, {4}, {1}, {}});
        checker.execs({{N, C / 4, H, W, 4},
                       {N, C / 4, H, W, 4},
                       {N, C / 4, H, W, 4},
                       {}});
        checker.execs({{N, C / 4, H, W, 4},
                       {1, C / 4, 1, 1, 4},
                       {1, C / 4, 1, 1, 4},
                       {}});
        checker.execs({{N, C / 32, H, W, 32},
                       {N, C / 32, H, W, 32},
                       {N, C / 32, H, W, 32},
                       {}});
        checker.execs({{N, C / 32, H, W, 32},
                       {1, C / 32, 1, 1, 32},
                       {1, C / 32, 1, 1, 32},
                       {}});
        checker.execs({{1}, {3, 5, 7}, {3, 5, 7}, {}});
        checker.execs({{3, 5, 7}, {3, 5, 1}, {3, 5, 1}, {}});
        checker.execs({{3, 5, 1}, {3, 5, 7}, {3, 5, 1}, {}});
        checker.execs({{1}, {3, 5, 7}, {1}, {}});
        checker.execs({{3, 5, 7}, {1}, {3, 5, 7}, {}});
    };
#define RUN_TERNARY_IBYTE(_dt) \
    run_ternary(4, 32, 10, 10, Mode::FUSE_MUL_ADD3, _dt);
    checker.set_rng(0, &i_rng).set_rng(1, &i_rng);
    RUN_TERNARY_IBYTE(dtype::Int8());
    checker.set_rng(0, &ui_rng).set_rng(1, &ui_rng);
    RUN_TERNARY_IBYTE(dtype::Uint8());
#undef RUN_TERNARY_IBYTE
}

// from common/elemwise.cpp
TEST_F(CUDA, ELEMWISE_BFLOAT16) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle_cuda());

    // unary
#define UNARY_TEST_CASE(_optr)                            \
    checker.set_param(Mode::_optr).execs({{1, 127}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {}});

#define BUILD_UNARY_TEST_CASE_FLOAT \
    UNARY_TEST_CASE(ABS)            \
    UNARY_TEST_CASE(LOG)            \
    UNARY_TEST_CASE(COS)            \
    UNARY_TEST_CASE(SIN)            \
    UNARY_TEST_CASE(FLOOR)          \
    UNARY_TEST_CASE(CEIL)           \
    UNARY_TEST_CASE(SIGMOID)        \
    UNARY_TEST_CASE(EXP)            \
    UNARY_TEST_CASE(TANH)           \
    UNARY_TEST_CASE(FAST_TANH)      \
    UNARY_TEST_CASE(RELU)           \
    UNARY_TEST_CASE(ROUND)

    checker.set_dtype(0, dtype::BFloat16());
    checker.set_dtype(1, dtype::BFloat16());
    UniformFloatRNG rng0(1e-2, 6e1);
    checker.set_rng(0, &rng0);
    checker.set_epsilon(1e-2);
    BUILD_UNARY_TEST_CASE_FLOAT

#undef UNARY_TEST_CASE
#undef BUILD_UNARY_TEST_CASE_FLOAT

    // binary
#define BINARY_COMPLATE_TEST_CASE(_optr)                                    \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {3, 4, 7}, {}});       \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 4, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{3, 4, 7}, {1, 4, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{1, 4, 1}, {3, 4, 7}, {}});       \
    checker.set_param(Mode::_optr).execs({{3, 4, 5, 7}, {1, 1, 1, 1}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 1, 1, 1}, {3, 4, 5, 7}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {}});             \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 2, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{1, 2, 1}, {1, 2, 2}, {}});       \
    checker.set_param(Mode::_optr).execs({{1, 2, 2}, {1, 1, 1}, {}});       \
    checker.set_param(Mode::_optr).execs({{1, 1, 1}, {1, 2, 2}, {}});       \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {}});

#define BUILD_BINARY_COMPLATE_TEST_CASE \
    BINARY_COMPLATE_TEST_CASE(ADD)      \
    BINARY_COMPLATE_TEST_CASE(MUL)      \
    BINARY_COMPLATE_TEST_CASE(MAX)      \
    BINARY_COMPLATE_TEST_CASE(MIN)      \
    BINARY_COMPLATE_TEST_CASE(SUB)

    UniformFloatRNG rng1(1e-5, 7e1);
    checker.set_rng(0, &rng1);
    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::BFloat16());
    checker.set_dtype(1, dtype::BFloat16());
    BUILD_BINARY_COMPLATE_TEST_CASE

#undef BINARY_COMPLATE_TEST_CASE
#undef BUILD_BINARY_COMPLATE_TEST_CASE

    // ternary
#define TERNARY_COMPLATE_TEST_CASE(_optr)                               \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 7}, {3, 4, 7}, {3, 4, 7}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 4, 1, 1}, {3, 4, 5, 7}, {1, 4, 1, 1}, {}});     \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 4, 1}, {3, 4, 7}, {1, 4, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{3, 4, 5, 7}, {3, 4, 5, 7}, {1, 1, 1, 1}, {}});     \
    checker.set_param(Mode::_optr).execs({{1, 7}, {1, 7}, {1, 7}, {}}); \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 2, 1}, {1, 2, 2}, {1, 2, 1}, {}});              \
    checker.set_param(Mode::_optr)                                      \
            .execs({{1, 2, 2}, {1, 2, 2}, {1, 1, 1}, {}});              \
    checker.set_param(Mode::_optr).execs({{3, 4, 1}, {3, 4, 1}, {3, 4, 1}, {}});

#define BUILD_TERNARY_COMPLATE_TEST_CASE \
    TERNARY_COMPLATE_TEST_CASE(FUSE_MUL_ADD3)

    UniformFloatRNG rng2(1e-5, 7e1);
    checker.set_rng(0, &rng2);
    checker.set_epsilon(1e-2);
    checker.set_dtype(0, dtype::BFloat16());
    checker.set_dtype(1, dtype::BFloat16());
    checker.set_dtype(2, dtype::BFloat16());
    BUILD_TERNARY_COMPLATE_TEST_CASE

#undef TERNARY_COMPLATE_TEST_CASE
#undef BUILD_TERNARY_COMPLATE_TEST_CASE
}

TEST_F(CUDA, ELEMWISE_ADD_BCAST_10_INT8_INPLACE) {
    constexpr size_t A = 2, B = 48, C0 = 14, C1 = 14, C = C0 * C1;
    SyncedTensor<dt_int8> t0(handle_cuda(),
                             {TensorShape{A, B, C0, C1}, dtype::Int8()}),
            t1(handle_cuda(), {TensorShape{1, B, C0, C1}, dtype::Int8()}),
            t2(handle_cuda(), {TensorShape{A, B, C0, C1}, dtype::Int8()});
    UniformIntRNG rng{-128, 127};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host();
    auto p2 = t2.ptr_mutable_host();
    for (size_t i = 0; i < A; ++i) {
        for (size_t j = 0; j < B; ++j) {
            for (size_t k = 0; k < C; ++k) {
                auto off0 = j * C + k;
                auto off1 = i * B * C + j * C + k;
                p2[off1] = p0[off1] + p1[off0];
            }
        }
    }

    auto opr = handle_cuda()->create_operator<ElemwiseForward>();
    opr->param().mode = ElemwiseForward::Mode::ADD;
    opr->exec({t0.tensornd_dev(), t1.tensornd_dev()}, t0.tensornd_dev());

    auto pt = t0.ptr_host();

    for (size_t i = 0; i < A; ++i) {
        for (size_t j = 0; j < B; ++j) {
            for (size_t k = 0; k < C; ++k) {
                auto off = i * B * C + j * C + k;
                ASSERT_EQ(pt[off], p2[off]);
            }
        }
    }
}

//! the memory of this test case is too large, sometimes will fail on tx1
TEST_F(CUDA, ELEMWISE_BENCHMARK_DENSE) {
    constexpr size_t A = 256 * 1024 * 64, S0 = 16, S1 = 256, S2 = 64, S3 = 64;
    static_assert(A == S0 * S1 * S2 * S3, "bad value");
    SyncedTensor<> t0(handle_cuda(),
                      {TensorShape{S0, S1, S2, S3}, dtype::Float32()}),
            t1(handle_cuda(), {TensorShape{S0, S1, S2, S3}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    run_tensor_add(handle_cuda(), t0.tensornd_dev(), t0.tensornd_dev(),
                   t1.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host();
    for (size_t i = 0; i < A; ++i) {
        ASSERT_EQ(p0[i] + p0[i], p1[i]) << "at index " << i << "/" << A;
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, ELEMWISE_BENCHMARK_BCAST_101) {
    constexpr size_t A = 511, B = 509, C0 = 23, C1 = 23, C = C0 * C1;
    SyncedTensor<> t0(handle_cuda(),
                      {TensorShape{A, B, C0, C1}, dtype::Float32()}),
            t1(handle_cuda(), {TensorShape{1, B, 1, 1}, dtype::Float32()}),
            t2(handle_cuda(), {TensorShape{A, B, C0, C1}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    run_tensor_add(handle_cuda(), t0.tensornd_dev(), t1.tensornd_dev(),
                   t2.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host(), p2 = t2.ptr_host();
    for (size_t i = 0; i < A; ++i) {
        for (size_t j = 0; j < B; ++j) {
            for (size_t k = 0; k < C; ++k) {
                auto off = i * B * C + j * C + k;
                ASSERT_EQ(p0[off] + p1[j], p2[off]);
            }
        }
    }
}

TEST_F(CUDA, ELEMWISE_BENCHMARK_BCAST_10) {
    constexpr size_t A = 11583, B = 11587;
    SyncedTensor<> t0(handle_cuda(), {TensorShape{A, B}, dtype::Float32()}),
            t1(handle_cuda(), {TensorShape{1, B}, dtype::Float32()}),
            t2(handle_cuda(), {TensorShape{A, B}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    run_tensor_add(handle_cuda(), t0.tensornd_dev(), t1.tensornd_dev(),
                   t2.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host(), p2 = t2.ptr_host();
    for (size_t i = 0; i < A; ++i) {
        for (size_t j = 0; j < B; ++j) {
            auto off = i * B + j;
            ASSERT_EQ(p0[off] + p1[j], p2[off]);
        }
    }
}

TEST_F(CUDA, ELEMWISE_BENCHMARK_BCAST_01) {
    constexpr size_t A = 11583, B = 11587;
    SyncedTensor<> t0(handle_cuda(), {TensorShape{1, A, B}, dtype::Float32()}),
            t1(handle_cuda(), {TensorShape{1, A, 1}, dtype::Float32()}),
            t2(handle_cuda(), {TensorShape{1, A, B}, dtype::Float32()});
    UniformFloatRNG rng{-2.f, 2.f};
    rng.gen(t0.tensornd_host());
    rng.gen(t1.tensornd_host());
    run_tensor_add(handle_cuda(), t0.tensornd_dev(), t1.tensornd_dev(),
                   t2.tensornd_dev());
    auto p0 = t0.ptr_host(), p1 = t1.ptr_host(), p2 = t2.ptr_host();
    for (size_t i = 0; i < A; ++i) {
        for (size_t j = 0; j < B; ++j) {
            auto off = i * B + j;
            ASSERT_EQ(p0[off] + p1[i], p2[off]);
        }
    }
}

TEST_F(CUDA, BENCHMARK_ELEMWISE_IBYTE) {
    Benchmarker<ElemwiseForward> bencher(handle_cuda());
    using Mode = ElemwiseForward::Param::Mode;
    auto run_bench = [&](size_t N, size_t C, size_t H, size_t W) {
        size_t nr_times = 100;
        bencher.set_times(nr_times)
                .set_param(Mode::FUSE_ADD_RELU)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8());
        auto time =
                bencher.execs({{N * C * H * W + 1}, {N * C * H * W + 1}, {}}) /
                nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (3.0 * (N * C * H * W + 1)) / (time * 1e6));
        time = bencher.execs({{N, C / 4, H, W, 4}, {N, C / 4, H, W, 4}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (3.0 * N * C * H * W) / (time * 1e6));
        time = bencher.execs({{N, C / 4, H, W, 4}, {1, C / 4, 1, 1, 4}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (C + 2.0 * N * C * H * W) / (time * 1e6));
        time = bencher.execs({{N, C / 4, H, W, 4}, {1}, {}}) / nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (2.0 * N * C * H * W + 1) / (time * 1e6));
        time = bencher.execs(
                       {{N, C / 32, H, W, 32}, {N, C / 32, H, W, 32}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (3.0 * N * C * H * W) / (time * 1e6));
        time = bencher.execs(
                       {{N, C / 32, H, W, 32}, {1, C / 32, 1, 1, 32}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (C + 2.0 * N * C * H * W) / (time * 1e6));
        bencher.set_dtype(0, dtype::Float32()).set_dtype(1, dtype::Float32());
        time = bencher.execs({{N, C / 4, H, W}, {N, C / 4, H, W}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (3.0 * N * C * H * W) / (time * 1e6));
        time = bencher.execs({{N, C / 4, H, W}, {1, C / 4, 1, 1}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (C + 2.0 * N * C * H * W) / (time * 1e6));
    };
    run_bench(256, 256, 56, 56);
}

TEST_F(CUDA, BENCHMARK_ELEMWISE_MIN_MAX) {
    Benchmarker<ElemwiseForward> bencher(handle_cuda());
    using Mode = ElemwiseForward::Param::Mode;
    UniformIntRNG const_1{1, 1}, rng{-128, 127};
    auto run_bench = [&](size_t N, size_t C, size_t H, size_t W, DType dtype) {
        size_t nr_times = 1000;
        bencher.set_times(nr_times)
                .set_param(Mode::MIN)
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype);
        auto time =
                bencher.execs({{N, C / 4, H, W, 4}, {N, C / 4, H, W, 4}, {}}) /
                nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (3.0 * N * C * H * W) / (time * 1e6));
        bencher.set_param(Mode::MAX).set_rng(0, &const_1).set_rng(1, &const_1);
        time = bencher.execs({{N, C / 4, H, W, 4}, {N, C / 4, H, W, 4}, {}}) /
               nr_times;
        printf("time = %.2fms, bandwidth = %.2fGB/s\n", time,
               (3.0 * N * C * H * W) / (time * 1e6));
    };
    run_bench(256, 256, 56, 56, dtype::Int8());
}
#endif

// vim: syntax=cpp.doxygen
