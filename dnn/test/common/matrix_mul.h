/**
 * \file dnn/test/common/matrix_mul.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>
#include <vector>

#include "megdnn/dtype.h"
#include "megdnn/handle.h"
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

namespace megdnn {
namespace test {
namespace matrix_mul {

// mask & 1 denotes transposeA; mask & 2 denotes transposeB
struct TestArg {
    size_t m, n, k, mask;
    size_t A_stride, B_stride, C_stride, b;
    size_t A_batch_stride, B_batch_stride, C_batch_stride;
    // stride = 0 means the default stride, the dim is contiguous, i.e. the
    // stride value which makes tensor compact.
    TestArg(size_t m, size_t n, size_t k, size_t mask, size_t A_stride = 0,
            size_t B_stride = 0, size_t C_stride = 0, size_t b = 1,
            size_t A_batch_stride = 0, size_t B_batch_stride = 0,
            size_t C_batch_stride = 0)
            : m{m},
              n{n},
              k{k},
              mask{mask},
              A_stride{A_stride},
              B_stride{B_stride},
              C_stride{C_stride},
              b{b},
              A_batch_stride{A_batch_stride},
              B_batch_stride{B_batch_stride},
              C_batch_stride{C_batch_stride} {}
};

std::vector<TestArg> get_matmul_args_no_mask();
std::vector<TestArg> get_matmul_args_mask(uint8_t mask);
std::vector<TestArg> get_matmul_args();
std::vector<TestArg> get_batched_matmul_args_mask(uint8_t mask);
std::vector<TestArg> get_batched_matmul_args();
std::vector<TestArg> get_matmul_mk_packed_args(size_t nbase);
std::vector<TestArg> get_batched_matmul_args_cublaslt();
std::vector<TestArg> get_batched_matmul_args_int8x8x32();

using TestArgFilterFunc = std::function<bool(const TestArg&)>;
template <typename Opr = megdnn::MatrixMul>
void check_matrix_mul(
        DType A_dtype, DType B_dtype, DType C_dtype, Handle* handle,
        const char* algo = nullptr,
        param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT,
        size_t nbase = 8, float eps = 1e-3, std::vector<TestArg>&& args = {});

void check_matrix_mul(
        DType A_dtype, DType B_dtype, DType C_dtype, Handle* handle,
        const char* algo = nullptr,
        param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT,
        size_t nbase = 8, float eps = 1e-3);

void check_batched_matrix_mul(DType A_dtype, DType B_dtype, DType C_dtype,
                              Handle* handle, const char* algo = nullptr,
                              float eps = 1e-3,
                              std::vector<TestArg>&& args = {});

#if MEGDNN_WITH_BENCHMARK
std::vector<TestArg> get_benchmark_matmul_args();
std::vector<TestArg> get_benchmark_matmul_mk_packed_args(size_t nbase);
//! benchmark performance with float matmul
void benchmark_with_contrast(
        Handle* handle, const std::vector<TestArg>& args, DType A_dtype,
        DType B_dtype, DType C_dtype, const char* algo = nullptr,
        param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT,
        DType contrast_A_dtype = dtype::Float32{},
        DType contrast_B_dtype = dtype::Float32{},
        DType contrast_C_dtype = dtype::Float32{},
        const char* contrast_algo = nullptr,
        param::MatrixMul::Format contrast_format =
                param::MatrixMul::Format::DEFAULT);
#endif

}  // namespace matrix_mul
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
