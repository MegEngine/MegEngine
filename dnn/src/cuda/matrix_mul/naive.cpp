/**
 * \file dnn/src/cuda/matrix_mul/naive.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/matrix_mul/naive.cuh"
#include <cuda.h>
#include "src/cuda/matrix_mul/algos.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

bool MatrixMulForwardImpl::AlgoNaive::is_available(const SizeArgs& args) const {
    return args.can_be_treated_as_int8x8x32();
}
void MatrixMulForwardImpl::AlgoNaive::exec(const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto m = args.tensor_c.layout.shape[0], n = args.tensor_c.layout.shape[1],
         k = args.tensor_a.layout.shape[param.transposeA ? 0 : 1];
    auto LDA = args.tensor_a.layout.stride[0],
         LDB = args.tensor_b.layout.stride[0],
         LDC = args.tensor_c.layout.stride[0];

    int8_t* A = args.tensor_a.compatible_ptr<dt_int8>();
    int8_t* B = args.tensor_b.compatible_ptr<dt_int8>();
    int32_t* C = args.tensor_c.compatible_ptr<dt_int32>();

    auto&& handle = concrete_handle(args.opr->handle());
    exec_gemm_int8_naive(A, B, C, m, n, k, LDA, LDB, LDC, param.transposeA,
                         param.transposeB, cuda_stream(handle));
}

// vim: syntax=cpp.doxygen
