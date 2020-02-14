/**
 * \file dnn/src/cuda/batched_matrix_mul/int8x8x32.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./int8x8x32.cuh"
#include <cuda.h>
#include "./algo.h"
#include "./helper.cuh"
#include "src/common/utils.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace batched_matrix_mul;

bool BatchedMatrixMulForwardImpl::AlgoInt8x8x32::is_available(
        const SizeArgs& args) const {
    return args.can_be_treated_as_int8x8x32();
}

void BatchedMatrixMulForwardImpl::AlgoInt8x8x32::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto batch_count = args.layout_a.shape[0];
    auto m = args.tensor_c.layout.shape[1], n = args.tensor_c.layout.shape[2],
         k = args.tensor_a.layout.shape[param.transposeA ? 1 : 2];
    auto LDA = args.tensor_a.layout.stride[0],
         LDB = args.tensor_b.layout.stride[0],
         LDC = args.tensor_c.layout.stride[0];

    auto STA = args.tensor_a.layout.stride[1],
         STB = args.tensor_b.layout.stride[1],
         STC = args.tensor_c.layout.stride[1];

    int8_t* A = args.tensor_a.compatible_ptr<dt_int8>();
    int8_t* B = args.tensor_b.compatible_ptr<dt_int8>();
    int32_t* C = args.tensor_c.compatible_ptr<dt_int32>();

    auto&& handle = concrete_handle(args.opr->handle());
    exec_igemm_8x8x32(A, B, C, batch_count, m, n, k, LDA, LDB, LDC, STA, STB,
                      STC, param.transposeA, param.transposeB,
                      cuda_stream(handle));
}

size_t BatchedMatrixMulForwardImpl::AlgoInt8x8x32::get_workspace_in_bytes(
        const SizeArgs&) const {
    return 0;
}

// vim: syntax=cpp.doxygen

