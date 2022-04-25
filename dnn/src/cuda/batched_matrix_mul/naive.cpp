/**
 * \file dnn/src/cuda/batched_matrix_mul/naive.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/batched_matrix_mul/naive.cuh"
#include <cuda.h>
#include "src/cuda/batched_matrix_mul/algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

#include "midout.h"
MIDOUT_DECL(megdnn_naive_matmul)

bool BatchedMatrixMulForwardImpl::AlgoNaive::is_available(const SizeArgs& args) const {
    auto&& layout_a = args.layout_a;
    auto&& layout_b = args.layout_b;
    auto&& layout_c = args.layout_c;
    return layout_a.dtype.enumv() == layout_b.dtype.enumv() &&
           (layout_a.dtype.enumv() == DTypeEnum::Float32 ||
            layout_a.dtype.enumv() == DTypeEnum::Float16) &&
           (layout_c.dtype.enumv() == DTypeEnum::Float32 ||
            layout_c.dtype.enumv() == DTypeEnum::Float16) &&
           args.opr->param().format == param::MatrixMul::Format::DEFAULT;
}
void BatchedMatrixMulForwardImpl::AlgoNaive::exec(const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto Batch = args.tensor_c.layout.shape[0];
    auto m = args.tensor_c.layout.shape[1], n = args.tensor_c.layout.shape[2],
         k = args.tensor_a.layout.shape[param.transposeA ? 1 : 2];
    auto LDA = args.tensor_a.layout.stride[1], LDB = args.tensor_b.layout.stride[1],
         LDC = args.tensor_c.layout.stride[1];

    auto&& handle = concrete_handle(args.opr->handle());

    using ComputeMode = Param::ComputeMode;
#define DISPATCH_CMODE(in_dt, out_dt, in_ct, out_ct, comp_ct, cmode)               \
    MIDOUT_BEGIN(                                                                  \
            megdnn_naive_matmul,                                                   \
            midout_iv(#in_dt #out_dt #in_ct, #out_ct, #comp_ct, #cmode)) {         \
        do {                                                                       \
            using namespace dtype;                                                 \
            if (args.tensor_a.layout.dtype.enumv() == DTypeTrait<in_dt>::enumv &&  \
                args.tensor_c.layout.dtype.enumv() == DTypeTrait<out_dt>::enumv && \
                param.compute_mode == cmode) {                                     \
                in_ct* A = args.tensor_a.compatible_ptr<in_ct>();                  \
                in_ct* B = args.tensor_b.compatible_ptr<in_ct>();                  \
                out_ct* C = args.tensor_c.compatible_ptr<out_ct>();                \
                exec_bgemm_naive<in_ct, in_ct, out_ct, comp_ct>(                   \
                        A, B, C, Batch, m, n, k, LDA, LDB, LDC, param.transposeA,  \
                        param.transposeB, cuda_stream(handle));                    \
                return;                                                            \
            }                                                                      \
        } while (0);                                                               \
    }                                                                              \
    MIDOUT_END();
#define DISPATCH(in_dt, out_dt, in_ct, out_ct, comp_ct) \
    DISPATCH_CMODE(in_dt, out_dt, in_ct, out_ct, comp_ct, ComputeMode::DEFAULT)

    DISPATCH(Float32, Float32, dt_float32, dt_float32, dt_float32);
    DISPATCH(Float16, Float16, dt_float16, dt_float16, dt_float16);
    DNN_INC_FLOAT16(DISPATCH_CMODE(
            Float16, Float16, dt_float16, dt_float16, dt_float32,
            ComputeMode::FLOAT32));
#undef DISPATCH_CMODE
#undef DISPATCH
    megdnn_throw(ssprintf(
            "unsupported Matmul(%s, %s) -> %s with cmode = %d",
            args.layout_a.dtype.name(), args.layout_b.dtype.name(),
            args.layout_c.dtype.name(), static_cast<int>(param.compute_mode)));
}

// vim: syntax=cpp.doxygen
