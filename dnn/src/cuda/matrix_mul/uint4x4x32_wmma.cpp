/**
 * \file dnn/src/cuda/matrix_mul/uint4x4x32_wmma.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algos.h"

#include "src/cuda/utils.h"
#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/uint4x4x32_wmma/wmma_matrix_mul.h"

using namespace megdnn;
using namespace cuda;
using namespace matrix_mul;

#if CUDA_VERSION >= 10000
bool MatrixMulForwardImpl::AlgoUInt4x4x32WMMA::is_available(
        const SizeArgs& args) const {
    if (args.opr->param().format != param::MatrixMul::Format::DEFAULT)
        return false;
    auto&& device_prop = current_device_prop();
    if (device_prop.major < 7 ||
        (device_prop.major == 7 && device_prop.minor < 5)) {
        return false;
    }
    auto&& param = args.opr->param();
    if (!param.transposeA && param.transposeB) {
        bool available =
                args.layout_a.dtype.enumv() == DTypeEnum::Quantized4Asymm &&
                args.layout_c.dtype.enumv() == DTypeEnum::QuantizedS32;
        size_t m = args.layout_c.shape[0], n = args.layout_c.shape[1];
        available &= (m % 8 == 0) && (n % 8 == 0);
        available &= (args.layout_a.stride[0] % 2 == 0) &&
                     (args.layout_b.stride[0] % 2 == 0);
        return available;
    }
    return false;
}

size_t MatrixMulForwardImpl::AlgoUInt4x4x32WMMA::get_workspace_in_bytes(
        const SizeArgs& args) const {
    size_t m = args.layout_c.shape[0], n = args.layout_c.shape[1];
    return (m + n) * sizeof(int32_t);
}

void MatrixMulForwardImpl::AlgoUInt4x4x32WMMA::exec(const ExecArgs& args) const {
    auto&& handle = concrete_handle(args.opr->handle());
    auto&& param = args.opr->param();
    if (!param.transposeA && param.transposeB) {
        exec_wmma_matrix_mul_quint4_nt(args.tensor_a, args.tensor_b,
                                       args.tensor_c, args.workspace,
                                       handle->stream());
    }
}
#endif

// vim: syntax=cpp.doxygen
