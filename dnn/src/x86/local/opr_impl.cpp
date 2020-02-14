/**
 * \file dnn/src/x86/local/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"
#include "./local_simd.h"

#include "src/x86/utils.h"
#include "src/common/utils.h"

using namespace megdnn;
using namespace x86;

size_t LocalImpl::get_workspace_in_bytes(const TensorLayout &,
        const TensorLayout &,
        const TensorLayout &dst)
{
    auto workspace_in_bytes = dst.total_nr_elems() * sizeof(float);
    return workspace_in_bytes;
}


LocalImpl::float_noncontig_batch_kern
LocalImpl::dispatch_float_noncontig_batch(
        const TensorLayout &src, const TensorLayout &, const TensorLayout &) {
    megdnn_assert(src.stride[0] > 0 &&
            static_cast<size_t>(src.stride[0]) >=
            src.total_nr_elems() / src.shape[0]);

    if (param().mode == Mode::CROSS_CORRELATION) {
        if (is_supported(SIMDType::FMA)) {
            return local_xcorr_FMA;
        } else if (is_supported(SIMDType::AVX)) {
            return local_xcorr_AVX;
        } else if (is_supported(SIMDType::SSE)) {
            return local_xcorr_SSE;
        } else {
            megdnn_throw(megdnn_mangle("no fma/avx/sse detected"));
        }
    } else {
        if (is_supported(SIMDType::FMA)) {
            return local_conv_FMA;
        } else if (is_supported(SIMDType::AVX)) {
            return local_conv_AVX;
        } else if (is_supported(SIMDType::SSE)) {
            return local_conv_SSE;
        } else {
            megdnn_throw(megdnn_mangle("no fma/avx/sse detected"));
        }
    }
}

void LocalImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    megdnn_assert(src.layout.dtype == dtype::Float32(),
                  "x86 do not support fp16 local operator");

    exec_use_float_noncontig_batch(src, filter, dst, workspace);
}

// vim: syntax=cpp.doxygen
