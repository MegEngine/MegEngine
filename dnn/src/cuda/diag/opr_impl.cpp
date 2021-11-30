/**
 * \file dnn/src/cuda/diag/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/diag/opr_impl.h"

#include "src/cuda/diag/diag.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void DiagImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    if (src.layout.ndim == 2) {
        auto src_stride0 = src.layout.stride[0];
        auto src_stride1 = src.layout.stride[1];
        auto dst_stride = dst.layout.stride[0];
        auto start =
                (param().k >= 0) ? param().k * src_stride1 : -param().k * src_stride0;

#define cb(DType)                                                               \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                 \
        using ctype = typename DTypeTrait<DType>::ctype;                        \
        diag::exec_internal_to_vector<ctype>(                                   \
                src.ptr<ctype>(), dst.ptr<ctype>(), start, dst.layout.shape[0], \
                src_stride0 + src_stride1, dst_stride, cuda_stream(handle()));  \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
    } else {
        auto n = dst.layout.shape[0];
        auto src_stride = src.layout.stride[0];
        auto dst_stride0 = dst.layout.stride[0];
        auto dst_stride1 = dst.layout.stride[1];
        auto offset = (param().k >= 0) ? 0 : -param().k;

#define cb(DType)                                                                      \
    if (dst.layout.dtype.enumv() == DTypeTrait<DType>::enumv) {                        \
        using ctype = typename DTypeTrait<DType>::ctype;                               \
        diag::exec_internal_to_matrix<ctype>(                                          \
                src.ptr<ctype>(), dst.ptr<ctype>(), offset, n, param().k, dst_stride0, \
                dst_stride1, src_stride, cuda_stream(handle()));                       \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
        cb(::megdnn::dtype::Bool)
#undef cb
    }
}

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
