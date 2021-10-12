/**
 * \file dnn/src/naive/correlation/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/correlation/opr_impl.h"
#include "src/cuda/correlation/correlation_cuda.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void CorrelationForwardImpl::exec(
        _megdnn_tensor_in data1, _megdnn_tensor_in data2, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(data1.layout, data2.layout, dst.layout, workspace.size);
    auto p = param();
    auto stream = cuda_stream(handle());
    int nthreads = dst.layout.total_nr_elems();
    int stride1 = p.stride1;
    int stride2 = p.stride2;
    int kernel_size = p.kernel_size;
    int max_displacement = p.max_displacement;
    int pad_size = p.pad_size;
    bool is_multiply = p.is_multiply;

    int tchannels = dst.layout[1];
    int theight = dst.layout[2], twidth = dst.layout[3];
    int bchannels = data1.layout[1];
    int bheight = data1.layout[2], bwidth = data1.layout[3];
    using namespace ::megdnn::cuda::correlation;

#define cb(DType)                                                                   \
    if (data1.layout.dtype == DType()) {                                            \
        using T = typename DTypeTrait<DType>::ctype;                                \
        forward_proxy<T>(                                                           \
                nthreads, data1.ptr<T>(), data2.ptr<T>(), dst.ptr<T>(), bchannels,  \
                bheight, bwidth, tchannels, theight, twidth, kernel_size,           \
                max_displacement, stride1, stride2, pad_size, is_multiply, stream); \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

void CorrelationBackwardData1Impl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out grad1, _megdnn_workspace workspace) {
    check_exec(diff.layout, data1.layout, data2.layout, grad1.layout, workspace.size);

    auto stream = cuda_stream(handle());
    int nthreads = grad1.layout.total_nr_elems();
    int stride1 = param().stride1;
    int stride2 = param().stride2;
    int kernel_size = param().kernel_size;
    int max_displacement = param().max_displacement;
    int pad_size = param().pad_size;
    bool is_multiply = param().is_multiply;

    int tchannels = diff.layout[1];
    int theight = diff.layout[2], twidth = diff.layout[3];
    int bchannels = data1.layout[1];
    int bheight = data1.layout[2], bwidth = data1.layout[3];

    using namespace ::megdnn::cuda::correlation;

#define cb(DType)                                                                  \
    if (diff.layout.dtype == DType()) {                                            \
        using T = typename DTypeTrait<DType>::ctype;                               \
        backward_proxy_data1<T>(                                                   \
                nthreads, diff.ptr<T>(), data1.ptr<T>(), data2.ptr<T>(),           \
                grad1.ptr<T>(), bchannels, bheight, bwidth, tchannels, theight,    \
                twidth, kernel_size, max_displacement, stride1, stride2, pad_size, \
                is_multiply, stream);                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

void CorrelationBackwardData2Impl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data1, _megdnn_tensor_in data2,
        _megdnn_tensor_out grad2, _megdnn_workspace workspace) {
    check_exec(diff.layout, data1.layout, data2.layout, grad2.layout, workspace.size);
    auto p = param();
    auto stream = cuda_stream(handle());
    int nthreads = grad2.layout.total_nr_elems();
    int stride1 = p.stride1;
    int stride2 = p.stride2;
    int kernel_size = p.kernel_size;
    int max_displacement = p.max_displacement;
    int pad_size = p.pad_size;
    bool is_multiply = p.is_multiply;

    int tchannels = diff.layout[1];
    int theight = diff.layout[2], twidth = diff.layout[3];
    int bchannels = data1.layout[1];
    int bheight = data1.layout[2], bwidth = data1.layout[3];

    using namespace ::megdnn::cuda::correlation;

#define cb(DType)                                                                  \
    if (diff.layout.dtype == DType()) {                                            \
        using T = typename DTypeTrait<DType>::ctype;                               \
        backward_proxy_data2<T>(                                                   \
                nthreads, diff.ptr<T>(), data1.ptr<T>(), data2.ptr<T>(),           \
                grad2.ptr<T>(), bchannels, bheight, bwidth, tchannels, theight,    \
                twidth, kernel_size, max_displacement, stride1, stride2, pad_size, \
                is_multiply, stream);                                              \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
