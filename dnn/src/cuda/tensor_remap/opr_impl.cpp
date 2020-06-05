/**
 * \file dnn/src/cuda/tensor_remap/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/tensor_remap/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/tensor_remap/tensor_remap.cuh"

namespace megdnn {
namespace cuda {

void IndexingRemapForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in map,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, map.layout, dst.layout, workspace.size);
    constexpr auto NDIM = TensorShape::MAX_NDIM;
    array_wrapper<int, NDIM> sstride;
    array_wrapper<int, NDIM> dstride;
    array_wrapper<uint32_t, NDIM> dshape;
    // Initialize array_wrappers.
    for (size_t i = 0_z; i < src.layout.ndim; ++i) {
        sstride.data[i] = src.layout.stride[i];
    }
    for (size_t i = 0_z; i < dst.layout.ndim; ++i) {
        dstride.data[i] = dst.layout.stride[i];
    }
    for (size_t i = 0_z; i < dst.layout.ndim; ++i) {
        dshape.data[i] = dst.layout.shape[i];
    }
        // Invoke kernel
#define cb(dt)                                                              \
    if (src.layout.dtype.enumv() == DTypeTrait<dt>::enumv) {                \
        using ctype = DTypeTrait<dt>::ctype;                                \
        tensor_remap::forward<ctype>(src.ptr<ctype>(), map.ptr<dt_int32>(), \
                                     dst.ptr<ctype>(), src.layout.ndim,     \
                                     dst.layout.ndim, sstride, dstride,     \
                                     dshape, cuda_stream(handle()));        \
        return;                                                             \
    }
    cb(dtype::Float32)
    cb(dtype::Int32)
#undef cb
    megdnn_throw(
            ssprintf("cuda indexing remap forward only support "
                     "float32/int32 dtype, got %s",
                     src.layout.dtype.name()));
}

void IndexingRemapBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_in map,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, map.layout, grad.layout, workspace.size);
    constexpr auto NDIM = TensorShape::MAX_NDIM;
    array_wrapper<int, NDIM> sstride;
    array_wrapper<int, NDIM> dstride;
    array_wrapper<uint32_t, NDIM> sshape;
    array_wrapper<uint32_t, NDIM> dshape;
    // Intialize array_wrappers.
    for (size_t i = 0_z; i < grad.layout.ndim; ++i) {
        sstride.data[i] = grad.layout.stride[i];
    }
    for (size_t i = 0_z; i < diff.layout.ndim; ++i) {
        dstride.data[i] = diff.layout.stride[i];
    }
    for (size_t i = 0_z; i < grad.layout.ndim; ++i) {
        sshape.data[i] = grad.layout.shape[i];
    }
    for (size_t i = 0_z; i < diff.layout.ndim; ++i) {
        dshape.data[i] = diff.layout.shape[i];
    }

        // Invoke kernel
#define cb(dt)                                                                \
    if (diff.layout.dtype.enumv() == DTypeTrait<dt>::enumv) {                 \
        using ctype = DTypeTrait<dt>::ctype;                                  \
        tensor_remap::backward<ctype>(                                        \
                diff.ptr<ctype>(), map.ptr<dt_int32>(), grad.ptr<ctype>(),    \
                grad.layout.ndim, diff.layout.ndim, sstride, dstride, sshape, \
                dshape, param().is_non_overlapping, cuda_stream(handle()));   \
        return;                                                               \
    }
    cb(dtype::Float32)
    cb(dtype::Int32)

    megdnn_throw(
            ssprintf("cuda indexing remap forward only support "
                     "float32/int32 dtype, got %s",
                     diff.layout.dtype.name()));
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
