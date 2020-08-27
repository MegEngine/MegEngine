/**
 * \file dnn/src/rocm/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "src/rocm/pooling/opr_impl.h"

#include "src/rocm/utils.h"

namespace megdnn {
namespace rocm {

void PoolingForwardImpl::setup_descs(const TensorLayout &src,
        const TensorLayout &dst)
{
    src_desc.set(src, param().format);
    dst_desc.set(dst, param().format);
    pooling_desc.set(this->param());
}

void PoolingForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    auto handle = miopen_handle(this->handle());
    setup_descs(src.layout, dst.layout);
    dt_float32 alpha = 1.0f, beta = 0.0f;
    miopen_check(miopenPoolingForward(handle, pooling_desc.desc, &alpha,
                                      src_desc.desc, src.raw_ptr, &beta,
                                      dst_desc.desc, dst.raw_ptr, false,
                                      nullptr, 0_z));
}

void PoolingBackwardImpl::setup_descs(const TensorLayout& src,
                                      const TensorLayout& dst,
                                      const TensorLayout& diff,
                                      const TensorLayout& grad) {
    src_desc.set(src);
    dst_desc.set(dst);
    diff_desc.set(diff);
    grad_desc.set(grad);
    pooling_desc.set(this->param());
}

void PoolingBackwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in dst,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, diff.layout, grad.layout, workspace.size);
    auto handle = miopen_handle(this->handle());
    setup_descs(src.layout, dst.layout, diff.layout, grad.layout);
    float alpha = 1.0f, beta = 0.0f;
    if (param().mode == param::Pooling::Mode::MAX) {
        //! FIXME: when using max pooling opr, the backward opr need the indices
        //! of the forward opr which stored in workspace. We have to recompute
        //! the indices by calling miopenPoolingForward again.
        miopen_check(miopenPoolingForward(handle, pooling_desc.desc, &alpha,
                                          src_desc.desc, src.raw_ptr, &beta,
                                          dst_desc.desc, dst.raw_ptr, true,
                                          workspace.raw_ptr, workspace.size));
    }
    miopen_check(miopenPoolingBackward(
            handle, pooling_desc.desc, &alpha, dst_desc.desc, dst.raw_ptr,
            diff_desc.desc, diff.raw_ptr, src_desc.desc, src.raw_ptr, &beta,
            grad_desc.desc, grad.raw_ptr, workspace.raw_ptr));
}

size_t PoolingBackwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                const TensorLayout& dst,
                const TensorLayout& diff,
                const TensorLayout& grad) {
    setup_descs(src, dst, diff, grad);
    size_t ws_size = 0_z;
    miopenPoolingGetWorkSpaceSize(dst_desc.desc, &ws_size); 
    return ws_size;
};

} // namespace rocm
} // namespace megdnn

// vim: syntax=cpp.doxygen
