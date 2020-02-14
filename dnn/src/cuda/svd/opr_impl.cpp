/**
 * \file dnn/src/cuda/svd/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./opr_impl.h"

#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include "src/cuda/utils.h"

#include <numeric>

namespace {

using namespace megdnn;
using namespace cuda;

TensorShape transposed_shape(const TensorShape& shape) {
    SmallVector<size_t> tshape(shape.ndim);
    for (size_t i = 0; i < shape.ndim; i++) {
        tshape[i] = shape[i];
    }
    std::iter_swap(tshape.rbegin(), tshape.rbegin() + 1);
    return tshape;
}

TensorLayout transposed_layout(const TensorLayout& layout) {
    megdnn_assert(layout.ndim >= 2);
    std::vector<size_t> permutation(layout.ndim);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::iter_swap(permutation.rbegin(), permutation.rbegin() + 1);
    return layout.dimshuffle(permutation);
}

void transpose(megdnn::cuda::HandleImpl* handle, const TensorND& src,
               const TensorND& dst) {
    TensorLayout t = transposed_layout(src.layout);
    megdnn_assert(t.total_nr_elems() == dst.layout.total_nr_elems());
    handle->relayout_opr()->exec({src.raw_ptr, t}, dst);
}

}  // namespace

namespace megdnn {
namespace cuda {

WorkspaceBundle SVDForwardImpl::get_workspace_bundle(size_t block_cnt, size_t m,
                                                     size_t n,
                                                     size_t dtype_size,
                                                     void* raw_ptr) {
    const size_t max_mn = std::max(m, n);
    const size_t min_mn = std::min(m, n);
    SmallVector<size_t> ws_sizes = {
            block_cnt * m * n * dtype_size,  // copy of src
            get_cusolver_buffer_size(max_mn, min_mn) * dtype_size,
            sizeof(int)  // devInfo
    };
    if (m > n) {
        ws_sizes.push_back(block_cnt * max_mn * max_mn * dtype_size);
        ws_sizes.push_back(block_cnt * min_mn * min_mn * dtype_size);
    }
    return {raw_ptr, std::move(ws_sizes), handle()->alignment_requirement()};
}

size_t SVDForwardImpl::get_cusolver_buffer_size(size_t m, size_t n) {
    int lwork;
    auto handle = concrete_handle(this->handle());
    cusolver_check(cusolverDnSgesvd_bufferSize(handle->cusolver_handle(), m, n,
                                               &lwork));
    return lwork;
}

size_t SVDForwardImpl::get_workspace_in_bytes(size_t block_cnt, size_t m,
                                              size_t n, size_t dtype_size) {
    megdnn_assert(dtype_size == 4);

    return get_workspace_bundle(block_cnt, m, n, dtype_size)
            .total_size_in_bytes();
}

void SVDForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out u,
                          _megdnn_tensor_out s, _megdnn_tensor_out vt,
                          _megdnn_workspace workspace) {
    Param p = param();
    check_exec(src.layout, u.layout, s.layout, vt.layout, workspace.size);

    size_t block_cnt, m, n;
    canonize_params(src.layout, &block_cnt, &m, &n);

    auto wbundle = get_workspace_bundle(
            block_cnt, m, n, src.layout.dtype.size(), workspace.raw_ptr);
    auto handle = concrete_handle(this->handle());

    bool need_transpose = m > n;
    size_t min_mn = std::min(m, n);
    size_t max_mn = std::max(m, n);
    TensorND cur_u, cur_v;
    signed char job = 'N';  // Do not compute singular vectors.
    if (p.compute_uv) {
        SmallVector<size_t> u_shape, vt_shape;
        if (p.full_matrices) {
            job = 'A';  // Compute all singular vectors.
            u_shape = {block_cnt, m, m};
            vt_shape = {block_cnt, n, n};
        } else {
            job = 'S';  // Compute first min(m, n) singular vectors.
            u_shape = {block_cnt, m, min_mn};
            vt_shape = {block_cnt, min_mn, n};
        }
        if (need_transpose) {
            cur_u = {wbundle.get_workspace(3).raw_ptr,
                     {transposed_shape(u_shape), dtype::Float32()}};
            cur_v = {wbundle.get_workspace(4).raw_ptr,
                     {transposed_shape(vt_shape), dtype::Float32()}};
        } else {
            cur_v = {u.raw_ptr, u.layout.reshape(u_shape)};
            cur_u = {vt.raw_ptr, vt.layout.reshape(vt_shape)};
        }
    } else {
        cur_u = cur_v = {nullptr, {{0, 0}, dtype::Float32()}};
    }

    TensorND inp_copy(wbundle.get_workspace(0).raw_ptr,
                      {{block_cnt, min_mn, max_mn}, dtype::Float32()});
    float* cusolver_ws = wbundle.get_workspace(1).ptr<float>();
    size_t cusolver_ws_size = wbundle.get_workspace(1).size / sizeof(float);
    int* info = wbundle.get_workspace(2).ptr<int>();
    TensorND s_blk(s.raw_ptr, s.layout.reshape({block_cnt, min_mn}));

    if (need_transpose) {
        ::transpose(handle, src, inp_copy);
    } else {
        handle->relayout_opr()->exec(src, inp_copy);
    }

    for (size_t blk = 0; blk < block_cnt; blk++) {
#define SUB(x) ((x).ptr<float>() + (blk) * (x).layout.stride[0])
#define SUB_LD(x) SUB(x), (x).layout.stride[1]
        cusolver_check(cusolverDnSgesvd(
                handle->cusolver_handle(), job, job, max_mn, min_mn,
                SUB_LD(inp_copy), SUB(s_blk), SUB_LD(cur_u), SUB_LD(cur_v),
                cusolver_ws, cusolver_ws_size, nullptr, info));
#undef SUB
#undef SUB_LD
    }

    if (p.compute_uv && need_transpose) {
        ::transpose(handle, cur_u, u);
        ::transpose(handle, cur_v, vt);
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
