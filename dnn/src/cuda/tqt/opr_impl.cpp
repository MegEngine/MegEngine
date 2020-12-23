/**
 * \file dnn/src/cuda/tqt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./opr_impl.h"
#include "./kern.cuh"
#include "src/common/utils.h"
namespace megdnn {
namespace cuda {

void TQTForwardImpl::exec(_megdnn_tensor_in input, _megdnn_tensor_in scale,
                          _megdnn_tensor_out output,
                          _megdnn_workspace workspace) {
    check_exec(input.layout, scale.layout, output.layout, workspace.size);

    if (!input.layout.is_contiguous() || !output.layout.is_contiguous())
        return exec_noncontig(input, scale, output);

    ElemwiseOpParamN<1> ele_param;
    ele_param[0] = scale;
    ele_param[0].layout = ele_param[0].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                   \
    if (input.layout.dtype == DType()) {                            \
        using T = typename DTypeTrait<DType>::ctype;                \
        run_elemwise<TQTKernOp<T>, T, 1>(ele_param, stream,         \
                                         {input, output, m_param}); \
        return;                                                     \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void TQTForwardImpl::exec_noncontig(_megdnn_tensor_in input,
                                    _megdnn_tensor_in scale,
                                    _megdnn_tensor_out output) {
    ElemwiseOpParamN<3> ele_param;
    ele_param[0] = input;
    ele_param[1] = scale;
    ele_param[1].layout = ele_param[1].layout.broadcast(input.layout);
    ele_param[2] = output;
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                    \
    if (input.layout.dtype == DType()) {                             \
        using T = typename DTypeTrait<DType>::ctype;                 \
        run_elemwise<TQTKernOpNonContig<T>, T, 3>(ele_param, stream, \
                                                  {m_param});        \
        return;                                                      \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void TQTBackwardImpl::exec(_megdnn_tensor_in diff, _megdnn_tensor_in input,
                           _megdnn_tensor_in scale, _megdnn_tensor_out grad_x,
                           _megdnn_tensor_out grad_s,
                           _megdnn_workspace workspace) {
    check_exec(diff.layout, input.layout, scale.layout, grad_x.layout,
               grad_s.layout, workspace.size);

    if (!input.layout.is_contiguous() || !diff.layout.is_contiguous() ||
        !grad_x.layout.is_contiguous() || !grad_s.layout.is_contiguous())
        return exec_noncontig(diff, input, scale, grad_x, grad_s);

    ElemwiseOpParamN<1> ele_param;
    ele_param[0] = scale;
    ele_param[0].layout = ele_param[0].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                           \
    if (grad_x.layout.dtype == DType()) {                                   \
        using T = typename DTypeTrait<DType>::ctype;                        \
        run_elemwise<TQTBwdKernOp<T>, T, 1>(                                \
                ele_param, stream, {diff, input, grad_x, grad_s, m_param}); \
        return;                                                             \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void TQTBackwardImpl::exec_noncontig(_megdnn_tensor_in diff,
                                     _megdnn_tensor_in input,
                                     _megdnn_tensor_in scale,
                                     _megdnn_tensor_out grad_x,
                                     _megdnn_tensor_out grad_s) {
    ElemwiseOpParamN<5> ele_param;
    ele_param[0] = diff;
    ele_param[1] = input;
    ele_param[2] = scale;
    ele_param[2].layout = ele_param[2].layout.broadcast(input.layout);
    ele_param[3] = grad_x;
    ele_param[4] = grad_s;
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                       \
    if (input.layout.dtype == DType()) {                                \
        using T = typename DTypeTrait<DType>::ctype;                    \
        run_elemwise<TQTBwdKernOpNonContig<T>, T, 5>(ele_param, stream, \
                                                     {m_param});        \
        return;                                                         \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn