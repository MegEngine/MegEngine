/**
 * \file dnn/src/cuda/fake_quant/opr_impl.cpp
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

void FakeQuantForwardImpl::exec(_megdnn_tensor_in input,
                                _megdnn_tensor_in scale,
                                _megdnn_tensor_in zero_point,
                                _megdnn_tensor_out output,
                                _megdnn_workspace workspace) {
    check_exec(input.layout, scale.layout, zero_point.layout, output.layout,
               workspace.size);

    if (!input.layout.is_contiguous() || !output.layout.is_contiguous()) {
        return exec_noncontig(input, scale, zero_point, output);
    }
    ElemwiseOpParamN<2> ele_param;
    ele_param[0] = scale;
    ele_param[0].layout = ele_param[0].layout.broadcast(input.layout);
    ele_param[1] = zero_point;
    ele_param[1].layout = ele_param[1].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                         \
    if (input.layout.dtype == DType()) {                                  \
        using T = typename DTypeTrait<DType>::ctype;                      \
        run_elemwise<FakeQuantKernOp<T>, T, 2>(ele_param, stream,         \
                                               {input, output, m_param}); \
        return;                                                           \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void FakeQuantForwardImpl::exec_noncontig(_megdnn_tensor_in input,
                                          _megdnn_tensor_in scale,
                                          _megdnn_tensor_in zero_point,
                                          _megdnn_tensor_out output) {
    ElemwiseOpParamN<4> ele_param;
    ele_param[0] = output;
    ele_param[1] = input;
    ele_param[2] = scale;
    ele_param[2].layout = ele_param[2].layout.broadcast(input.layout);
    ele_param[3] = zero_point;
    ele_param[3].layout = ele_param[3].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto stream = cuda_stream(handle());

#define cb(DType)                                                          \
    if (input.layout.dtype == DType()) {                                   \
        using T = typename DTypeTrait<DType>::ctype;                       \
        run_elemwise<FakeQuantKernOpNonContig<T>, T, 4>(ele_param, stream, \
                                                        {m_param});        \
        return;                                                            \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void FakeQuantBackwardImpl::exec(_megdnn_tensor_in diff,
                                 _megdnn_tensor_in input,
                                 _megdnn_tensor_in scale,
                                 _megdnn_tensor_in zero_point,
                                 _megdnn_tensor_out grad,
                                 _megdnn_workspace workspace) {
    check_exec(diff.layout, input.layout, scale.layout, zero_point.layout,
               grad.layout, workspace.size);

    if (!input.layout.is_contiguous() || !diff.layout.is_contiguous() ||
        !grad.layout.is_contiguous()) {
        return exec_noncontig(diff, input, scale, zero_point, grad);
    }
    ElemwiseOpParamN<2> ele_param;
    ele_param[0] = scale;
    ele_param[0].layout = ele_param[0].layout.broadcast(input.layout);
    ele_param[1] = zero_point;
    ele_param[1].layout = ele_param[1].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());
#define cb(DType)                                                 \
    if (grad.layout.dtype == DType()) {                           \
        using T = typename DTypeTrait<DType>::ctype;              \
        run_elemwise<FakeQuantBwdKernOp<T>, T, 2>(                \
                ele_param, stream, {diff, input, grad, m_param}); \
        return;                                                   \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

void FakeQuantBackwardImpl::exec_noncontig(_megdnn_tensor_in diff,
                                           _megdnn_tensor_in input,
                                           _megdnn_tensor_in scale,
                                           _megdnn_tensor_in zero_point,
                                           _megdnn_tensor_out grad) {
    ElemwiseOpParamN<5> ele_param;
    ele_param[0] = grad;
    ele_param[1] = diff;
    ele_param[2] = input;
    ele_param[3] = scale;
    ele_param[3].layout = ele_param[3].layout.broadcast(input.layout);
    ele_param[4] = zero_point;
    ele_param[4].layout = ele_param[4].layout.broadcast(input.layout);
    ele_param.init_from_given_tensor();
    auto m_param = param();
    auto stream = cuda_stream(handle());
#define cb(DType)                                                             \
    if (grad.layout.dtype == DType()) {                                       \
        using T = typename DTypeTrait<DType>::ctype;                          \
        run_elemwise<FakeQuantBwdKernOpNonContig<T>, T, 5>(ele_param, stream, \
                                                           {m_param});        \
        return;                                                               \
    }
    cb(megdnn::dtype::Float32)
#undef cb
}

}  // namespace cuda
}  // namespace megdnn
