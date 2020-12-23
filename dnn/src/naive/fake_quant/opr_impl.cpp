/**
 * \file dnn/src/naive/fakequant/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/naive/fake_quant/opr_impl.h"
#include <cmath>
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"
namespace {
using namespace megdnn;

template <typename T>
void forward_impl(const ElemwiseOpParamN<4> src, float qmin, float qmax) {
    auto inp = tensor_iter_valonly<T>(src[0]).begin();
    auto out = tensor_iter_valonly<T>(src[1]).begin();
    auto scale = tensor_iter_valonly<T>(src[2]).begin();
    auto zero_point = tensor_iter_valonly<T>(src[3]).begin();
    size_t total = src[0].layout.total_nr_elems();
    for (size_t i = 0; i < total; ++i) {
        T x = round(*inp / (*scale)) + *zero_point;
        x = x <= qmin ? qmin : x;
        x = x >= qmax ? qmax : x;
        *out = (x - *zero_point) * *scale;
        ++inp;
        ++out;
        ++scale;
        ++zero_point;
    }
}

template <typename T>
void backward_impl(const ElemwiseOpParamN<5> src, float qmin, float qmax) {
    auto diff = tensor_iter_valonly<T>(src[0]).begin();
    auto input = tensor_iter_valonly<T>(src[1]).begin();
    auto scale = tensor_iter_valonly<T>(src[2]).begin();
    auto zero_point = tensor_iter_valonly<T>(src[3]).begin();
    auto grad = tensor_iter_valonly<T>(src[4]).begin();
    size_t total = src[0].layout.total_nr_elems();
    for (size_t i = 0; i < total; ++i) {
        T x = round(*input / (*scale)) + *zero_point;
        *grad = (x >= qmin && x <= qmax) ? *diff : 0.0;
        ++diff;
        ++input;
        ++scale;
        ++zero_point;
        ++grad;
    }
}

}  // namespace
namespace megdnn {
namespace naive {

void FakeQuantForwardImpl::exec(_megdnn_tensor_in input,
                                _megdnn_tensor_in scale,
                                _megdnn_tensor_in zero_point,
                                _megdnn_tensor_out output,
                                _megdnn_workspace workspace) {
    check_exec(input.layout, scale.layout, zero_point.layout, output.layout,
               workspace.size);
    ElemwiseOpParamN<4> src;
    src[0] = input;
    src[1] = output;
    src[2] = scale;
    src[2].layout = src[2].layout.broadcast(input.layout);
    src[3] = zero_point;
    src[3].layout = src[3].layout.broadcast(input.layout);
#define cb(DType)                                                  \
    if (input.layout.dtype == DType()) {                           \
        using T = typename DTypeTrait<DType>::ctype;               \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                              \
                forward_impl<T>(src, param().qmin, param().qmax)); \
        return;                                                    \
    }
    cb(dtype::Float32)
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
    ElemwiseOpParamN<5> src;
    src[0] = diff;
    src[1] = input;
    src[2] = scale;
    src[2].layout = src[2].layout.broadcast(input.layout);
    src[3] = zero_point;
    src[3].layout = src[3].layout.broadcast(input.layout);
    src[4] = grad;
#define cb(DType)                                                       \
    if (diff.layout.dtype == DType() && grad.layout.dtype == DType() && \
        input.layout.dtype == DType()) {                                \
        using T = typename DTypeTrait<DType>::ctype;                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                   \
                backward_impl<T>(src, param().qmin, param().qmax));     \
        return;                                                         \
    }
    cb(dtype::Float32)
#undef cb
}
}  // namespace naive
}  // namespace megdnn
