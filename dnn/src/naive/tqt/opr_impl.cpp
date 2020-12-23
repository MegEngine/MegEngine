/**
 * \file dnn/src/naive/tqt/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/naive/tqt/opr_impl.h"
#include <cmath>
#include "megdnn/tensor_iter.h"
#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace {
using namespace megdnn;

template <typename T>
void forward_impl(const ElemwiseOpParamN<3> src, float qmin, float qmax) {
    auto inp = tensor_iter_valonly<T>(src[0]).begin();
    auto out = tensor_iter_valonly<T>(src[1]).begin();
    auto scale = tensor_iter_valonly<T>(src[2]).begin();
    size_t total = src[0].layout.total_nr_elems();
    for (size_t i = 0; i < total; ++i) {
        T t = pow(2, *scale);
        T x = round(*inp / t);
        x = x <= qmin ? qmin : x;
        x = x >= qmax ? qmax : x;
        *out = x * t;
        ++inp;
        ++out;
        ++scale;
    }
}

template <typename T>
void backward_impl(const ElemwiseOpParamN<5> src, float qmin, float qmax) {
    auto diff = tensor_iter_valonly<T>(src[0]).begin();
    auto input = tensor_iter_valonly<T>(src[1]).begin();
    auto scale = tensor_iter_valonly<T>(src[2]).begin();
    auto grad_x = tensor_iter_valonly<T>(src[3]).begin();
    auto grad_s = tensor_iter_valonly<T>(src[4]).begin();
    size_t total = src[0].layout.total_nr_elems();

    for (size_t i = 0; i < total; ++i) {
        T t = pow(2, *scale);
        T scaled = *input / t;
        T rounded = round(scaled);
        rounded = rounded <= qmin ? qmin : rounded;
        rounded = rounded >= qmax ? qmax : rounded;
        bool mask_clip = scaled < -0.5 + qmin && scaled > 0.5 + qmax;
        bool mask_quant = !mask_clip;

        *grad_x = *diff * mask_quant;
        T grad_quant = *diff * mask_quant * (rounded - scaled) * t * log(2.0);
        T grad_clip = *diff * mask_clip * rounded * t * log(2.0);
        *grad_s = grad_quant + grad_clip;

        ++input;
        ++diff;
        ++scale;
        ++grad_x;
        ++grad_s;
    }
}

}  // namespace
namespace megdnn {
namespace naive {

void TQTForwardImpl::exec(_megdnn_tensor_in input, _megdnn_tensor_in scale,
                          _megdnn_tensor_out output,
                          _megdnn_workspace workspace) {
    check_exec(input.layout, scale.layout, output.layout, workspace.size);
    ElemwiseOpParamN<3> src;
    src[0] = input;
    src[1] = output;
    src[2] = scale;
    src[2].layout = src[2].layout.broadcast(input.layout);
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

void TQTBackwardImpl::exec(_megdnn_tensor_in diff, _megdnn_tensor_in input,
                           _megdnn_tensor_in scale, _megdnn_tensor_out grad_x,
                           _megdnn_tensor_out grad_s,
                           _megdnn_workspace workspace) {
    check_exec(diff.layout, input.layout, scale.layout, grad_x.layout,
               grad_s.layout, workspace.size);
    ElemwiseOpParamN<5> src;
    src[0] = diff;
    src[1] = input;
    src[2] = scale;
    src[2].layout = src[2].layout.broadcast(input.layout);
    src[3] = grad_x;
    src[4] = grad_s;
#define cb(DType)                                                         \
    if (diff.layout.dtype == DType() && grad_x.layout.dtype == DType() && \
        input.layout.dtype == DType()) {                                  \
        using T = typename DTypeTrait<DType>::ctype;                      \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                                     \
                backward_impl<T>(src, param().qmin, param().qmax));       \
        return;                                                           \
    }
    cb(dtype::Float32)
#undef cb
}

}  // namespace naive
}  // namespace megdnn
