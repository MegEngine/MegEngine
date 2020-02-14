/**
 * \file dnn/src/naive/add_update/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"

#include "megdnn/tensor_iter.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace {

using namespace megdnn;

template <typename T>
void forward(const ElemwiseOpParamN<2> src, const AddUpdate::Param& param) {
    T alpha(param.alpha), beta(param.beta), bias(param.bias);

    auto iter0 = tensor_iter_valonly<T>(src[0]).begin();
    auto iter1 = tensor_iter_valonly<T>(src[1]).begin();
    for (size_t i = 0, it = src[0].layout.total_nr_elems(); i < it; ++i) {
        *iter0 = alpha * *iter0 + beta * *iter1 + bias;
        ++iter0;
        ++iter1;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace naive {

void AddUpdateForwardImpl::exec(_megdnn_tensor_inout dest,
                                _megdnn_tensor_in delta) {
    check_exec(dest.layout, delta.layout);
    ElemwiseOpParamN<2> src = make_param(dest, delta);
#define cb(DType)                                                   \
    if (dest.layout.dtype == DType()) {                             \
        using ctype = typename DTypeTrait<DType>::ctype;            \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward<ctype>(src, m_param)); \
        return;                                                     \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
