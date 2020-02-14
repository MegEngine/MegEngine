/**
 * \file dnn/src/fallback/add_update/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/naive/add_update/opr_impl.h"

#include "src/common/utils.h"
#include "src/fallback/handle.h"

namespace {

using namespace megdnn;

template <typename T>
void forward(_megdnn_tensor_inout dest, _megdnn_tensor_in delta,
             const AddUpdate::Param& param) {
    T alpha(param.alpha), beta(param.beta), bias(param.bias);

    T* iter0 = dest.ptr<T>();
    T* iter1 = delta.ptr<T>();
    for (size_t i = 0, it = dest.layout.total_nr_elems(); i < it; ++i) {
        *iter0 = alpha * *iter0 + beta * *iter1 + bias;
        ++iter0;
        ++iter1;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace fallback {

void AddUpdateImpl::exec(_megdnn_tensor_inout dest, _megdnn_tensor_in delta) {
    check_exec(dest.layout, delta.layout);
    // eq_shape is the same as eq_layout when both input tensors are contiguous.
    if (!dest.layout.is_contiguous() || !delta.layout.is_contiguous() ||
        !dest.layout.eq_shape(delta.layout)) {
        return naive::AddUpdateForwardImpl::exec(dest, delta);
    }

#define cb(DType)                                                           \
    if (dest.layout.dtype == DType()) {                                     \
        using ctype = typename DTypeTrait<DType>::ctype;                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward<ctype>(dest, delta, m_param)); \
        return;                                                             \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb
    megdnn_assert_internal(0);
}

}  // namespace fallback
}  // namespace megdnn
// vim: syntax=cpp.doxygen
