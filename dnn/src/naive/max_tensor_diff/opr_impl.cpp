/**
 * \file dnn/src/naive/max_tensor_diff/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "src/naive/handle.h"

#include "src/common/utils.h"

#include <cstring>

using namespace megdnn;
using namespace naive;

namespace {
template <typename T>
float exec_forward(const T* src1, const T* src2, const size_t nr_elem) {
    float maxerr = 0.f;
    rep(i, nr_elem) {
        float x = src1[i];
        float y = src2[i];
        float diff = std::isfinite(x) && std::isfinite(y) ? fabs(x-y) /
            fmax(fmin(fabs(x),fabs(y)), 1) : INFINITY;
        maxerr = std::max(diff, maxerr);
    }

    return maxerr;
}
}  // anonymous namespace

float MaxTensorDiffImpl::exec(_megdnn_tensor_in src1, _megdnn_tensor_in src2,
                              _megdnn_workspace workspace) {
    check_exec(src1.layout, src2.layout, workspace.size);
    float result = 0.f;

    auto run = [&]() {

#define cb(DType)                                                          \
    if (src1.layout.dtype == DType()) {                                    \
        using ctype = typename DTypeTrait<DType>::ctype;                   \
        result = exec_forward<ctype>(src1.ptr<ctype>(), src2.ptr<ctype>(), \
                                     src1.layout.total_nr_elems());        \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
#undef cb

    };

    auto handle = static_cast<HandleImpl*>(this->handle());
    handle->dispatch_kern(run);
    handle->megcore_dispatcher()->sync();

    return result;
}

// vim: syntax=cpp.doxygen
