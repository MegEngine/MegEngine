/**
 * \file dnn/src/naive/check_non_finite/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/check_non_finite/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace {
using namespace megdnn;

#define src_ctype dt_float32
#define wtype dt_int32

void reduce_fwd(const src_ctype* sptr, wtype* dptr, size_t size) {
    std::function<wtype(size_t, size_t)> func;
    func = [&](size_t l, size_t r) -> wtype {
        if (l + 1 < r) {
            size_t mid = l + (r - l) / 2;
            return func(l, mid) | func(mid, r);
        } else {
            return static_cast<wtype>(!std::isfinite(sptr[l]));
        }
    };

    dptr[0] = func(0, size);
}

}  // namespace

namespace megdnn {
namespace naive {

size_t CheckNonFiniteImpl::get_workspace_in_bytes(const TensorLayout&,
                                               const TensorLayout&) {
    return 0;
}

void CheckNonFiniteImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                           _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);

    auto handle = static_cast<HandleImpl*>(this->handle());
    MEGDNN_DISPATCH_CPU_KERN(
            handle, reduce_fwd(src.ptr<dt_float32>(), dst.ptr<dt_int32>(),
                               src.layout.total_nr_elems()));
}
}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
