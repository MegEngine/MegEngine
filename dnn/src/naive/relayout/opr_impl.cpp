/**
 * \file dnn/src/naive/relayout/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/relayout/opr_impl.h"
#include "src/common/utils.h"
#include "megdnn/tensor_iter.h"
#include "src/naive/handle.h"

#include "midout.h"

MIDOUT_DECL(naive_relayout)

using namespace megdnn;
using namespace naive;

namespace {

    template<typename ctype>
    void do_copy(const TensorND &dst, const TensorND &src) {
        auto idst = tensor_iter_valonly<ctype>(dst).begin(),
             isrc = tensor_iter_valonly<ctype>(src).begin();
        for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++ i) {
            *idst = *isrc;
            ++ idst;
            ++ isrc;
        }
    }

    bool is_cpu_handle(Handle *handle) {
        megcorePlatform_t plat;
        megcoreDeviceHandle_t dh;
        megcoreGetDeviceHandle(handle->megcore_computing_handle(), &dh);
        megcoreGetPlatform(dh, &plat);
        return plat == megcorePlatformCPU;
    }
}

void RelayoutForwardImpl::exec(
        _megdnn_tensor_in src0, _megdnn_tensor_out dst0,
        Handle *src_handle) {
    check_cpu_handle(src_handle);
    TensorND src = src0, dst = dst0;
    check_layout_and_canonize(src.layout, dst.layout);
    do_exec(src, dst);
}

void RelayoutForwardImpl::do_exec(_megdnn_tensor_in src,
                                  _megdnn_tensor_out dst) {
    MIDOUT_BEGIN(naive_relayout, midout_iv(0)) {
        switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                    \
    case DTypeEnum::_dt: {                                         \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                              \
                do_copy<DTypeTrait<dtype::_dt>::ctype>(dst, src)); \
        return;                                                    \
    }
            MEGDNN_FOREACH_DTYPE_NAME(cb)
            MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
            default:
                megdnn_throw("bad dtype");
        }
    }
    MIDOUT_END();
}

void RelayoutForwardImpl::check_cpu_handle(Handle *handle) {
    megdnn_assert(!handle || handle == this->handle()
            || is_cpu_handle(handle),
            "relayout from non-CPU to CPU not supported");
}

// vim: syntax=cpp.doxygen
