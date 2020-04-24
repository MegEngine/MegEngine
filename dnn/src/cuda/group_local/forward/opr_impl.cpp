/**
 * \file dnn/src/cuda/group_local/forward/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/group_local/opr_impl.h"
#include <memory>

#include "megdnn/opr_param_defs.h"
#include "src/common/utils.h"
#include "src/cuda/utils.h"

#include "src/cuda/group_local/forward/kern.cuh"
#include "src/cuda/local/opr_impl.h"

#include "src/cuda/local/local.cuh"

using namespace megdnn;
using namespace cuda;

namespace {

std::unique_ptr<LocalForward> get_opr(Handle* handle,
                                      param::Convolution param) {
    auto&& opr = handle->create_operator<LocalForward>();
    opr->param() = param;
    return std::move(opr);
}

template <typename T>
void incr_ptr(T*& dst, ptrdiff_t delta) {
    dst = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(dst) + delta);
}

TensorLayout prepare_src_dst(const TensorLayout& input, size_t g) {
    TensorLayout ret = input;
    megdnn_assert(ret[1] % g == 0);
    ret[1] = ret[1] / g;
    ret.init_contiguous_stride();
    //! change stride of batch
    ret.stride[0] = input.stride[0];
    return ret;
}

TensorLayout prepare_filter(const TensorLayout& filter) {
    //! group, OH, OW, ICg, FH, FW, OCg -> OH, OW, IcCg, FH, FW, OCg
    return {{filter[1], filter[2], filter[3], filter[4], filter[5], filter[6]},
            filter.dtype};
}

}  // namespace

void GroupLocalForwardImpl::exec(_megdnn_tensor_in src,
                                 _megdnn_tensor_in filter,
                                 _megdnn_tensor_out dst,
                                 _megdnn_workspace workspace) {
    megdnn_assert(src.layout.dtype == dtype::Float32(),
                  "cuda do not support fp16 group local operator");
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);

    auto handle = concrete_handle(this->handle());
    auto G = filter.layout[0];
    auto IH = src.layout.shape[2], IW = src.layout.shape[3],
         OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    if (prefer_inference_kernel(src.layout, filter.layout, dst.layout)) {
        auto N = src.layout.shape[0], ICg = src.layout.shape[1] / G,
             OCg = dst.layout.shape[1] / G;
        auto FH = filter.layout.shape[4], FW = filter.layout.shape[5];
        auto PH = param().pad_h, PW = param().pad_w;
        auto SH = param().stride_h, SW = param().stride_w;
        const float* sptr = src.ptr<dt_float32>();
        const float* fptr = filter.ptr<dt_float32>();
        float* dptr = dst.ptr<dt_float32>();
        float* wptr = workspace.ptr<dt_float32>();
        auto stream = cuda_stream(this->handle());

        group_local::exec(sptr, fptr, dptr, wptr, N, ICg, IH, IW, OCg, OH, OW,
                          FH, FW, G, PH, PW, SH, SW, stream);
    } else {
        auto&& opr = get_opr(handle, param());
        TensorND src_g = {src.raw_ptr, prepare_src_dst(src.layout, G)};
        TensorND dst_g = {dst.raw_ptr, prepare_src_dst(dst.layout, G)};
        TensorND filter_g = {filter.raw_ptr, prepare_filter(filter.layout)};
        for (size_t g = 0; g < G; ++g) {
            opr->exec(src_g, filter_g, dst_g, workspace);
            incr_ptr(src_g.raw_ptr, src_g.layout.stride[1] *
                                            src_g.layout.shape[1] *
                                            src_g.layout.dtype.size());
            incr_ptr(dst_g.raw_ptr, dst_g.layout.stride[1] *
                                            dst_g.layout.shape[1] *
                                            dst_g.layout.dtype.size());
            incr_ptr(filter_g.raw_ptr, filter_g.layout.span().dist_byte());
        }
    }
}

GroupLocalForwardImpl::GroupLocalForwardImpl(Handle* handle)
        : GroupLocalForward(handle) {}

size_t GroupLocalForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                     const TensorLayout& filter,
                                                     const TensorLayout& dst) {
    if (prefer_inference_kernel(src, filter, dst)) {
        return 0;
    } else {
        auto G = filter[0];
        TensorLayout src_g = prepare_src_dst(src, G);
        TensorLayout dst_g = prepare_src_dst(dst, G);
        TensorLayout filter_g = prepare_filter(filter);
        return get_opr(handle(), param())
                ->get_workspace_in_bytes(src_g, filter_g, dst_g);
    }
}

bool GroupLocalForwardImpl::prefer_inference_kernel(const TensorLayout& src,
                                                    const TensorLayout& filter,
                                                    const TensorLayout& dst) {
    MEGDNN_MARK_USED_VAR(filter);
    MEGDNN_MARK_USED_VAR(dst);
    auto handle = concrete_handle(this->handle());
    size_t N = src.shape[0], IH = src.shape[2], IW = src.shape[3];
    return N <= 8 && handle->device_prop().sharedMemPerBlock >=
                             group_local::get_share_mem_in_bytes(IH, IW);
}

// vim: syntax=cpp.doxygen
