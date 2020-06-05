/**
 * \file dnn/src/naive/tensor_remap/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/tensor_remap/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;
namespace {

template <typename ctype>
void forward(const TensorND& src, const TensorND& map, const TensorND& dst) {
    auto&& sshape = src.layout;
    auto&& mshape = map.layout;
    auto&& dshape = dst.layout;
    // Last element is zero to facilitate maddr calculation.
    std::vector<size_t> didx(dshape.ndim + 1, 0_z);
    do {
        auto maddr = get_linear_addr(didx.data(), mshape.shape, mshape.ndim);
        std::vector<size_t> sidx(sshape.ndim);
        for (size_t i = 0_z; i < sshape.ndim; ++i) {
            sidx[i] = map.ptr<dt_int32>()[maddr + i];
        }
        auto saddr = get_linear_addr_noncont(sidx.data(), src.layout);
        auto daddr = get_linear_addr_noncont(didx.data(), dst.layout);
        dst.ptr<ctype>()[daddr] = src.ptr<ctype>()[saddr];
    } while (get_next_addr(didx.data(), dshape.shape, dshape.ndim));
}

template <typename ctype>
void backward(const TensorND& diff, const TensorND& map, const TensorND& grad) {
    auto&& sshape = grad.layout;
    auto&& mshape = map.layout;
    auto&& dshape = diff.layout;
    std::vector<size_t> sidx(sshape.ndim, 0_z);
    {
        // Set grad to zero.
        do {
            auto saddr = get_linear_addr_noncont(sidx.data(), grad.layout);
            grad.ptr<ctype>()[saddr] = 0.0f;
        } while (get_next_addr(sidx.data(), sshape.shape, sshape.ndim));
    }
    std::vector<size_t> didx(dshape.ndim + 1, 0_z);
    do {
        auto maddr = get_linear_addr(didx.data(), mshape.shape, mshape.ndim);
        std::vector<size_t> sidx(sshape.ndim);
        for (size_t i = 0_z; i < sshape.ndim; ++i) {
            sidx[i] = map.ptr<dt_int32>()[maddr + i];
        }
        auto saddr = get_linear_addr_noncont(sidx.data(), grad.layout);
        auto daddr = get_linear_addr_noncont(didx.data(), diff.layout);
        grad.ptr<ctype>()[saddr] += diff.ptr<ctype>()[daddr];
    } while (get_next_addr(didx.data(), dshape.shape, dshape.ndim));
}

}  // anonymous namespace

void IndexingRemapForwardImpl::exec(_megdnn_tensor_in src,
                                    _megdnn_tensor_in map,
                                    _megdnn_tensor_out dst,
                                    _megdnn_workspace workspace) {
    check_exec(src.layout, map.layout, dst.layout, workspace.size);
    switch (src.layout.dtype.enumv()) {
#define cb(dt)                                                  \
    case DTypeTrait<dt>::enumv:                                 \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                           \
                forward<DTypeTrait<dt>::ctype>(src, map, dst)); \
        return;
        cb(dtype::Float32)
        cb(dtype::Int32)
#undef cb

        default:
            megdnn_throw(
                    ssprintf("unsupported dtype %s in indexing "
                             "remap forward naive\n",
                             src.layout.dtype.name()));
    }
}

void IndexingRemapBackwardImpl::exec(_megdnn_tensor_in diff,
                                     _megdnn_tensor_in map,
                                     _megdnn_tensor_out grad,
                                     _megdnn_workspace workspace) {
    check_exec(diff.layout, map.layout, grad.layout, workspace.size);
    switch (diff.layout.dtype.enumv()) {
#define cb(dt)                                                     \
    case DTypeTrait<dt>::enumv:                                    \
        MEGDNN_DISPATCH_CPU_KERN_OPR(                              \
                backward<DTypeTrait<dt>::ctype>(diff, map, grad)); \
        return;
        cb(dtype::Float32)
        cb(dtype::Int32)
#undef cb
        default:
            megdnn_throw(ssprintf(
                    "unsupported dtype %s in indexing remap backward naive\n",
                    diff.layout.dtype.name()));
    }
}

// vim: syntax=cpp.doxygen
