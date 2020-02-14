/**
 * \file dnn/src/naive/tensor_remap/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/tensor_remap/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"

namespace megdnn {
namespace naive {

void IndexingRemapForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in map,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, map.layout, dst.layout, workspace.size);
    auto kern = [=]() {
        auto &&sshape = src.layout;
        auto &&mshape = map.layout;
        auto &&dshape = dst.layout;
        // Last element is zero to facilitate maddr calculation.
        std::vector<size_t> didx(dshape.ndim+1, 0_z);
        do {
            auto maddr = get_linear_addr(didx.data(), mshape.shape, mshape.ndim);
            std::vector<size_t> sidx(sshape.ndim);
            for (size_t i = 0_z; i < sshape.ndim; ++i) {
                sidx[i] = map.ptr<dt_int32>()[maddr+i];
            }
            auto saddr = get_linear_addr_noncont(sidx.data(), src.layout);
            auto daddr = get_linear_addr_noncont(didx.data(), dst.layout);
            dst.ptr<dt_float32>()[daddr] = src.ptr<dt_float32>()[saddr];
        } while (get_next_addr(didx.data(), dshape.shape, dshape.ndim));
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

void IndexingRemapBackwardImpl::exec(_megdnn_tensor_in diff,
        _megdnn_tensor_in map,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(diff.layout, map.layout, grad.layout, workspace.size);
    auto kern = [=]() {
        auto &&sshape = grad.layout;
        auto &&mshape = map.layout;
        auto &&dshape = diff.layout;
        std::vector<size_t> sidx(sshape.ndim, 0_z);
        {
            // Set grad to zero.
            do {
                auto saddr = get_linear_addr_noncont(sidx.data(), grad.layout);
                grad.ptr<dt_float32>()[saddr] = 0.0f;
            } while (get_next_addr(sidx.data(), sshape.shape, sshape.ndim));
        }
        std::vector<size_t> didx(dshape.ndim+1, 0_z);
        do {
            auto maddr = get_linear_addr(didx.data(), mshape.shape, mshape.ndim);
            std::vector<size_t> sidx(sshape.ndim);
            for (size_t i = 0_z; i < sshape.ndim; ++i) {
                sidx[i] = map.ptr<dt_int32>()[maddr+i];
            }
            auto saddr = get_linear_addr_noncont(sidx.data(), grad.layout);
            auto daddr = get_linear_addr_noncont(didx.data(), diff.layout);
            grad.ptr<dt_float32>()[saddr] += diff.ptr<dt_float32>()[daddr];
        } while (get_next_addr(didx.data(), dshape.shape, dshape.ndim));
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen
