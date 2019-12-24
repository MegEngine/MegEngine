/**
 * \file test/common/extra_impl_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "test/common/opr_proxy.h"
#include "megdnn/oprs/general.h"
#include "megdnn/basic_types.h"
#include "megdnn/handle.h"

namespace megdnn {
namespace test {

template <typename Opr, int NR_OUTPUTS=1, typename Proxy = OprProxy<Opr>>
std::function<void(const TensorNDArray&)> extra_impl_helper(
        Handle* h, const typename Opr::Param& p) {
    auto impl = [](const TensorNDArray& tensors, Handle* h,
                   const typename Opr::Param& p) {
        static_assert(NR_OUTPUTS <= OprTrait<Opr>::arity,
                      "OutNumber should less than or equal to arity.");
        Proxy proxy;
        auto fp32_opr = h->create_operator<Opr>();
        auto type_cvt = h->create_operator<TypeCvt>();
        fp32_opr->param() = p;

        TensorNDArray fp32_tensors;
        for (size_t i = 0; i < tensors.size(); ++i) {
            auto layout = tensors[i].layout;
            layout.dtype = dtype::Float32();
            fp32_tensors.emplace_back(malloc(layout.span().dist_byte()),
                                      layout);
            type_cvt->exec(tensors[i], fp32_tensors[i]);
        }

        proxy.exec(fp32_opr.get(), fp32_tensors);

        for (size_t i = fp32_tensors.size() - NR_OUTPUTS;
             i < fp32_tensors.size(); ++i) {
            type_cvt->exec(fp32_tensors[i], tensors[i]);
        }

        for (size_t i = 0; i < tensors.size(); ++i) {
            free(fp32_tensors[i].raw_ptr);
        }
    };
    return std::bind(impl, std::placeholders::_1, h, std::cref(p));
}

template <>
std::function<void(const TensorNDArray&)> extra_impl_helper<AddUpdate>(
        Handle* h, const AddUpdate::Param& p);

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
