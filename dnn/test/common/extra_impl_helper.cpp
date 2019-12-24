/**
 * \file test/common/extra_impl_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/extra_impl_helper.h"

namespace megdnn {
namespace test {

template <>
std::function<void(const TensorNDArray&)> extra_impl_helper<AddUpdate>(
        Handle* h, const AddUpdate::Param& p) {
    auto impl = [](const TensorNDArray& tensors, Handle* h,
                   const AddUpdate::Param& p) {
        auto fp32_opr = h->create_operator<AddUpdate>();
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

        fp32_opr->exec(fp32_tensors[0], fp32_tensors[1]);

        type_cvt->exec(fp32_tensors[0], tensors[0]);

        for (size_t i = 0; i < tensors.size(); ++i) {
            free(fp32_tensors[i].raw_ptr);
        }
    };
    return std::bind(impl, std::placeholders::_1, h, std::cref(p));
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
