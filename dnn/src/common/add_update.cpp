/**
 * \file dnn/src/common/add_update.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs.h"

#include "src/common/add_update_helper.h"
#include "src/common/utils.h"

namespace megdnn {

void AddUpdateForward::check_exec(const TensorLayout& dst,
                                  const TensorLayout& delta) {
    // delta can not be broadcasted to dst if dst.total_nr_elems() <
    // delta.total_nr_elems()
    megdnn_assert(dst.dtype == delta.dtype &&
                  dst.total_nr_elems() >= delta.total_nr_elems() &&
                  dst.is_non_overlapping_strong());
    if (dst.dtype.category() == DTypeCategory::INT) {
        auto check_fv = [](float fv) {
            int iv = fv;
            megdnn_assert(
                    float(iv) == fv && float(iv + 1) == fv + 1.f &&
                            float(iv - 1) == fv - 1.f,
                    "bad arg value in AddUpdate: dtype is int, but value is %g "
                    "which can not be precisely converted to int",
                    fv);
        };
        check_fv(m_param.alpha);
        check_fv(m_param.beta);
        check_fv(m_param.bias);
    }
}

ElemwiseOpParamN<2> AddUpdateForwardHelper::make_param(
        _megdnn_tensor_inout dst, _megdnn_tensor_in delta) {
    ElemwiseOpParamN<2> src;
    src[0] = dst;
    src[1] = delta;
    src[1].layout = src[1].layout.broadcast(dst.layout);
    src.init_from_given_tensor();

    return src;
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen
