/**
 * \file dnn/src/common/elemwise_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/elemwise_helper.cuh"
#include "src/common/utils.h"

namespace megdnn {

    template<int arity>
    void ElemwiseOpParamN<arity>::init_from_given_tensor() {
        megdnn_assert(!size && max_ndim == -1);
        max_ndim = 0;
        for (int i = 0; i < arity; ++ i) {
            TensorLayout &layout = param[i].layout;
            layout = layout.collapse_contiguous();
            auto cur = layout.total_nr_elems();
            if (!i) {
                size = cur;
            } else {
                megdnn_assert(size == cur);
            }
            max_ndim = std::max<int>(max_ndim, layout.ndim);
        }
        megdnn_assert(size > 0 && max_ndim > 0);
    }

    template<int arity>
    void ElemwiseOpParamN<arity>::assert_initialized() const {
        megdnn_assert(size, "uninitialized ElemwiseOpParamN");
    }

    template struct ElemwiseOpParamN<6>;
    template struct ElemwiseOpParamN<5>;
    template struct ElemwiseOpParamN<4>;
    template struct ElemwiseOpParamN<3>;
    template struct ElemwiseOpParamN<2>;
    template struct ElemwiseOpParamN<1>;

    void ElemwiseOpParamN<0>::assert_initialized() const {
        megdnn_assert(size, "uninitialized ElemwiseOpParamN");
    }
}

// vim: ft=cpp syntax=cpp.doxygen
