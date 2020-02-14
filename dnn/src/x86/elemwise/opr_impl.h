/**
 * \file dnn/src/x86/elemwise/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/elemwise/opr_impl.h"

namespace megdnn {
namespace x86 {

class ElemwiseImpl final: public fallback::ElemwiseImpl {
    bool exec_unary();
    bool exec_binary();
    bool exec_ternary_fma3();

    public:
        using fallback::ElemwiseImpl::ElemwiseImpl;
        void exec(const TensorNDArray &srcs,
                _megdnn_tensor_out dst) override;
};

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen


