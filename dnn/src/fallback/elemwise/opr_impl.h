/**
 * \file dnn/src/fallback/elemwise/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/naive/elemwise/opr_impl.h"

namespace megdnn {
namespace fallback {

class ElemwiseImpl: public naive::ElemwiseForwardImpl {
    template<typename dtype, uint32_t mode>
    void unary_kern(const ElemwiseOpParamN<1> &param);

    template<uint32_t mode>
    void exec_UNARY_INT();

    template<uint32_t mode>
    void exec_UNARY_FLOAT();

    template<typename dtype, uint32_t mode>
    void binary_kern(const ElemwiseOpParamN<2> &param);

    template<uint32_t mode>
    void exec_BINARY_INT();

    template<uint32_t mode>
    void exec_BINARY_FLOAT();

    public:
        using naive::ElemwiseForwardImpl::ElemwiseForwardImpl;
        void exec(const TensorNDArray &srcs,
                _megdnn_tensor_out dst) override;

        bool is_thread_safe() const override { return true; }
};

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen


