/**
 * \file dnn/src/naive/separable_conv/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/naive/convolution/opr_impl.h"
namespace megdnn {
namespace naive {

class SeparableConvForwardImpl: public SeparableConvForward {
    public:
        //SeparableConvForwardImpl(Handle *handle): SeparableConvForward(handle) {}
        using SeparableConvForward::SeparableConvForward;
        void exec(_megdnn_tensor_in src,
                _megdnn_tensor_in filter_x,
                _megdnn_tensor_in filter_y,
                _megdnn_tensor_out dst,
                _megdnn_workspace workspace) override;

        size_t get_workspace_in_bytes(const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &,
                const TensorLayout &) override
        {
            // TODO: deduce the size of ring buffer.
            return 0;
        }

};

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen
