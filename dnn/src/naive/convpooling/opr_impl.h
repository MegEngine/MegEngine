/**
 * \file dnn/src/naive/convpooling/opr_impl.h
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
#include "src/naive/pooling/opr_impl.h"
#include "src/naive/elemwise/opr_impl.h"

namespace megdnn {
namespace naive {
    
class ConvPoolingForwardImpl final: public ConvPoolingForward {
    public:
        ConvPoolingForwardImpl(Handle *handle);
        void exec( const _megdnn_in TensorND src,
                   const _megdnn_in TensorND filter, 
                   const _megdnn_in TensorND bias, 
                  _megdnn_out TensorND dst,
                  _megdnn_out Workspace workspace) override;
        void deduce_layout(
                const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                TensorLayout & dst) override;
        void check_layout(
                const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                TensorLayout & dst,
                size_t workspace_limit_in_bytes) override;
        size_t get_workspace_in_bytes(const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                const TensorLayout & dst) override;
    private:
        void setParamOfSublayers();
        TensorLayout conv_dst_layout;
        PoolingForwardImpl *poolFwd;
        ElemwiseForwardImpl *nonlineFwd; 
        ConvolutionForwardImpl *convFwd;
};

} // namespace naive
} // namespace megdnn
// vim: syntax=cpp.doxygen