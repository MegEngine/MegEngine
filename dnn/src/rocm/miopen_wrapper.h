/**
 * \file dnn/src/rocm/miopen_wrapper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/oprs/nn.h"
#include "src/rocm/miopen_with_check.h"

namespace megdnn {
namespace rocm {

class TensorDesc {
public:
    TensorDesc();
    //! default layout is nchw
    void set(const TensorLayout& layout,
             const param::Convolution::Format =
                     param::Convolution::Format::NCHW);
    ~TensorDesc();
    miopenTensorDescriptor_t desc;
};

class ConvDesc {
public:
    ConvDesc();
    //! We need more information to determine detphwise convolution
    void set(const param::Convolution& param, const size_t nr_group,
             const bool is_depthwise = false);
    ~ConvDesc();
    miopenConvolutionDescriptor_t desc;
};

class PoolingDesc {
public:
    PoolingDesc();
    void set(const param::Pooling& param);
    ~PoolingDesc();
    miopenPoolingDescriptor_t desc;
};

class LRNDesc {
public:
    LRNDesc();
    void set(const param::LRN& param);
    ~LRNDesc();
    miopenLRNDescriptor_t desc;
};

class BNParamDesc {
public:
    BNParamDesc();
    void set(const miopenTensorDescriptor_t xDesc, miopenBatchNormMode_t mode);
    ~BNParamDesc();
    miopenTensorDescriptor_t desc;
};

// for now miopen do not support 3d convolution

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
