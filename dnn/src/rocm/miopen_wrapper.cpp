/**
 * \file dnn/src/rocm/miopen_wrapper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "megdnn/opr_param_defs.h"
#include "src/rocm/miopen_wrapper.h"

#include "src/common/utils.h"
#include "src/rocm/utils.h"

namespace {

using namespace megdnn;

miopenDataType_t to_miopen_dtype(DType type,
                                 const param::Convolution::Format format = {}) {
    MEGDNN_MARK_USED_VAR(format);
    //! TODO check quantized type
    switch (type.enumv()) {
        case DTypeEnum::Float32:
            return miopenFloat;
#if !MEGDNN_DISABLE_FLOAT16
        case DTypeEnum::Float16:
            return miopenHalf;
#endif
        case DTypeEnum::Int32:
        case DTypeEnum::QuantizedS32:
            return miopenInt32;
        case DTypeEnum::QuantizedS8:
        case DTypeEnum::Int8:
            return miopenInt8;
        default:
            megdnn_throw(
                    megdnn_mangle("dtype must be float16/float32/int8/int32"));
    }
}
}  // namespace

namespace megdnn {
namespace rocm {

TensorDesc::TensorDesc() {
    miopen_check(miopenCreateTensorDescriptor(&desc));
}

TensorDesc::~TensorDesc() {
    miopen_check(miopenDestroyTensorDescriptor(desc));
}

void TensorDesc::set(const TensorLayout& layout,
                     const param::Convolution::Format format) {
    megdnn_assert(format == param::Convolution::Format::NCHW,
                  "for now, miopen only support NCHW format");
    megdnn_assert_eq_size_t(layout.ndim, 4_z);
    int n = layout[0];
    int c = layout[1];
    int h = layout[2];
    int w = layout[3];
    miopen_check(miopenSet4dTensorDescriptor(
            desc, to_miopen_dtype(layout.dtype), n, c, h, w));
}

ConvDesc::ConvDesc() {
    miopen_check(miopenCreateConvolutionDescriptor(&desc));
}

ConvDesc::~ConvDesc() {
    miopen_check(miopenDestroyConvolutionDescriptor(desc));
}

void ConvDesc::set(const param::Convolution& param, const size_t nr_group,
                   const bool is_depthwise) {
    miopenConvolutionMode_t mode;
    if (param.mode == param::Convolution::Mode::CROSS_CORRELATION) {
        mode = miopenConvolution;
        if (param.sparse == param::Convolution::Sparse::GROUP) {
            mode = is_depthwise ? miopenDepthwise : miopenGroupConv;
        }
    } else {
        megdnn_throw(megdnn_mangle(
                "for now, miopen do not support non xcorr convolution"));
    }

    miopen_check(miopenInitConvolutionDescriptor(
            desc, mode, param.pad_h, param.pad_w, param.stride_h,
            param.stride_w, param.dilate_h, param.dilate_w));
    if (mode == miopenGroupConv || mode == miopenDepthwise) {
        miopen_check(miopenSetConvolutionGroupCount(desc, nr_group));
    }
    //! miopen do not support set compute_type, so mixed precision training is
    //! not supported
}

PoolingDesc::PoolingDesc() {
    miopen_check(miopenCreatePoolingDescriptor(&desc));
}

PoolingDesc::~PoolingDesc() {
    miopen_check(miopenDestroyPoolingDescriptor(desc));
}

void PoolingDesc::set(const param::Pooling& param) {
    miopenPoolingMode_t mode;
    switch (param.mode) {
        case param::Pooling::Mode::MAX:
            mode = miopenPoolingMax;
            break;
        case param::Pooling::Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mode = miopenPoolingAverage;
            break;
        case param::Pooling::Mode::AVERAGE:
            mode = miopenPoolingAverageInclusive;
            break;
        default:
            megdnn_throw(megdnn_mangle("Unsupported pooling mode for miopen"));
    }
    miopen_check(miopenSet2dPoolingDescriptor(
            desc, mode, param.window_h, param.window_w, param.pad_h,
            param.pad_w, param.stride_h, param.stride_w));
}

LRNDesc::LRNDesc() {
    miopen_check(miopenCreateLRNDescriptor(&desc));
}

LRNDesc::~LRNDesc() {
    miopen_check(miopenDestroyLRNDescriptor(desc));
}

void LRNDesc::set(const param::LRN& param) {
    MEGDNN_MARK_USED_VAR(param);
//! TODO MIOpen has two LRN Mode, miopenLRNWithinChannel and
//! miopenLRNCrossChannel, need to check what do these modes mean.
}

BNParamDesc::BNParamDesc() {
    miopen_check(miopenCreateTensorDescriptor(&desc));
}

void BNParamDesc::set(const miopenTensorDescriptor_t xDesc,
                      miopenBatchNormMode_t mode) {
    miopen_check(miopenDeriveBNTensorDescriptor(desc, xDesc, mode));
}

BNParamDesc::~BNParamDesc() {
    miopen_check(miopenDestroyTensorDescriptor(desc));
}

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
