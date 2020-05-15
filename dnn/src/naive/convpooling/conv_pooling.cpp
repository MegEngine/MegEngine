/**
 * \file dnn/src/naive/convpooling/conv_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/convpooling/opr_impl.h"

#include "src/naive/handle.h"
#include "src/common/utils.h"
#include "megdnn/dtype.h"
#include <cstring>

namespace megdnn {
namespace naive {

ConvPoolingForwardImpl::ConvPoolingForwardImpl(Handle *handle):
    ConvPoolingForward(handle) {
    convFwd = new ConvolutionForwardImpl(this->handle());
    poolFwd = new PoolingForwardImpl(this->handle());
    return;
}

void ConvPoolingForwardImpl::setParamOfSublayers() {
    Convolution::Param &cparam = convFwd->param();
    cparam.pad_h = this->param().conv_pad_h;
    cparam.pad_w = this->param().conv_pad_w;
    cparam.stride_h = this->param().conv_stride_h;
    cparam.stride_w = this->param().conv_stride_w;
    // Alternative: Convolution::Mode::CONVOLUTION
    if(this->param().convMode == ConvPoolingBase::Param::ConvMode::CONVOLUTION) {
        cparam.mode = Convolution::Param::Mode::CONVOLUTION;
    } else {
        cparam.mode = Convolution::Param::Mode::CROSS_CORRELATION;
    }
    Pooling::Param &pparam = poolFwd->param();
    pparam.window_h = this->param().pool_shape_h;
    pparam.window_w = this->param().pool_shape_w;
    pparam.stride_h = this->param().pool_stride_h;
    pparam.stride_w = this->param().pool_stride_w;
    pparam.pad_h = this->param().pool_pad_h;
    pparam.pad_w = this->param().pool_pad_w;
    if(this->param().poolMode == ConvPoolingBase::Param::PoolMode::AVERAGE) {
        pparam.mode = PoolingBase::Param::Mode::AVERAGE;
    } else {
        pparam.mode = PoolingBase::Param::Mode::MAX;
    }
}

void ConvPoolingForwardImpl::check_layout(const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                TensorLayout & dst,
                size_t /*workspace_limit_in_bytes*/) {
    TensorLayout dst_expected;
    this->deduce_layout(src, filter, bias, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    megdnn_assert(bias.shape[1] == dst.shape[1]);
    megdnn_assert(dst.shape[1] == filter.shape[0]);
    //megdnn_assert_eq_layout(workspace_expected, workspace);
    return;
}


void ConvPoolingForwardImpl::deduce_layout(
        const TensorLayout &srcl,
        const TensorLayout &filterl,
        const TensorLayout & /*biasl*/,
        TensorLayout &dstl) {
    setParamOfSublayers();
    convFwd->deduce_layout(srcl, filterl, conv_dst_layout);
    poolFwd->deduce_layout(conv_dst_layout, dstl);

}

size_t ConvPoolingForwardImpl::get_workspace_in_bytes(const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                const TensorLayout & /*dst*/) {
	// Worksapce contains the output of convolution layer in the workspace.
	TensorLayout tmp_layout;
    this->deduce_layout(src, filter, bias, tmp_layout);
    return conv_dst_layout.total_nr_elems() * sizeof(float);
}


void ConvPoolingForwardImpl::exec(const _megdnn_in TensorND src,
                   const _megdnn_in TensorND filter,
                   const _megdnn_in TensorND bias,
                  _megdnn_out TensorND dst,
                  _megdnn_out Workspace workspace) {
    Workspace empty_wsp;
    TensorND conv_dst((float*)(workspace.raw_ptr), conv_dst_layout);
    //convFwd->check_layout(src.layout, filter.layout, workspace.layout, empty_wsp.layout);
    check_layout(src.layout, filter.layout, bias.layout, dst.layout, workspace.size);
    convFwd->exec(src, filter, conv_dst, nullptr, empty_wsp);

    // calculate bias
    int conv_dst_batch = conv_dst.layout.shape[0];
    int conv_dst_channel = conv_dst.layout.shape[1];
    int chann_stride =  conv_dst.layout.shape[2] * conv_dst.layout.shape[3];
    float *conv_dst_ptr = conv_dst.ptr<float>();

    for(int batch = 0; batch < conv_dst_batch; ++batch) {
        for(int chan = 0; chan < conv_dst_channel; ++chan) {
            float bias_val = bias.ptr<float>()[chan];

            for(int i = 0; i < chann_stride; ++i, ++conv_dst_ptr) {
                conv_dst_ptr[0] += bias_val;
            }
        }
    }

    // calculate nonline
    nonlineFwd = new ElemwiseForwardImpl(this->handle());
    switch(this->param().nonlineMode) {
        case Param::NonlineMode::RELU:
            nonlineFwd->param().mode = Elemwise::Param::Mode::RELU;
            nonlineFwd->exec({conv_dst}, conv_dst);
        break;
        case Param::NonlineMode::SIGMOID:
            nonlineFwd->param().mode = Elemwise::Param::Mode::SIGMOID;
            nonlineFwd->exec({conv_dst}, conv_dst);
        break;
        case Param::NonlineMode::IDENTITY:
        break;
        default:
        break;
    }

    poolFwd->exec(conv_dst, dst, empty_wsp);
}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen
