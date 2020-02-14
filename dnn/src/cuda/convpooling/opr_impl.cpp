/**
 * \file dnn/src/cuda/convpooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/convpooling/opr_impl.h"
#include "src/cuda/convpooling/conv_pooling.h"
#include "src/cuda/utils.h"
#include "src/cuda/handle.h"

namespace megdnn {
namespace cuda {
using namespace conv_pool;

void get_dest_shape(size_t ih, size_t iw, size_t fh, size_t fw,
        size_t sh, size_t sw, size_t ph, size_t pw,
        size_t &oh, size_t &ow, bool is_floor = true)
{
    megdnn_assert(ih+2*ph >= fh, "input height=%zu, padding height=%zu, "
            "filter height=%zu", ih, ph, fh);
    megdnn_assert(iw+2*pw >= fw, "input width=%zu, padding width=%zu, "
            "filter width=%zu", iw, pw, fw);
    megdnn_assert(sh && sw, "invalid stride setting: (%zu, %zu)", sh, sw);
    if (is_floor) {
        oh = (ih+2*ph-fh)/sh + 1;
        ow = (iw+2*pw-fw)/sw + 1;
    } else {
        oh = (ih+2*ph-fh+sh-1)/sh + 1;
        ow = (iw+2*pw-fw+sw-1)/sw + 1;
    }
}

ConvPoolingForwardImpl::ConvPoolingForwardImpl(Handle *handle):
    ConvPoolingForward(handle) {
    return;
}

size_t ConvPoolingForwardImpl::get_workspace_in_bytes(const TensorLayout & /*src*/,
                const TensorLayout & /*filter*/,
                const TensorLayout & /*bias*/,
                const TensorLayout & /*dst*/) {
    return 0;
}

void ConvPoolingForwardImpl::deduce_layout(
        const TensorLayout & srcl,
        const TensorLayout & filterl,
        const TensorLayout & /*bias*/,
        TensorLayout & dstl) {

    megdnn_assert_contiguous(srcl);
    megdnn_assert_contiguous(filterl);
    auto &src = srcl.shape;
    auto &filter = filterl.shape;
    //auto &wsp = workspace.shape;
    //wsp = TensorShape({0, 0, 0, 0});
    //megdnn_assert(src.ndim == 4_z, "%s", errmsg_c);
    //megdnn_assert(filter.ndim == 4_z, "%s", errmsg_c);
    megdnn_assert(srcl.ndim == 4_z, "%s", "src.ndim != 4");
    megdnn_assert(filterl.ndim == 4_z, "%s", "filter.ndim != 4");
    size_t n  = src[0];
    size_t ic = src[1];
    size_t ih = src[2];
    size_t iw = src[3];
    size_t oc = filter[0];
    megdnn_assert(filter[1] == ic, "%s", "filter[1] != ic");
    size_t fh = filter[2];
    size_t fw = filter[3];
    size_t conv_sh = this->param().conv_stride_h;
    size_t conv_sw = this->param().conv_stride_w;
    size_t pool_sh = this->param().pool_stride_h;
    size_t pool_sw = this->param().pool_stride_w;
    size_t conv_ph = this->param().conv_pad_h;
    size_t conv_pw = this->param().conv_pad_w;
    size_t pool_ph = this->param().pool_pad_h;
    size_t pool_pw = this->param().pool_pad_w;
    size_t poolh = this->param().pool_shape_h;
    size_t poolw = this->param().pool_shape_w;
    size_t conv_oh, conv_ow, oh, ow;
    // Shape of the output of convoluation.
    get_dest_shape(ih, iw, fh, fw, conv_sh, conv_sw,
    			   conv_ph, conv_pw, conv_oh, conv_ow);
    // Shape of the output of pooling.
    get_dest_shape(conv_oh, conv_ow, poolh, poolw,
    			pool_sh, pool_sw, pool_ph, pool_pw, oh, ow);

    dstl = TensorLayout(TensorShape{n, oc, oh, ow}, srcl.dtype);
    //workspace = Workspace(NULL, 0);
    //workspace.gen_default_stride();
}

void ConvPoolingForwardImpl::check_layout (
                const TensorLayout & src,
                const TensorLayout & filter,
                const TensorLayout & bias,
                TensorLayout & dst,
                size_t /* workspace_limit_in_bytes */
                ) {

    TensorLayout dst_expected;
    deduce_layout(src, filter, bias, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);

    megdnn_assert(bias.shape[1] == dst.shape[1]);
    megdnn_assert(dst.shape[1] == filter.shape[0]);
}

void ConvPoolingForwardImpl::exec(const _megdnn_in TensorND src,
                   const _megdnn_in TensorND filter,
                   const _megdnn_in TensorND bias,
                  _megdnn_out TensorND dst,
                  _megdnn_out Workspace workspace) {
	check_layout(src.layout, filter.layout, bias.layout, dst.layout, workspace.size);
	auto stream = cuda_stream(this->handle());
	size_t N = src.layout.shape[0];
    size_t IC = src.layout.shape[1];
    size_t IH = src.layout.shape[2];
    size_t IW = src.layout.shape[3];
    size_t OC = dst.layout.shape[1];
    size_t OH = dst.layout.shape[2];
    size_t OW = dst.layout.shape[3];

    size_t FH = filter.layout.shape[2];
    size_t FW = filter.layout.shape[3];
    size_t CONV_PH = this->param().conv_stride_h;
    size_t CONV_PW = this->param().conv_stride_w;
    size_t CONV_SH = this->param().conv_stride_h;
    size_t CONV_SW = this->param().conv_stride_w;
    size_t POOL_H = this->param().pool_shape_h;
    size_t POOL_W = this->param().pool_shape_w;

   	PoolModeCu poolMode;
    switch(this->param().poolMode) {
        case Param::PoolMode::AVERAGE:
            poolMode = AVERAGE;
        break;
        case Param::PoolMode::MAX:
            poolMode = MAX;
        break;
        default:
            poolMode = AVERAGE;
    }

    ConvModeCu convMode;
    switch(this->param().convMode) {
        case Param::ConvMode::CROSS_CORRELATION:
            convMode = CROSS_CORRELATION;
        break;
        case Param::ConvMode::CONVOLUTION:
            convMode = CONVOLUTION;
        break;
        default:
            convMode = CROSS_CORRELATION;
    }

    NonlineModeCu nonlineMode;
    switch(this->param().nonlineMode) {
        case Param::NonlineMode::IDENTITY:
            nonlineMode = IDENTITY;
        break;
        case Param::NonlineMode::RELU:
            nonlineMode = RELU;
        break;
        case Param::NonlineMode::SIGMOID:
            nonlineMode = SIGMOID;
        break;
        default:
            nonlineMode = IDENTITY;
    }

    float *src_ptr = static_cast<float*>(src.raw_ptr),
    *filter_ptr = static_cast<float*>(filter.raw_ptr),
    *bias_ptr = static_cast<float*>(bias.raw_ptr),
    *dst_ptr = static_cast<float*>(dst.raw_ptr);

   	switch (this->param().method) {
		case Param::Method::WITH_SHARED_MEM:
        // This method is out-of-date.
	    /*
        start_gpu_xcorr_pool_with_shared_mem(stream, src_ptr, filter_ptr, dst_ptr,
				N, IC,  IH,  IW, OC,  OH,  OW,
				FH,  FW, CONV_PH,  CONV_PW, CONV_SH,  CONV_SW,
				this->param().pool_shape_w,
				poolMode,
				this->param().relu,
				bias_ptr);

	    break;
        */
    	case Param::Method::WITH_TEXTURE_OBJ:
    	start_gpu_xcorr_pool_with_texture_obj(stream, src_ptr, filter_ptr, dst_ptr,
				N, IC,  IH,  IW, OC,  OH,  OW,
				FH,  FW, CONV_PH,  CONV_PW, CONV_SH,  CONV_SW,
                POOL_H, POOL_W,
                poolMode, convMode, nonlineMode, bias_ptr);
        break;

    	default:
    	start_gpu_xcorr_pool_with_texture_obj(stream, src_ptr, filter_ptr, dst_ptr,
				N, IC,  IH,  IW, OC,  OH,  OW,
				FH,  FW, CONV_PH,  CONV_PW, CONV_SH,  CONV_SW,
                POOL_H, POOL_W,
                poolMode, convMode, nonlineMode, bias_ptr);
	}
}


} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
