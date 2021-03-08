/**
 * \file
 * dnn/src/cuda/convolution/backward_data/implicit_gemm_int8_nchw4_dp4a.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./algo.h"
#include "src/cuda/utils.h"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/convolution/backward_data/cutlass_deconvolution_wrapper.cuh"

using namespace megdnn;
using namespace cuda;

bool ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::
        is_available(const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    if (fm.format != Param::Format::NCHW)
        return false;

    bool available = true;

    auto src_dtype = args.diff_layout->dtype,
         filter_dtype = args.filter_layout->dtype,
         dst_dtype = args.grad_layout->dtype;

    available &= (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // TODO support group deconv int8
    available &= (fm.group == 1);
    // ic and oc must be multiples of 4
    available &= ((fm.group * fm.icpg) % 4 == 0 && (fm.group * fm.ocpg) % 4 == 0);
    // mode must be cross correlation
    available &= !fm.should_flip;
    // mode must be 2D
    available &= fm.spatial_ndim == 2;
    // TODO: support dialtion
    available &= (fm.dilation[0] == 1 && fm.dilation[1] == 1);
    // FIXME: too large filter size is not supported now
    available &= fm.spatial[0] * fm.spatial[1] <= 64;
    // only support sm_61 or later, platform should have fast native int8
    // support
    available &= is_compute_capability_required(6, 1);

    return available;
}

WorkspaceBundle ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::
        get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const {
    size_t ws_filter = args.filter_layout->span().dist_byte();
    size_t ws_diff = args.diff_layout->span().dist_byte();
    size_t ws_grad = args.grad_layout->span().dist_byte();
    return WorkspaceBundle{raw_ptr, {ws_filter, ws_diff, ws_grad}};
}

size_t ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::
        get_workspace_in_bytes(const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::exec(
        const ExecArgs& args) const {
    auto&& fm = args.filter_meta;
    size_t n = args.diff_layout->operator[](0),
           co = args.diff_layout->operator[](1),
           ho = args.diff_layout->operator[](2),
           wo = args.diff_layout->operator[](3);
    size_t ci = args.grad_layout->operator[](1),
           hi = args.grad_layout->operator[](2),
           wi = args.grad_layout->operator[](3);
    size_t fh = fm.spatial[0], fw = fm.spatial[1];
    size_t sh = fm.stride[0], sw = fm.stride[1];
    size_t ph = fm.padding[0], pw = fm.padding[1];

    auto&& stream = cuda_stream(args.opr->handle());

    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);

    int8_t* inner_filter_ptr = nullptr;
    int8_t* inner_diff_ptr = nullptr;
    // TODO: weight preprocess
    {
        inner_filter_ptr = reinterpret_cast<int8_t*>(bundle.get(0));
        // reformat filter from nchw to n4hwc4
        TensorLayout exec_src{{co / 4, 4, ci, fh, fw}, dtype::Int8()};
        TensorLayout exec_dst{{co / 4, fh, fw, ci, 4}, dtype::Int8()};

        exec_src = exec_src.dimshuffle({0, 3, 4, 2, 1});

        auto&& relayout =
                args.opr->handle()->create_operator<RelayoutForward>();
        relayout->exec({args.filter_tensor->raw_ptr, exec_src},
                       {inner_filter_ptr, exec_dst});
    }
    {
        inner_diff_ptr = reinterpret_cast<int8_t*>(bundle.get(1));
        // reformat diff from nchw to nchw4
        TensorLayout exec_src{{n, co / 4, 4, ho, wo}, dtype::Int8()};
        TensorLayout exec_dst{{n, co / 4, ho, wo, 4}, dtype::Int8()};

        exec_src = exec_src.dimshuffle({0, 1, 3, 4, 2});

        auto&& relayout =
                args.opr->handle()->create_operator<RelayoutForward>();
        relayout->exec({args.diff_tensor->raw_ptr, exec_src},
                       {inner_diff_ptr, exec_dst});
    }
    int8_t* inner_grad_ptr = reinterpret_cast<int8_t*>(bundle.get(2));

    convolution::ConvParam kern_param;
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ho = ho,
    kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.sh = sh, kern_param.sw = sw, kern_param.fh = fh,
    kern_param.fw = fw;

    float diff_scale =
                  args.diff_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS8>().scale,
          grad_scale =
                  args.grad_layout->dtype.param<dtype::QuantizedS8>().scale;
    float alpha = diff_scale * filter_scale / grad_scale;

    // only use 16x64x8_16x64x8_2stages impl
    cutlass_wrapper::do_deconv_int8_implicit_gemm_dp4a_ncdiv4hw4(
            inner_diff_ptr, inner_filter_ptr, inner_grad_ptr, nullptr,
            kern_param, alpha, cutlass_wrapper::GemmCoord{16, 64, 8},
            cutlass_wrapper::GemmCoord{16, 64, 8}, 2, stream);

    after_kernel_launch();

    {
        // reformat grad from nchw4 to nchw
        TensorLayout exec_src{{n, ci / 4, hi, wi, 4}, dtype::Int8()};
        TensorLayout exec_dst{{n, ci / 4, 4, hi, wi}, dtype::Int8()};

        exec_src = exec_src.dimshuffle({0, 1, 4, 2, 3});

        auto&& relayout =
                args.opr->handle()->create_operator<RelayoutForward>();
        relayout->exec({inner_grad_ptr, exec_src},
                       {args.grad_tensor->raw_ptr, exec_dst});
    }
}
// vim: syntax=cpp.doxygen
