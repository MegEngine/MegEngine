/**
 * \file dnn/src/cuda/convolution/backward_data/implicit_gemm_int8_nchw4_dp4a.cpp
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
#include "src/cuda/convolution/backward_data/deconv_int8_helper.cuh"

using namespace megdnn;
using namespace cuda;

bool ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        is_available(const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    if (fm.format != Param::Format::NCHW4)
        return false;

    if (!args.grad_layout->is_contiguous() ||
        !args.diff_layout->is_contiguous()) {
        return false;
    }

    bool available = true;

    auto src_dtype = args.diff_layout->dtype,
         filter_dtype = args.filter_layout->dtype,
         dst_dtype = args.grad_layout->dtype;

    available &= (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // TODO support group deconv int8
    available &= (fm.group == 1);
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

WorkspaceBundle ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const {
    size_t ws_filter = args.filter_layout->span().dist_byte();
    return WorkspaceBundle{raw_ptr, {ws_filter}};
}

size_t ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        get_workspace_in_bytes(const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::exec(
        const ExecArgs& args) const {
    auto&& fm = args.filter_meta;
    size_t n = args.diff_layout->operator[](0),
           co = args.diff_layout->operator[](1) * 4,
           ho = args.diff_layout->operator[](2),
           wo = args.diff_layout->operator[](3);
    size_t ci = args.grad_layout->operator[](1) * 4,
           hi = args.grad_layout->operator[](2),
           wi = args.grad_layout->operator[](3);
    size_t fh = fm.spatial[0], fw = fm.spatial[1];
    size_t sh = fm.stride[0], sw = fm.stride[1];
    size_t ph = fm.padding[0], pw = fm.padding[1];

    auto&& stream = cuda_stream(args.opr->handle());

    int8_t* filter_ptr = nullptr;
    // TODO: weight preprocess
    {
        filter_ptr = reinterpret_cast<int8_t*>(args.workspace.raw_ptr);
        // reformat filter from nc4hw4 to n4hwc4
        megdnn::cuda::deconv::reorder_filter_nc4hw4_to_n4hwc4(
                filter_ptr, args.filter_tensor->compatible_ptr<int8_t>(), co,
                ci, fh, fw, stream);
    }
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
    cutlass_wrapper::do_deconv_int8_implicit_gemm_dp4a_ncdiv4hw4(
            args.diff_tensor->compatible_ptr<int8_t>(), filter_ptr,
            args.grad_tensor->compatible_ptr<int8_t>(), nullptr, kern_param,
            alpha,
            cutlass_wrapper::GemmCoord{m_algo_param.threadblock_m,
                                       m_algo_param.threadblock_n,
                                       m_algo_param.threadblock_k},
            cutlass_wrapper::GemmCoord{m_algo_param.warp_m, m_algo_param.warp_n,
                                       m_algo_param.warp_k},
            m_algo_param.stage, stream);

    after_kernel_launch();
}

void ConvolutionBackwardDataImpl::AlgoPack::fill_int8_dp4a_algos() {
    using AlgoParam = AlgoInt8NCHW4DotProdImplicitGemm::AlgoParam;
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 128, 16, 16, 64, 16, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 128, 16, 16, 128, 16, 1});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 128, 32, 32, 64, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{64, 128, 32, 64, 32, 32, 2});
}

// vim: syntax=cpp.doxygen
