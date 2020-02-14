/**
 * \file dnn/src/cuda/batch_conv_bias/gemm_int8_nchw4_dp4a.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/cuda/batch_conv_bias/algo.h"
#include "src/cuda/batch_conv_bias/batch_conv_bias.cuh"
#include "src/cuda/batch_conv_bias/opr_impl.h"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/convolution_helper/bias_visitor.cuh"
#include "src/cuda/convolution_helper/epilogue.cuh"
#include "src/cuda/convolution_helper/layout.cuh"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;
namespace {
template <typename BiasVisitor, typename Epilogue>
void dispatch_kernel(const int8_t* d_src, const int8_t* d_filter,
                     BiasVisitor bias_visitor, Epilogue epilogue,
                     const ConvParam& param, float alpha, float beta,
                     cudaStream_t stream) {
    void (*kern_wrapper)(const int8_t*, const int8_t*, BiasVisitor, Epilogue,
                         const ConvParam&, float, float, cudaStream_t);
    using namespace batch_conv_bias;
    int img_pixels = param.ho * param.wo;
    if (img_pixels % 4 == 0) {
        kern_wrapper =
                do_batch_conv_bias_int8_gemm_ncdiv4hw4_ldg_128<BiasVisitor,
                                                               Epilogue>;
    } else {
        kern_wrapper =
                do_batch_conv_bias_int8_gemm_ncdiv4hw4<BiasVisitor, Epilogue>;
    }
    megdnn_assert(kern_wrapper != nullptr);
    return kern_wrapper(d_src, d_filter, bias_visitor, epilogue, param, alpha,
                        beta, stream);
}

template <typename BiasVisitor>
void dispatch_nonlinear_mode(const int8_t* d_src, const int8_t* d_filter,
                             BiasVisitor bias_visitor, const int8_t* d_z,
                             int8_t* d_dst, const ConvParam& param, float alpha,
                             float beta, float gamma, float scale,
                             cudaStream_t stream,
                             param::BatchConvBias::NonlineMode nonlinear_mode) {
    using NonlineMode = megdnn::param_enumv::BatchConvBias::NonlineMode;
    Layout<Format::NCHW4> layout;
    layout.init(param.n, param.co, param.ho, param.wo);
    using namespace batch_conv_bias;
#define DISPATCH_CONV_INT8_EPILOGUE(_act_op)                                 \
    do {                                                                     \
        IConvEpilogue<_act_op> epilogue{d_dst,                               \
                                        d_z,                                 \
                                        layout.batch_stride,                 \
                                        layout.channel_stride / 4,           \
                                        layout.height_stride,                \
                                        layout.width_stride,                 \
                                        gamma,                               \
                                        _act_op{scale, 1.f / scale}};        \
        dispatch_kernel<BiasVisitor, IConvEpilogue<_act_op>>(                \
                d_src, d_filter, bias_visitor, epilogue, param, alpha, beta, \
                stream);                                                     \
        return;                                                              \
    } while (0)
#define cb(_nonline_mode)                                                      \
    if (static_cast<uint32_t>(nonlinear_mode) == NonlineMode::_nonline_mode) { \
        DISPATCH_CONV_INT8_EPILOGUE(Activation<NonlineMode::_nonline_mode>);   \
    }
    MEGDNN_FOREACH_NONLINE_MODE(cb);
    megdnn_throw("unsupported nonlinear mode for conv bias operator");
#undef cb
#undef DISPATCH_CONV_INT8_EPILOGUE
}

#define INST(_visitor)                                                    \
    template void dispatch_nonlinear_mode<_visitor>(                      \
            const int8_t* d_src, const int8_t* d_filter,                  \
            _visitor bias_visitor, const int8_t* d_z, int8_t* d_dst,      \
            const ConvParam& param, float alpha, float beta, float gamma, \
            float scale, cudaStream_t stream,                             \
            param::BatchConvBias::NonlineMode nonlinear_mode);

INST(PerChannelBiasVisitor);

#undef INST
}  // namespace

bool BatchConvBiasForwardImpl::AlgoInt8NCHW4DotProdGemm::is_available(
        const SizeArgs& args) const {
    if (args.bias_layout.ndim <= 0)
        return false;

    using Param = param::BatchConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    bool available = true;
    auto&& param = args.opr->param();
    if (!conv_bias::check_bias_share_in_channel(args.bias_layout, param.format))
        return false;
    if (param.format != Format::NCHW4)
        return false;
    UNPACK_BATCH_CONV_BIAS_NCHW4_PARAM(args.src_layout, args.filter_layout,
                                       args.dst_layout, param);
    // TODO support group conv
    available &= param.sparse == Sparse::DENSE;
    // mode must be cross correlation
    available &= param.mode == Mode::CROSS_CORRELATION;
    // check data type
    auto src_dtype = args.src_layout.dtype,
         filter_dtype = args.filter_layout.dtype,
         bias_dtype = args.bias_layout.dtype, dst_dtype = args.dst_layout.dtype;
    available &= (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  bias_dtype.enumv() == DTypeEnum::QuantizedS32 &&
                  dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // TODO: support dialtion
    available &= dh == 1 && dw == 1;
    // can be treat as gemm
    available &=
            (fh == 1 && sh == 1 && fw == 1 && sw == 1 && ph == 0 && pw == 0);
    // only support sm_61 or later, platform should have fast native int8
    // support
    available &= is_compute_capability_required(6, 1);
    return available;
}

size_t
BatchConvBiasForwardImpl::AlgoInt8NCHW4DotProdGemm::get_workspace_in_bytes(
        const SizeArgs& /* args */) const {
    return 0;
}

void BatchConvBiasForwardImpl::AlgoInt8NCHW4DotProdGemm::exec(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    UNPACK_BATCH_CONV_BIAS_NCHW4_PARAM(args.src_layout, args.filter_layout,
                                       args.dst_layout, param);
    auto&& stream = cuda_stream(args.opr->handle());

    ConvParam kern_param;
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ho = ho,
    kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.sh = sh, kern_param.sw = sw, kern_param.fh = fh,
    kern_param.fw = fw;

    float src_scale = args.src_layout.dtype.param<dtype::QuantizedS8>().scale,
          filter_scale =
                  args.filter_layout.dtype.param<dtype::QuantizedS8>().scale,
          bias_scale =
                  args.bias_layout.dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout.dtype.param<dtype::QuantizedS8>().scale;
    float alpha = src_scale * filter_scale / dst_scale,
          beta = bias_scale / dst_scale;
    int8_t* z_dev_ptr = nullptr;
    float gamma = 1.f;
    if (args.z_layout.ndim > 0) {
        z_dev_ptr = args.z_tensor->compatible_ptr<int8_t>();
        float z_scale = args.z_layout.dtype.param<dtype::QuantizedS8>().scale;
        gamma = z_scale / dst_scale;
    }
    PerChannelBiasVisitor bias_visitor;
    bias_visitor.bias = args.bias_tensor->compatible_ptr<int32_t>();
    dispatch_nonlinear_mode<PerChannelBiasVisitor>(
            args.src_tensor->compatible_ptr<int8_t>(),
            args.filter_tensor->compatible_ptr<int8_t>(), bias_visitor,
            z_dev_ptr, args.dst_tensor->compatible_ptr<int8_t>(), kern_param,
            alpha, beta, gamma, dst_scale, stream, param.nonlineMode);
}

// vim: syntax=cpp.doxygen
