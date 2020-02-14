/**
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include "./quint4x4x32_wmma/activation_u4.cuh"
#include "./quint4x4x32_wmma/reduce_with_scale_data.cuh"
#include "./quint4x4x32_wmma/reduce_with_scale_filter.cuh"
#include "./quint4x4x32_wmma/wmma_conv_integer_u4.cuh"

using namespace megdnn;
using namespace cuda;
using namespace activation_u4;

#if CUDA_VERSION >= 10000
bool ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;

    bool available = true;
    auto&& filter_meta = args.filter_meta;
    // FH, FW must be 3, 5, 7
    available &= (filter_meta.spatial[0] == 3 && filter_meta.spatial[1] == 3) ||
                 (filter_meta.spatial[0] == 5 && filter_meta.spatial[1] == 5) ||
                 (filter_meta.spatial[0] == 7 && filter_meta.spatial[1] == 7);
    // stride must be 1
    available &= (filter_meta.stride[0] == 1 && filter_meta.stride[1] == 1);
    // OW must be a multiple of 8
    available &= (args.dst_layout->operator[](3) % 8 == 0);
    // only support dense conv
    auto&& param = args.opr->param();
    using Param = param::ConvBias;
    available &= (param.sparse == Param::Sparse::DENSE);
    // only support cross correlation convolution
    available &= (!args.filter_meta.should_flip);
    // dilate should be 1
    available &= (filter_meta.dilation[0] == 1 && filter_meta.dilation[1] == 1);
    // format should be NCHW8
    available &= (param.format == Param::Format::NCHW8);
    // device support sm_75
    auto&& device_prop = current_device_prop();
    available &= (device_prop.major > 7 ||
                  (device_prop.major == 7 && device_prop.minor >= 5));
    // nonlinmode should be RELU or Identity
    available &= param.nonlineMode == Param::NonlineMode::RELU ||
                 param.nonlineMode == Param::NonlineMode::IDENTITY;
    // IC should be a multiple of 32
    available &= (args.src_layout->operator[](1) * 8) % 32 == 0;
    return available;
}

WorkspaceBundle ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA::get_workspace_bundle(
        dt_byte* raw_ptr, const SizeArgs& args) const {
    // ws_size_zp_filter = OC
    size_t N = args.src_layout->operator[](0);
    size_t OC = args.filter_layout->operator[](0),
           IC = args.filter_layout->operator[](1) * 8,
           FH = args.filter_layout->operator[](2),
           FW = args.filter_layout->operator[](3);
    size_t OH = args.dst_layout->operator[](2),
           OW = args.dst_layout->operator[](3);

    size_t ws_size_zp_filter = OC * sizeof(int32_t);
    // for reduce filter
    {
        size_t A = OC, B = IC * FH * FW / 8, C = 1;
        ws_size_zp_filter += _do_dispatch_reduce_workspace_in_bytes(A, B, C);
    }
    size_t ws_size_zp_data = N * OH * OW * sizeof(int32_t);
    size_t ws_size_relayout_filter = get_workspace_in_bytes_do_conv(args);
    if (ws_size_relayout_filter > 0) {
        WorkspaceBundle ws{
                raw_ptr,
                {ws_size_zp_filter, ws_size_zp_data, ws_size_relayout_filter}};
        return ws;
    }
    WorkspaceBundle ws{raw_ptr, {ws_size_zp_filter, ws_size_zp_data}};
    return ws;
}

size_t ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

bool ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA::use_kernel_fhxfw(
        const SizeArgs& args) const {
    return (args.filter_meta.spatial[0] == 3 &&
            args.filter_meta.spatial[1] == 3);
}

size_t ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA::get_workspace_in_bytes_do_conv(
        const SizeArgs& args) const {
    if (use_kernel_fhxfw(args))
        return 0_z;
    size_t OC = args.filter_layout->operator[](0),
           IC = args.filter_layout->operator[](1) * 8,
           FH = args.filter_layout->operator[](2),
           FW = args.filter_layout->operator[](3);
    return OC * IC * FH * FW / 2;
}

void ConvBiasForwardImpl::AlgoQUInt4x4x32WMMA::exec(
        const ExecArgs& args) const {
    auto&& handle = concrete_handle(args.opr->handle());
    auto&& ws_bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto&& ws_zp_filter = ws_bundle.get_workspace(0);
    auto&& ws_zp_data = ws_bundle.get_workspace(1);
    size_t N = args.src_layout->operator[](0),
           IC = args.src_layout->operator[](1) * 8,
           IH = args.src_layout->operator[](2),
           IW = args.src_layout->operator[](3),
           OC = args.filter_layout->operator[](0),
           FH = args.filter_meta.spatial[0], FW = args.filter_meta.spatial[1],
           OH = args.dst_layout->operator[](2),
           OW = args.dst_layout->operator[](3),
           PH = args.filter_meta.padding[0], PW = args.filter_meta.padding[1],
           SH = args.filter_meta.stride[0], SW = args.filter_meta.stride[1];
    int32_t zp_data =
            args.src_layout->dtype.param<dtype::Quantized4Asymm>().zero_point;
    int32_t zp_filter =
            args.filter_layout->dtype.param<dtype::Quantized4Asymm>()
                    .zero_point;
    int32_t zp_data_filter = zp_data * zp_filter * FH * FW * IC;
    auto&& stream = cuda_stream(handle);
    // zp filter
    _do_dispatch_reduce_with_scale_filter_u4(
            static_cast<uint8_t*>(args.filter_tensor->raw_ptr), -zp_data, OC,
            FH * FW * IC / 8, ws_zp_filter.ptr<int32_t>(), stream);
    // zp data
    _do_dispatch_reduce_with_scale_data_u4(
            ws_zp_data.ptr<int32_t>(),
            static_cast<uint8_t*>(args.src_tensor->raw_ptr), N, IH, IW, OH, OW,
            PH, PW, FH, FW, SH, SW, IC, -zp_filter,
            static_cast<uint8_t>(zp_data), stream);

    // do conv
    if (use_kernel_fhxfw(args)) {
        wmma_conv_integer_subbyte::_do_wmma_conv_integer_subbyte_fhxfw(
                static_cast<uint8_t*>(args.src_tensor->raw_ptr),
                static_cast<uint8_t*>(args.filter_tensor->raw_ptr),
                args.dst_tensor->compatible_ptr<int32_t>(), N, IH, IW, OH, OW,
                PH, PW, IC, OC, FH, FW, SH, SW, static_cast<uint8_t>(zp_data),
                stream);
    } else {
        auto&& ws_relayout_filter = ws_bundle.get_workspace(2);
        wmma_conv_integer_subbyte::_do_wmma_conv_integer_subbyte_1xfw(
                static_cast<uint8_t*>(args.src_tensor->raw_ptr),
                static_cast<uint8_t*>(args.filter_tensor->raw_ptr),
                args.dst_tensor->compatible_ptr<int32_t>(),
                ws_relayout_filter.ptr<uint8_t>(), N, IH, IW, OH, OW, PH, PW,
                IC, OC, FH, FW, SH, SW, static_cast<uint8_t>(zp_data), stream);
    }
    // do activation
    int s0 = args.bias_layout->stride[0], s1 = args.bias_layout->stride[1],
        s2 = args.bias_layout->stride[2], s3 = args.bias_layout->stride[3];
    s0 = args.bias_layout->shape[0] == 1 ? 0 : s0;
    s1 = args.bias_layout->shape[1] == 1 ? 0 : s1;
    s2 = args.bias_layout->shape[2] == 1 ? 0 : s2;
    s3 = args.bias_layout->shape[3] == 1 ? 0 : s3;
    activation_u4::BiasVisitor visitor{
            args.bias_tensor->compatible_ptr<int32_t>(), s0, s1, s2, s3};
    auto&& param = args.opr->param();
    if (param.nonlineMode == Param::NonlineMode::RELU) {
        _do_dispatch_activation_u4<ActivationRELU>(
                args.dst_tensor->compatible_ptr<int32_t>(), visitor,
                ws_zp_data.ptr<int32_t>(), ws_zp_filter.ptr<int32_t>(),
                zp_data_filter, N, OC, OH, OW, stream);
    } else if (param.nonlineMode == Param::NonlineMode::IDENTITY) {
        _do_dispatch_activation_u4<ActivationIdentity>(
                args.dst_tensor->compatible_ptr<int32_t>(), visitor,
                ws_zp_data.ptr<int32_t>(), ws_zp_filter.ptr<int32_t>(),
                zp_data_filter, N, OC, OH, OW, stream);
    }
}
#endif

// vim: syntax=cpp.doxygen
