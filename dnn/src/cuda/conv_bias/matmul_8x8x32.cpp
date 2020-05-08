/**
 * \file dnn/src/cuda/conv_bias/matmul_8x8x32.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/conv_bias.h"
#include "src/cuda/utils.h"
#include "src/cuda/utils.cuh"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/matmul/im2col_nhwc_int8.cuh"

using namespace megdnn;
using namespace cuda;

bool ConvBiasForwardImpl::AlgoMatmul8x8x32::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;
    if (!is_compute_capability_required(6, 1))
        return false;

    auto dst_layout = *args.dst_layout;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
    }

    using NonlineMode = param::ConvBias::NonlineMode;
    auto&& fm = args.filter_meta;
    bool available =
            (args.nonlinear_mode == NonlineMode::IDENTITY ||
             args.nonlinear_mode == NonlineMode::RELU) &&
            ((args.src_layout->dtype == dtype::Int8() &&
              dst_layout.dtype == dtype::Int32() &&
              fm.dtype.enumv() == DTypeEnum::Int8) ||
             (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8 &&
              dst_layout.dtype.enumv() == DTypeEnum::QuantizedS32)) &&
            fm.group == 1 && fm.spatial_ndim == 2 &&
            (fm.format == Param::Format::NHWC ||
             fm.format == Param::Format::NCHW4);
    return available;
};

template <param::ConvBias::Format format>
WorkspaceBundle ConvBiasForwardImpl::AlgoMatmul8x8x32::get_bundle(
        const SizeArgs& args) const {
    size_t src_unroll_part, filter_reshape_part;
    size_t relayout_src_part = 0, relayout_filter_part = 0,
           relayout_dst_part = 0;
    auto&& fm = args.filter_meta;
    size_t n, ih, iw, oh, ow, fh, fw, ic, oc;
    n = args.dst_layout->shape[0];
    fh = fm.spatial[0];
    fw = fm.spatial[1];
    if (format == Param::Format::NHWC) {
        oh = args.dst_layout->shape[1];
        ow = args.dst_layout->shape[2];
        ic = args.src_layout->shape[3];
        oc = args.dst_layout->shape[3];
    } else {
        // NCHW4
        ic = args.src_layout->shape[1] * 4;
        ih = args.src_layout->shape[2];
        iw = args.src_layout->shape[3];
        oc = args.dst_layout->shape[1] * 4;
        oh = args.dst_layout->shape[2];
        ow = args.dst_layout->shape[3];

        relayout_src_part = n * ic * ih * iw * sizeof(int8_t);
        relayout_filter_part = ic * oc * fh * fw * sizeof(int8_t);
        relayout_dst_part = n * oc * oh * ow * sizeof(int32_t);
    }
    // short for ``leading dimension''
    size_t ld = (fh * fw * ic + 3) & ~3;
    if (need_src_unroll(args)) {
        src_unroll_part = n * oh * ow * ld * sizeof(int8_t);
    } else {
        src_unroll_part = 0;
    }
    if (need_filter_reshape(args)) {
        filter_reshape_part = oc * ld * sizeof(int8_t);
    } else {
        filter_reshape_part = 0;
    }

    SmallVector<size_t> sizes = {src_unroll_part, filter_reshape_part,
                                 relayout_src_part, relayout_filter_part,
                                 relayout_dst_part};

    auto dst_layout = *args.dst_layout;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    return WorkspaceBundle(nullptr, sizes);
}

size_t ConvBiasForwardImpl::AlgoMatmul8x8x32::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.filter_meta.format == Param::Format::NHWC) {
        auto bundle = get_bundle<Param::Format::NHWC>(args);
        return bundle.total_size_in_bytes();
    } else {
        // NCHW4
        auto bundle = get_bundle<Param::Format::NCHW4>(args);
        return bundle.total_size_in_bytes();
    }
}

template <param::ConvBias::Format format>
void ConvBiasForwardImpl::AlgoMatmul8x8x32::exec_internal(
        const ExecArgs& args) const {
    auto stream = args.handle->stream();
    auto cublas_handle = args.handle->cublas_handle();
    auto alpha = args.handle->one_device_i32();
    auto beta = args.handle->zero_device_i32();
    auto&& fm = args.filter_meta;
    auto bundle = get_bundle<format>(args);
    bundle.set(args.workspace.raw_ptr);

    TensorND src_tensor, dst_tensor, filter_tensor;
    if (format == Param::Format::NHWC) {
        src_tensor = *args.src_tensor;
        dst_tensor = *args.dst_tensor;
        filter_tensor = *args.filter_tensor;
    } else {
        // NCHW4
        auto to_nhwc = [](const TensorLayout& layout,
                          void* raw_ptr) -> TensorND {
            return {raw_ptr,
                    {{layout[0], layout[2], layout[3], layout[1] * 4},
                     layout.dtype}};
        };
        src_tensor = to_nhwc(*args.src_layout, bundle.get(2));
        filter_tensor = to_nhwc(args.filter_tensor->layout, bundle.get(3));
        dst_tensor = to_nhwc(*args.dst_layout, bundle.get(4));

        auto relayout = [&](const TensorND& src, void* dst_ptr) {
            auto N = src.layout[0], C = src.layout[1] * 4, H = src.layout[2],
                 W = src.layout[3];
            args.handle->relayout_opr()->exec(
                    {src.raw_ptr,
                     TensorLayout{{N, H, W, C / 4, 4},
                                  {
                                      src.layout.stride[0],
                                      src.layout.stride[2],
                                      src.layout.stride[3],
                                      src.layout.stride[1],
                                      src.layout.stride[4]
                                  },
                                  src.layout.dtype}},
                    {dst_ptr,
                     TensorLayout{{N, H, W, C / 4, 4}, src.layout.dtype}});
        };
        relayout(*args.src_tensor, src_tensor.raw_ptr);
        relayout(*args.filter_tensor, filter_tensor.raw_ptr);
    }

    size_t N, IH, IW, IC;
    N = src_tensor.layout.shape[0];
    IH = src_tensor.layout.shape[1];
    IW = src_tensor.layout.shape[2];
    IC = src_tensor.layout.shape[3];

    auto IWS = src_tensor.layout.stride[2];
    auto FH = fm.spatial[0], FW = fm.spatial[1];
    auto OH = dst_tensor.layout.shape[1], OW = dst_tensor.layout.shape[2],
         OC = dst_tensor.layout.shape[3];
    auto OWS = dst_tensor.layout.stride[2];
    auto PH = fm.padding[0], PW = fm.padding[1];
    auto SH = fm.stride[0], SW = fm.stride[1];
    auto DH = fm.dilation[0], DW = fm.dilation[1];
    auto LD = (FH * FW * IC + 3) & ~3;

    int8_t *inp0 = nullptr, *inp1 = nullptr;
    ptrdiff_t inp0_stride = 0, inp1_stride = 0;

    if (need_src_unroll(args)) {
        inp0 = static_cast<int8_t*>(bundle.get(0));
        inp0_stride = LD;
        im2col_nhwc_int8(src_tensor.compatible_ptr<dt_int8>(), inp0, N, IH, IW,
                         IC, IWS, OH, OW, OC, OWS, FH, FW, PH, PW, SH, SW, DH,
                         DW, LD, fm.should_flip, stream);
    } else {
        inp0 = src_tensor.compatible_ptr<dt_int8>();
        inp0_stride = IWS;
    }
    if (need_filter_reshape(args)) {
        // copy (OC, FH*FW*IC) to (OC, FH*FW*IC) with stride=LD
        inp1 = static_cast<int8_t*>(bundle.get(1));
        cuda_check(cudaMemcpy2DAsync(
                inp1, LD * sizeof(int8_t), filter_tensor.raw_ptr,
                FH * FW * IC * sizeof(int8_t), FH * FW * IC * sizeof(int8_t),
                OC, cudaMemcpyDeviceToDevice, stream));
        inp1_stride = LD;
    } else {
        inp1 = filter_tensor.compatible_ptr<dt_int8>();
        inp1_stride = FH * FW * IC;
    }
    cublas_check(cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC,
                              N * OH * OW, FH * FW * IC, alpha, inp1, CUDA_R_8I,
                              inp1_stride, inp0, CUDA_R_8I, inp0_stride, beta,
                              dst_tensor.compatible_ptr<dt_int32>(), CUDA_R_32I,
                              OWS, CUDA_R_32I, CUBLAS_GEMM_DFALT));

    if (format == Param::Format::NCHW4) {
        args.handle->relayout_opr()->exec(
                {dst_tensor.compatible_ptr<int32_t>(),
                 TensorLayout{{N, OC / 4, OH, OW, 4},
                              {static_cast<ptrdiff_t>(OH * OW * OC), 4,
                               static_cast<ptrdiff_t>(OC * OW),
                               static_cast<ptrdiff_t>(OC), 1},
                              dst_tensor.layout.dtype}},
                *args.dst_tensor);
    }
}

void ConvBiasForwardImpl::AlgoMatmul8x8x32::exec(const ExecArgs& args) const {
    ExecArgs conv_args = args;
    auto conv_dst_tensor = *args.dst_tensor;
    if (args.filter_meta.format == Param::Format::NHWC) {
        auto bundle = get_bundle<Param::Format::NHWC>(args);
        bundle.set(args.workspace.raw_ptr);
        if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
            conv_dst_tensor.raw_ptr = bundle.get(bundle.nr_workspace() - 1);
            conv_dst_tensor.layout.dtype = DType();
            args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                                args.filter_layout->dtype,
                                                conv_dst_tensor.layout.dtype);
        }
        conv_args.dst_tensor = &conv_dst_tensor;
        conv_args.dst_layout = &conv_dst_tensor.layout;
    } else {
        auto bundle = get_bundle<Param::Format::NCHW4>(args);
        bundle.set(args.workspace.raw_ptr);
        if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
            conv_dst_tensor.raw_ptr = bundle.get(bundle.nr_workspace() - 1);
            conv_dst_tensor.layout.dtype = DType();
            args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                                args.filter_layout->dtype,
                                                conv_dst_tensor.layout.dtype);
        }
        conv_args.dst_tensor = &conv_dst_tensor;
        conv_args.dst_layout = &conv_dst_tensor.layout;
    }

    if (args.filter_meta.format == Param::Format::NHWC) {
        exec_internal<Param::Format::NHWC>(conv_args);
    } else {
        // NCHW4
        exec_internal<Param::Format::NCHW4>(conv_args);
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

bool ConvBiasForwardImpl::AlgoMatmul8x8x32::need_filter_reshape(
        const SizeArgs& args) const {
    // cublasGemmEx requires the stride of the filter matrix to be multiples
    // of 4.
    auto&& fm = args.filter_meta;
    size_t ic;
    if (args.filter_meta.format == Param::Format::NHWC) {
        ic = args.src_layout->shape[3];
    } else {
        // NCHW4
        ic = args.src_layout->shape[1] * 4;
    }
    return !(ic * fm.spatial[0] * fm.spatial[1] % 4 == 0);
}

bool ConvBiasForwardImpl::AlgoMatmul8x8x32::need_src_unroll(
        const SizeArgs& args) const {
    // cublasGemmEx requires the stride of the unrolled src to be multiples
    // of 4.
    size_t stride;
    if (args.filter_meta.format == Param::Format::NHWC) {
        stride = args.src_layout->stride[2];
    } else {
        // NCHW4
        stride = args.src_layout->shape[1] * 4;
    }

    auto&& fm = args.filter_meta;
    return !(fm.spatial[0] == 1 && fm.spatial[1] == 1 && fm.stride[0] == 1 &&
             fm.stride[1] == 1 && fm.padding[0] == 0 && fm.padding[1] == 0 &&
             stride % 4 == 0);
}
// vim: syntax=cpp.doxygen
