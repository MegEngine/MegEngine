/**
 * \file dnn/src/cuda/conv_bias/sass_implicit_gemm_int4_nchw64_imma.cpp
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
#include "src/cuda/conv_bias/sass_helper.cuh"
#include "src/cuda/sass_loader.h"
#include "src/cuda/utils.h"
#include "src/common/conv_bias.h"

using namespace megdnn;
using namespace cuda;
using namespace sass;

namespace {
#if !MEGDNN_TEGRA_X1
// all stride are in bytes
void compute_conv2d_offset(size_t fh, size_t fw, size_t ics, size_t ihs,
                           Conv2dConstantOffset& constant_offset) {
    constexpr int interleaved = 64;
    constexpr int size_bits = 4;
    constexpr int threablock_k = 128;
    constexpr int inc_step = threablock_k / interleaved;
    size_t i = 0;
    int* s32 = reinterpret_cast<int*>(&(constant_offset.c_offset[0]));
    for (; i < inc_step; i++) {
        int c = i / (fh * fw);
        int khkw = i % (fh * fw);
        int kh = khkw / fw;
        int kw = khkw % fw;
        s32[2 * i] = c * ics + kh * ihs + kw * interleaved * size_bits / 8;
        int8_t* s8 = reinterpret_cast<int8_t*>(&(s32[2 * i + 1]));
        s8[0] = kh;
        s8[1] = kw;
        s8[2] = -kh;
        s8[3] = -kw;
    }
    for (; i < (inc_step + fh * fw * inc_step); i++) {
        int c = i / (fh * fw);
        int khkw = i % (fh * fw);
        int kh = khkw / fw;
        int kw = khkw % fw;
        s32[2 * i] = c * ics + kh * ihs + kw * interleaved * size_bits / 8;
        int8_t* s8 = reinterpret_cast<int8_t*>(&(s32[2 * i + 1]));
        s8[0] = kh;
        s8[1] = kw;
        s8[2] = -kh;
        s8[3] = -kw;
        int i_ = i - inc_step;
        c = i_ / (fh * fw);
        khkw = i_ % (fh * fw);
        kh = khkw / fw;
        kw = khkw % fw;
        s32[2 * i] -= c * ics + kh * ihs + kw * interleaved * size_bits / 8;
    }
}
#endif
};  // namespace

std::string ConvBiasForwardImpl::AlgoSASSInt4NCHW64IMMAImplicitGemm::kernel_key(
        const SizeArgs& args) const {
    std::string kernel_key;
    using NonlineMode = Param::NonlineMode;
    auto&& param = args.opr->param();
    if (args.z_layout->ndim > 0) {
        kernel_key =
                ssprintf("%s_conv_bias_int4_fuse_z_imma8832_ldg16_%ux%u",
                         current_device_arch_name(), m_tile_nhw, m_tile_oc);
    } else {
        kernel_key =
                ssprintf("%s_conv_bias_int4_imma8832_ldg16_%ux%u",
                         current_device_arch_name(), m_tile_nhw, m_tile_oc);
    }
    if (param.nonlineMode == NonlineMode::H_SWISH) {
        kernel_key += "_hswish";
    } else {
        megdnn_assert(param.nonlineMode == NonlineMode::RELU ||
                      param.nonlineMode == NonlineMode::IDENTITY);
        kernel_key += "_relu";
    }
    return kernel_key;
}

bool ConvBiasForwardImpl::AlgoSASSInt4NCHW64IMMAImplicitGemm::is_available(
        const SizeArgs& args) const {
    if (args.bias_layout->ndim <= 0)
        return false;
    using Param = param::ConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    bool available = true;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    if (!check_bias_share_in_channel(*(args.bias_layout), param.format))
        return false;
    if (param.format != Format::NCHW64)
        return false;
    UNPACK_CONV_BIAS_NCHW64_PARAM(*(args.src_layout), fm, *(args.dst_layout),
                                  param);
    // TODO support group conv
    available &= param.sparse == Sparse::DENSE;
    // mode must be cross correlation
    available &= param.mode == Mode::CROSS_CORRELATION;
    // check data type
    auto src_dtype = args.src_layout->dtype,
         filter_dtype = args.filter_layout->dtype,
         bias_dtype = args.bias_layout->dtype,
         dst_dtype = args.dst_layout->dtype;
    available &= (src_dtype.enumv() == DTypeEnum::QuantizedS4 &&
                  filter_dtype.enumv() == DTypeEnum::QuantizedS4 &&
                  bias_dtype.enumv() == DTypeEnum::QuantizedS32 &&
                  dst_dtype.enumv() == DTypeEnum::QuantizedS4);
    // TODO: support dialtion
    available &= dh == 1 && dw == 1;
    // ensure precomputed offsets are positive integers
    available &= hi >= fh && wi >= fw;
    // only support sm_75 or later, platform should have tensorcore int8
    // support
    available &= is_compute_capability_required(7, 5);
    // param buffer size is 4K, use 3K to store precomputed offset, fh * fw <=
    // (3*1024/4/2/2) - 1
    available &= fh * fw <= 191;
    return available;
}

size_t
ConvBiasForwardImpl::AlgoSASSInt4NCHW64IMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.preprocessed_filter == nullptr) {
        return args.filter_layout->span().dist_byte() +
               args.bias_layout->span().dist_byte();
    }
    return 0_z;
}

void ConvBiasForwardImpl::AlgoSASSInt4NCHW64IMMAImplicitGemm::exec(
        const ExecArgs& args) const {
#if MEGDNN_TEGRA_X1
    megdnn_throw("sass kernel is disabled at compile time for TX1");
#else
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    UNPACK_CONV_BIAS_NCHW64_PARAM(*(args.src_layout), fm, *(args.dst_layout),
                                  param);
    auto&& stream = cuda_stream(args.opr->handle());
    constexpr int interleaved = 64;

    void* bias_ptr = nullptr;
    void* filter_ptr = nullptr;
    if (args.preprocessed_filter) {
        megdnn_assert(args.preprocessed_filter->tensors.size() == 2);
        filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
        bias_ptr = args.preprocessed_filter->tensors[1].raw_ptr;
    } else {
        // reorder filter and bias
        filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
        bias_ptr =
                reinterpret_cast<void*>(args.workspace.raw_ptr +
                                        args.filter_layout->span().dist_byte());
        if (args.z_layout->ndim > 0) {
            reorder_imma_filter_bias<4, 64>(
                    reinterpret_cast<int8_t*>(filter_ptr),
                    reinterpret_cast<int32_t*>(bias_ptr),
                    reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr),
                    args.bias_tensor->compatible_ptr<int32_t>(), co, ci, fh, fw,
                    stream);
        } else {
            reorder_imma_filter_bias<4, 64, true>(
                    reinterpret_cast<int8_t*>(filter_ptr),
                    reinterpret_cast<int32_t*>(bias_ptr),
                    reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr),
                    args.bias_tensor->compatible_ptr<int32_t>(), co, ci, fh, fw,
                    stream);
        }
    }

    uint32_t u32_n = n, u32_ci = ci, u32_hi = hi, u32_wi = wi, u32_fh = fh,
             u32_fw = fw, u32_sh = sh, u32_sw = sw, u32_ph = ph, u32_pw = pw,
             u32_co = co, u32_ho = ho, u32_wo = wo;
    Conv2dInt4Param kern_param(u32_n, u32_ci, u32_hi, u32_wi, u32_fh, u32_fw,
                               u32_sh, u32_sw, u32_ph, u32_pw, u32_co, u32_ho,
                               u32_wo, interleaved);

    Conv2dConstantOffset kern_coffset;
    compute_conv2d_offset(fh, fw, kern_param.ics, kern_param.ihs, kern_coffset);
    // The starting address of Turing param buffer is c[0x0][0x160]
    kern_coffset.c_offset_param.begin = param_buffer_start_address();
    kern_coffset.c_offset_param.size = 16 * (1 + fh * fw);
    kern_coffset.c_offset_param.max = 16 * fh * fw;
    kern_coffset.c_offset_param.rewind = 16 * (1 - fh * fw);

    auto kern_key = kernel_key(args);
    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS4>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS4>().scale,
          bias_scale =
                  args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;
    float alpha = src_scale * filter_scale / dst_scale,
          beta = bias_scale / dst_scale;
    float inv_dst_scale = 1.f / dst_scale;

    unsigned int tx = m_threads, ty = 1;
    unsigned int gridx = div_ceil<unsigned int>(
            static_cast<unsigned int>(n * ho * wo), m_tile_nhw);
    unsigned int gridy =
            div_ceil<unsigned int>(static_cast<unsigned int>(co), m_tile_oc);
    void* src_ptr = const_cast<void*>(args.src_tensor->raw_ptr);
    void* dst_ptr = const_cast<void*>(args.dst_tensor->raw_ptr);

    using NonlineMode = Param::NonlineMode;
    auto&& kernel = SASSKernelLoader::instance().get_kernel(kern_key, kern_key);
    if (args.z_layout->ndim > 0) {
        void* z_ptr = const_cast<void*>(args.z_tensor->raw_ptr);
        float z_scale = args.z_layout->dtype.param<dtype::QuantizedS4>().scale;
        float gamma = z_scale / dst_scale;
        std::vector<void*> params = {&src_ptr, &filter_ptr, &bias_ptr, &z_ptr,
                                     &dst_ptr, &alpha,      &beta,     &gamma};
        kern_coffset.c_offset_param.begin +=
                sizeof(src_ptr) + sizeof(filter_ptr) + sizeof(bias_ptr) +
                sizeof(z_ptr) + sizeof(dst_ptr) + sizeof(alpha) + sizeof(beta) +
                sizeof(gamma);

        uint32_t relu = param.nonlineMode == NonlineMode::RELU ? 1 : 0;
        if (param.nonlineMode == NonlineMode::H_SWISH) {
            params.push_back(&dst_scale);
            params.push_back(&inv_dst_scale);
            kern_coffset.c_offset_param.begin +=
                    sizeof(dst_scale) + sizeof(inv_dst_scale);
        } else {
            params.push_back(&relu);
            kern_coffset.c_offset_param.begin += sizeof(relu);
        }
        params.push_back(&kern_param);
        kern_coffset.c_offset_param.begin += sizeof(kern_param);
        kern_coffset.c_offset_param.begin +=
                sizeof(kern_coffset.c_offset_param);
        kern_coffset.c_offset_param.max += kern_coffset.c_offset_param.begin;
        params.push_back(&kern_coffset);
        cucheck(cuLaunchKernel(kernel, gridx, gridy, 1, tx, ty, 1, 0, stream,
                               params.data(), 0));
    } else {
        std::vector<void*> params = {&src_ptr, &filter_ptr, &bias_ptr,
                                     &dst_ptr, &alpha,      &beta};

        kern_coffset.c_offset_param.begin +=
                sizeof(src_ptr) + sizeof(filter_ptr) + sizeof(bias_ptr) +
                sizeof(dst_ptr) + sizeof(alpha) + sizeof(beta);

        uint32_t relu = param.nonlineMode == NonlineMode::RELU ? 1 : 0;
        if (param.nonlineMode == NonlineMode::H_SWISH) {
            params.push_back(&dst_scale);
            params.push_back(&inv_dst_scale);
            kern_coffset.c_offset_param.begin +=
                    sizeof(dst_scale) + sizeof(inv_dst_scale);
        } else {
            params.push_back(&relu);
            kern_coffset.c_offset_param.begin += sizeof(relu);
        }
        params.push_back(&kern_param);
        kern_coffset.c_offset_param.begin += sizeof(kern_param);
        kern_coffset.c_offset_param.begin +=
                sizeof(kern_coffset.c_offset_param);
        kern_coffset.c_offset_param.max += kern_coffset.c_offset_param.begin;
        params.push_back(&kern_coffset);
        cucheck(cuLaunchKernel(kernel, gridx, gridy, 1, tx, ty, 1, 0, stream,
                               params.data(), 0));
    }
    after_kernel_launch();
#endif
}

size_t ConvBiasForwardImpl::AlgoSASSInt4NCHW64IMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0_z;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::
        AlgoSASSInt4NCHW64IMMAImplicitGemm::deduce_preprocessed_filter_layout(
                const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous(),
            args.bias_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoSASSInt4NCHW64IMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    UNPACK_CONV_BIAS_NCHW64_PARAM(*(args.src_layout), fm, *(args.dst_layout),
                                  param);
    auto&& stream = cuda_stream(args.opr->handle());
    reorder_imma_filter_bias<4, 64>(
            reinterpret_cast<int8_t*>(
                    args.preprocessed_filter->tensors[0].raw_ptr),
            args.preprocessed_filter->tensors[1].compatible_ptr<int32_t>(),
            reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr),
            args.bias_tensor->compatible_ptr<int32_t>(), co, ci, fh, fw,
            stream);
}

// vim: syntax=cpp.doxygen
