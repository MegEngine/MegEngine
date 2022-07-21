/**
 * \file dnn/src/cuda/conv_bias/ptx_implicit_gemm_uint4_int4_nchw64_imma.cpp
 */

#include "./algo.h"
#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/ptx_helper.cuh"
#include "src/cuda/conv_bias/reduce_filter.cuh"
#include "src/cuda/ptx/uint4_int4/kern.cuh"
#include "src/cuda/ptx_loader.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace ptx;

namespace {
// all stride are in bytes
void compute_conv2d_offset(
        size_t fh, size_t fw, size_t ics, size_t ihs,
        Conv2dConstantOffset& constant_offset) {
    constexpr int interleaved = 64;
    constexpr int size_bits = 4;
    constexpr int threablock_k = 128;
    constexpr int inc_step = threablock_k / interleaved;
    size_t i = 0;
    int* s32 = &(constant_offset.c_offset[0]);
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
};  // namespace

std::string ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::kernel_key(
        const SizeArgs& args) const {
    std::string kernel_key;
    using NonlineMode = Param::NonlineMode;
    auto&& param = args.opr->param();
    if (args.z_layout->ndim > 0) {
        kernel_key = ssprintf(
                "%s_conv_bias_uint4_int4_fuse_z_imma8832_ldg16_%ux%u",
                current_device_arch_name(), m_tile_nhw, m_tile_oc);
    } else {
        kernel_key = ssprintf(
                "%s_conv_bias_uint4_int4_imma8832_ldg16_%ux%u",
                current_device_arch_name(), m_tile_nhw, m_tile_oc);
    }
    megdnn_assert(
            param.nonlineMode == NonlineMode::RELU ||
            param.nonlineMode == NonlineMode::IDENTITY);
    kernel_key += "_relu";
    return kernel_key;
}

bool ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::is_available(
        const SizeArgs& args) const {
    if (!args.src_layout->is_contiguous() || !args.dst_layout->is_contiguous()) {
        return false;
    }
    if (args.bias_layout->ndim <= 0)
        return false;
    using Param = param::ConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    using NonlineMode = Param::NonlineMode;
    bool available = true;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    if (!check_bias_share_in_channel(*(args.bias_layout), param.format))
        return false;
    if (param.format != Format::NCHW64)
        return false;
    UNPACK_CONV_BIAS_NCHW64_PARAM(*(args.src_layout), fm, *(args.dst_layout), param);
    // TODO support group conv
    available &= param.sparse == Sparse::DENSE;
    // mode must be cross correlation
    available &= param.mode == Mode::CROSS_CORRELATION;
    // nonlineMode must be RELU or IDENTITY
    available &=
            (param.nonlineMode == NonlineMode::RELU ||
             param.nonlineMode == NonlineMode::IDENTITY);
    // check data type
    auto src_dtype = args.src_layout->dtype, filter_dtype = args.filter_layout->dtype,
         bias_dtype = args.bias_layout->dtype, dst_dtype = args.dst_layout->dtype;
    available &=
            (src_dtype.enumv() == DTypeEnum::Quantized4Asymm &&
             filter_dtype.enumv() == DTypeEnum::QuantizedS4 &&
             bias_dtype.enumv() == DTypeEnum::QuantizedS32 &&
             dst_dtype.enumv() == DTypeEnum::Quantized4Asymm);
    // TODO: support dialtion
    available &= dh == 1 && dw == 1;
    // ensure precomputed offsets are positive integers
    available &= hi >= fh && wi >= fw;
    // only support sm_86 or later, platform should have tensorcore int4
    // support
    available &=
            (is_compute_capability_equalto(8, 0) ||
             is_compute_capability_equalto(8, 6));
    // param buffer size is 4K, use 3K to store precomputed offset
    size_t kMaxFilterPixels = CONSTANT_BUFFER_SIZE / (2 * 128 / 64) - 1;
    available &= fh * fw <= kMaxFilterPixels;
    return available;
}

size_t ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::
        get_workspace_in_bytes(const SizeArgs& args) const {
    if (args.preprocessed_filter == nullptr) {
        size_t OC = args.filter_layout->operator[](0),
               IC = args.filter_layout->operator[](1) * 64,
               FH = args.filter_layout->operator[](2),
               FW = args.filter_layout->operator[](3);
        size_t ws_size_reduce_filter = OC * sizeof(int32_t);
        // for reduce filter
        {
            size_t A = OC, B = IC * FH * FW / 8, C = 1;
            ws_size_reduce_filter += do_dispatch_reduce_workspace_in_bytes(A, B, C);
        }
        return args.filter_layout->span().dist_byte() +
               args.bias_layout->span().dist_byte() + ws_size_reduce_filter;
    }
    return 0_z;
}

void ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::exec(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    UNPACK_CONV_BIAS_NCHW64_PARAM(*(args.src_layout), fm, *(args.dst_layout), param);
    auto&& stream = cuda_stream(args.opr->handle());
    constexpr int interleaved = 64;

    void* bias_ptr = nullptr;
    void* filter_ptr = nullptr;
    if (args.preprocessed_filter) {
        megdnn_assert(args.preprocessed_filter->tensors.size() == 2);
        filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr();
        bias_ptr = args.preprocessed_filter->tensors[1].raw_ptr();
    } else {
        // reorder filter and bias
        filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
        bias_ptr = reinterpret_cast<void*>(
                args.workspace.raw_ptr + args.filter_layout->span().dist_byte());
        void* reduce_filter_ptr = reinterpret_cast<void*>(
                args.workspace.raw_ptr + args.filter_layout->span().dist_byte() +
                args.bias_layout->span().dist_byte());
        reorder_filter_bias(args, reduce_filter_ptr, filter_ptr, bias_ptr);
    }

    uint32_t u32_n = n, u32_ci = ci, u32_hi = hi, u32_wi = wi, u32_fh = fh, u32_fw = fw,
             u32_sh = sh, u32_sw = sw, u32_ph = ph, u32_pw = pw, u32_co = co,
             u32_ho = ho, u32_wo = wo;
    Conv2dInt4Param kern_param(
            u32_n, u32_ci, u32_hi, u32_wi, u32_fh, u32_fw, u32_sh, u32_sw, u32_ph,
            u32_pw, u32_co, u32_ho, u32_wo, interleaved);

    Conv2dConstantOffset kern_coffset;
    compute_conv2d_offset(fh, fw, kern_param.ics, kern_param.ihs, kern_coffset);
    // begin is not need
    kern_coffset.c_offset_param.begin = param_buffer_start_address();
    kern_coffset.c_offset_param.size = 4 * (1 + fh * fw);
    kern_coffset.c_offset_param.max = 4 * fh * fw;
    kern_coffset.c_offset_param.rewind = 4 * (1 - fh * fw);

    float src_scale = args.src_layout->dtype.param<dtype::Quantized4Asymm>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::Quantized4Asymm>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS4>().scale;

    uint32_t src_zero_point =
            (uint32_t)(args.src_layout->dtype.param<dtype::Quantized4Asymm>()
                               .zero_point);
    uint32_t pk_src_zero_point = 0;
    for (int i = 0; i < 8; i++) {
        pk_src_zero_point <<= 4;
        pk_src_zero_point |= (src_zero_point & 0xF);
    }
    float dst_zero_point =
            (float)(args.dst_layout->dtype.param<dtype::Quantized4Asymm>().zero_point);
    float alpha = src_scale * filter_scale / dst_scale, beta = 1.f;

    unsigned int tx = m_threads, ty = 1;
    unsigned int gridx =
            div_ceil<unsigned int>(static_cast<unsigned int>(n * ho * wo), m_tile_nhw);
    unsigned int gridy =
            div_ceil<unsigned int>(static_cast<unsigned int>(co), m_tile_oc);
    void* src_ptr = const_cast<void*>(args.src_tensor->raw_ptr());
    void* dst_ptr = const_cast<void*>(args.dst_tensor->raw_ptr());

    using NonlineMode = Param::NonlineMode;

    auto kern_key = kernel_key(args);
    auto&& kernel = PTXKernelLoader::instance().get_kernel(kern_key);
    if (args.z_layout->ndim > 0) {
        void* z_ptr = const_cast<void*>(args.z_tensor->raw_ptr());
        auto z_param = args.z_layout->dtype.param<dtype::Quantized4Asymm>();
        int32_t z_zero_point = (int32_t)z_param.zero_point;
        float z_scale = z_param.scale;

        float gamma = z_scale / dst_scale;
        std::vector<void*> params = {&src_ptr, &filter_ptr, &bias_ptr, &z_ptr,
                                     &dst_ptr, &alpha,      &beta,     &gamma};
        kern_coffset.c_offset_param.begin += sizeof(src_ptr) + sizeof(filter_ptr) +
                                             sizeof(bias_ptr) + sizeof(z_ptr) +
                                             sizeof(dst_ptr) + sizeof(alpha) +
                                             sizeof(beta) + sizeof(gamma);

        kern_coffset.c_offset_param.begin += sizeof(pk_src_zero_point);
        params.push_back(&pk_src_zero_point);
        kern_coffset.c_offset_param.begin += sizeof(z_zero_point);
        params.push_back(&z_zero_point);
        kern_coffset.c_offset_param.begin += sizeof(dst_zero_point);
        params.push_back(&dst_zero_point);

        uint32_t relu = param.nonlineMode == NonlineMode::RELU ? 1 : 0;
        params.push_back(&relu);
        kern_coffset.c_offset_param.begin += sizeof(relu);
        params.push_back(&kern_param);
        kern_coffset.c_offset_param.begin += sizeof(kern_param);
        kern_coffset.c_offset_param.begin += sizeof(kern_coffset.c_offset_param);
        params.push_back(&kern_coffset);

        dim3 grid(gridx, gridy, 1);
        dim3 block(tx, ty, 1);

        kernel(grid, block, stream, params.data());
    } else {
        std::vector<void*> params = {&src_ptr, &filter_ptr, &bias_ptr,
                                     &dst_ptr, &alpha,      &beta};
        kern_coffset.c_offset_param.begin += sizeof(src_ptr) + sizeof(filter_ptr) +
                                             sizeof(bias_ptr) + sizeof(dst_ptr) +
                                             sizeof(alpha) + sizeof(beta);

        kern_coffset.c_offset_param.begin += sizeof(pk_src_zero_point);
        params.push_back(&pk_src_zero_point);
        kern_coffset.c_offset_param.begin += sizeof(dst_zero_point);
        params.push_back(&dst_zero_point);

        uint32_t relu = param.nonlineMode == NonlineMode::RELU ? 1 : 0;
        params.push_back(&relu);
        kern_coffset.c_offset_param.begin += sizeof(relu);
        params.push_back(&kern_param);
        kern_coffset.c_offset_param.begin += sizeof(kern_param);
        kern_coffset.c_offset_param.begin += sizeof(kern_coffset.c_offset_param);
        params.push_back(&kern_coffset);

        dim3 grid(gridx, gridy, 1);
        dim3 block(tx, ty, 1);

        kernel(grid, block, stream, params.data());
    }
    after_kernel_launch();
}

size_t ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    size_t OC = args.filter_layout->operator[](0),
           IC = args.filter_layout->operator[](1) * 64,
           FH = args.filter_layout->operator[](2),
           FW = args.filter_layout->operator[](3);
    size_t ws_size_reduce_filter = OC * sizeof(int32_t);
    // for reduce filter
    {
        size_t A = OC, B = IC * FH * FW / 8, C = 1;
        ws_size_reduce_filter += do_dispatch_reduce_workspace_in_bytes(A, B, C);
    }
    return ws_size_reduce_filter;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous(),
            args.bias_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::reorder_filter_bias(
        const ExecArgs& args, void* reduce_filter, void* reordered_filter,
        void* reordered_bias) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    UNPACK_CONV_BIAS_NCHW64_PARAM(*(args.src_layout), fm, *(args.dst_layout), param);
    auto&& stream = cuda_stream(args.opr->handle());

    float src_scale = args.src_layout->dtype.param<dtype::Quantized4Asymm>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS4>().scale,
          bias_scale = args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::Quantized4Asymm>().scale;

    float scaled_src_zero_point =
            args.src_layout->dtype.param<dtype::Quantized4Asymm>().zero_point *
            src_scale * filter_scale / dst_scale;
    // NCHW64 reduce CHW64
    do_dispatch_reduce_with_scale_filter_4bit<true>(
            reinterpret_cast<uint8_t*>(args.filter_tensor->raw_ptr()), 1, co,
            ci * fh * fw / 8, static_cast<int32_t*>(reduce_filter), stream);

    reorder_imma_filter_bias_fusion_zero_point<4, 64>(
            reinterpret_cast<int8_t*>(reordered_filter),
            reinterpret_cast<float*>(reordered_bias),
            reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr()),
            args.bias_tensor->compatible_ptr<int32_t>(), bias_scale / dst_scale,
            static_cast<int32_t*>(reduce_filter), scaled_src_zero_point, co, ci, fh, fw,
            stream);
}

void ConvBiasForwardImpl::AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    reorder_filter_bias(
            args, args.workspace.raw_ptr,
            args.preprocessed_filter->tensors[0].raw_ptr(),
            args.preprocessed_filter->tensors[1].raw_ptr());
}
// vim: syntax=cpp.doxygen
