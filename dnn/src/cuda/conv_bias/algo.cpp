/**
 * \file dnn/src/cuda/conv_bias/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

ConvBiasForwardImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);
    non_cudnn_algos.push_back(&chanwise_small);

    non_cudnn_algos.push_back(&inplace_matmul);
    non_cudnn_algos.push_back(&matmul);
    non_cudnn_algos.push_back(&matmul8x8x32);
    non_cudnn_algos.push_back(&batched_matmul);
    non_cudnn_algos.push_back(&a1x1);

    fill_cudnn_algos();
    for (auto&& algo : cudnn_conv_bias_activations) {
        all_algos.push_back(&algo);
    }

    //! add conv+nonlinear algos
    std::vector<AlgoBase*> conv_algos;
    conv_algos.push_back(&chanwise);
    conv_algos.push_back(&chanwise_small);
    conv_algos.push_back(&chanwise8x8x32);
    for (auto&& algo : cudnn_convs) {
        conv_algos.push_back(&algo);
    }
    conv_algos.push_back(&inplace_matmul);
    conv_algos.push_back(&matmul);
    conv_algos.push_back(&matmul8x8x32);
    conv_algos.push_back(&batched_matmul);
    conv_algos.push_back(&a1x1);

    conv_algos.reserve(conv_algos.size() * 2);
    //! add gconv algos by AlgoGroupConvGeneral
    size_t algo_size = conv_algos.size();
    for (size_t i = 3; i < algo_size; ++i) {
        gconv_refhold.emplace_back(new AlgoGroupConvGeneral(conv_algos[i]));
        algo2gconv[conv_algos[i]] = gconv_refhold.back().get();
        conv_algos.push_back(gconv_refhold.back().get());
    }

    for (auto&& algo : conv_algos) {
        all_algos.push_back(algo);
    }
    non_cudnn_algos.push_back(all_algos.rbegin()[4]);  // group inplace_matmul
    non_cudnn_algos.push_back(all_algos.rbegin()[3]);  // group matmul
    non_cudnn_algos.push_back(all_algos.rbegin()[2]);  // group matmul_8x8x32
    non_cudnn_algos.push_back(all_algos.rbegin()[1]);  // group batched_matmul
    non_cudnn_algos.push_back(all_algos.rbegin()[0]);  // group 1x1

    algo_size = all_algos.size();
    for (size_t i = 0; i < algo_size; ++i) {
        bfloat16_refhold.emplace_back(new AlgoBFloat16(all_algos[i]));
        all_algos.push_back(bfloat16_refhold.back().get());
        bfloat16_algos.push_back(bfloat16_refhold.back().get());
    }

    size_t all_algo_size = all_algos.size();
#if CUDA_VERSION >= 10000
    fill_imma_algos();
    all_algos.push_back(&wmma_quint4x4x32);
    for (auto&& algo : int8_nchw4_imma) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : int8_chwn4_imma) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : int8_chwn4_imma_reorder_filter) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : int8_chwn4_imma_unroll_width) {
        all_algos.push_back(&algo);
    }
#if CUDA_VERSION >= 10020
    for (auto&& algo : int8_nchw32_imma) {
        all_algos.push_back(&algo);
    }
#endif
#endif
    fill_dp4a_algos();
    for (auto&& algo : int8_nchw4_dotprod) {
        all_algos.push_back(&algo);
    }
    all_algos.push_back(&int8_chwn4_dotprod);
    for (size_t i = all_algo_size; i < all_algos.size(); ++i) {
        non_cudnn_algos.push_back(all_algos[i]);
    }

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

ConvBiasForwardImpl::AlgoPack ConvBiasForwardImpl::sm_algo_pack;

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvBiasForwardImpl)

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvBiasForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter)
        : SizeArgs(o, src, filter, o->check_layout_fwd(src, filter, dst), bias,
                   z, dst, preprocessed_filter) {}

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvBiasForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const CanonizedFilterMeta& filter_meta,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst, const PreprocessedFilter* preprocessed_filter)
        : BiasForwardSizeArgs{concrete_handle(o->handle()),
                              &src,
                              &filter,
                              &bias,
                              &z,
                              filter_meta,
                              &dst,
                              o->param().nonlineMode},
          opr{o},
          preprocessed_filter{preprocessed_filter} {}

ConvBiasForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvBiasForwardImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in filter, _megdnn_tensor_in bias, _megdnn_tensor_in z,
        _megdnn_tensor_out dst, _megdnn_workspace workspace,
        const PreprocessedFilter* preprocessed_filter)
        : SizeArgs(opr, src.layout, filter.layout, bias.layout, z.layout,
                   dst.layout, preprocessed_filter),
          src_tensor{&src},
          filter_tensor{&filter},
          bias_tensor{&bias},
          z_tensor{&z},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string ConvBiasForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    std::string nonlinear_mode_str;
    switch (nonlinear_mode) {
        case param::ConvBias::NonlineMode::RELU:
            nonlinear_mode_str = "RELU";
            break;
        case param::ConvBias::NonlineMode::SIGMOID:
            nonlinear_mode_str = "SIGMOID";
            break;
        case param::ConvBias::NonlineMode::IDENTITY:
            nonlinear_mode_str = "IDENTITY";
            break;
        default:
            megdnn_throw("invalid conv bias nonlinear mode");
    }
    return megdnn_mangle(ssprintf(
            "src=%s, filter=%u{%u,%u,%u,%u}, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s, "
            "nonlinear_mode=%s",
            src_layout->to_string().c_str(), fm.group, fm.ocpg, fm.icpg,
            fm.spatial[0], fm.spatial[1], dst_layout->to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
            fm.dilation[0], fm.dilation[1], !fm.should_flip,
            src_layout->dtype.name(), dst_layout->dtype.name(),
            nonlinear_mode_str.c_str()));
}

void ConvBiasForwardImpl::AlgoPack::fill_cudnn_algos() {
    for (auto&& algo : CudnnAlgoPack::conv_fwd_algos()) {
        cudnn_conv_bias_activations.push_back(algo.first);
        cudnn_convs.push_back(algo.first);
    }
}

#if CUDA_VERSION >= 10000
void ConvBiasForwardImpl::AlgoPack::fill_imma_algos() {
    int8_chwn4_imma.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize::IMMA16x16x16});
    int8_chwn4_imma.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize::IMMA32x8x16});
    int8_chwn4_imma.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemm::MMATileSize::IMMA8x32x16});
    int8_nchw4_imma.push_back(
            {AlgoInt8NCHW4IMMAImplicitGemm::MMATileSize::IMMA16x16x16});
    int8_nchw4_imma.push_back(
            {AlgoInt8NCHW4IMMAImplicitGemm::MMATileSize::IMMA32x8x16});
    int8_nchw4_imma.push_back(
            {AlgoInt8NCHW4IMMAImplicitGemm::MMATileSize::IMMA8x32x16});
    int8_chwn4_imma_reorder_filter.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmReorderFilter::MMATileSize::
                     IMMA16x16x16});
    int8_chwn4_imma_reorder_filter.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmReorderFilter::MMATileSize::
                     IMMA32x8x16});
    int8_chwn4_imma_reorder_filter.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmReorderFilter::MMATileSize::
                     IMMA8x32x16});
    int8_chwn4_imma_unroll_width.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth::MMATileSize::
                     IMMA16x16x16});
    int8_chwn4_imma_unroll_width.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth::MMATileSize::
                     IMMA32x8x16});
    int8_chwn4_imma_unroll_width.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth::MMATileSize::
                     IMMA8x32x16});
#if CUDA_VERSION >= 10020
    {
        using AlgoParam = AlgoInt8NCHW32IMMAImplicitGemm::AlgoParam;
        int8_nchw32_imma.emplace_back(AlgoParam{128, 256, 64, 64, 64, 64});
        int8_nchw32_imma.emplace_back(AlgoParam{256, 128, 64, 64, 64, 64});
        int8_nchw32_imma.emplace_back(AlgoParam{128, 128, 64, 64, 64, 64});
        int8_nchw32_imma.emplace_back(AlgoParam{64, 128, 64, 32, 64, 64});
        int8_nchw32_imma.emplace_back(AlgoParam{128, 64, 64, 64, 32, 64});
        int8_nchw32_imma.emplace_back(AlgoParam{64, 64, 64, 32, 32, 64});
        int8_nchw32_imma.emplace_back(AlgoParam{32, 64, 64, 32, 16, 64});
    }
#endif
}
#endif

void ConvBiasForwardImpl::AlgoPack::fill_dp4a_algos() {
    using AlgoParam = AlgoInt8NCHW4DotProdImplicitGemm::AlgoParam;
    int8_nchw4_dotprod.emplace_back(AlgoParam{128, 128, 32, 64, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{128, 64, 32, 64, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{64, 128, 32, 64, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 128, 32, 32, 64, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{64, 64, 32, 64, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 64, 32, 32, 64, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{64, 32, 32, 64, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 32, 32, 32, 32, 32, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 128, 16, 16, 128, 16, 1});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8, 2});
}


ConvBiasForwardImpl::AlgoBase*
ConvBiasForwardImpl::AlgoPack::cudnn_conv_from_enum(
        cudnnConvolutionFwdAlgo_t algo) {
    for (auto&& i : cudnn_convs) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(
            megdnn_mangle(ssprintf("can not find cudnn conv fwd algorithm %d",
                                   static_cast<int>(algo))));
}

ConvBiasForwardImpl::AlgoBase*
ConvBiasForwardImpl::AlgoPack::cudnn_conv_bias_act_from_enum(
        cudnnConvolutionFwdAlgo_t algo) {
    for (auto&& i : cudnn_conv_bias_activations) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(megdnn_mangle(
            ssprintf("can not find cudnn conv bias act algorithm %d",
                     static_cast<int>(algo))));
}

// vim: syntax=cpp.doxygen
