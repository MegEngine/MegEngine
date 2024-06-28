#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

ConvBiasForwardImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);
    non_cudnn_algos.push_back(&chanwise_small);
    non_cudnn_algos.push_back(&depthwise_large_filter);

    non_cudnn_algos.push_back(&inplace_matmul);
    non_cudnn_algos.push_back(&matmul);
    non_cudnn_algos.push_back(&matmul8x8x32);
    non_cudnn_algos.push_back(&batched_matmul);
    non_cudnn_algos.push_back(&int1_simple);

#if CUDNN_VERSION >= 8020 && 0  // FIXME(hc): need fix
    all_algos.push_back(&cudnn_conv_v8);
    all_algos.push_back(&cudnn_conv_bias_activation_v8);
#endif

    fill_cudnn_algos();
    for (auto&& algo : cudnn_conv_bias_activations) {
        all_algos.push_back(&algo);
    }

    //! add conv+nonlinear algos
    std::vector<AlgoBase*> conv_algos;
    conv_algos.push_back(&chanwise);
    conv_algos.push_back(&chanwise_small);
    conv_algos.push_back(&depthwise_large_filter);
    conv_algos.push_back(&chanwise8x8x32);
    for (auto&& algo : cudnn_convs) {
        conv_algos.push_back(&algo);
    }
    conv_algos.push_back(&inplace_matmul);
    conv_algos.push_back(&matmul);
    conv_algos.push_back(&matmul8x8x32);
    conv_algos.push_back(&batched_matmul);
    conv_algos.push_back(&group);
    conv_algos.push_back(&int1_simple);

    for (auto&& algo : conv_algos) {
        all_algos.push_back(algo);
    }

    all_algos.push_back(&bfloat16);
    bfloat16_algos.push_back(&bfloat16);

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
    for (auto&& algo : int8_nhwc_imma) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : int4_int4_nchw64_imma) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : uint4_int4_nchw64_imma) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : int4_int4_nhwc_imma) {
        all_algos.push_back(&algo);
    }
    for (auto&& algo : uint4_int4_nhwc_imma) {
        all_algos.push_back(&algo);
    }
#endif
#endif
    fill_dp4a_algos();
    for (auto&& algo : int8_nchw4_dotprod) {
        all_algos.push_back(&algo);
    }
    fill_dwconv_algos();
    all_algos.push_back(&int8_chwn4_dotprod);
    all_algos.push_back(&fallback_nchw_qs8);

    fill_ptx_algos();
    for (auto&& algo : algo_ptx_conv2d_u4_s4) {
        all_algos.push_back(&algo);
    }

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
        const ConvBiasForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst, const PreprocessedFilter* preprocessed_filter)
        : SizeArgs(
                  o, src, filter, o->make_canonized_filter_meta(src.ndim, filter), bias,
                  z, dst, preprocessed_filter) {}

ConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvBiasForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const CanonizedFilterMeta& filter_meta,
        const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter)
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
        ConvBiasForwardImpl* opr, _megdnn_tensor_in src, _megdnn_tensor_in filter,
        _megdnn_tensor_in bias, _megdnn_tensor_in z, _megdnn_tensor_out dst,
        _megdnn_workspace workspace, const PreprocessedFilter* preprocessed_filter)
        : SizeArgs(
                  opr, src.layout, filter.layout, bias.layout, z.layout, dst.layout,
                  preprocessed_filter),
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
        case param::ConvBias::NonlineMode::H_SWISH:
            nonlinear_mode_str = "H_SWISH";
            break;
        default:
            megdnn_throw("invalid conv bias nonlinear mode");
    }
    return ssprintf(
            "src=%s, filter=%s, bias=%s, z=%s, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s, "
            "nonlinear_mode=%s",
            src_layout->to_string().c_str(), filter_layout->to_string().c_str(),
            bias_layout->to_string().c_str(), z_layout->to_string().c_str(),
            dst_layout->to_string().c_str(), fm.padding[0], fm.padding[1], fm.stride[0],
            fm.stride[1], fm.dilation[0], fm.dilation[1], !fm.should_flip,
            src_layout->dtype.name(), dst_layout->dtype.name(),
            nonlinear_mode_str.c_str());
}

param::Convolution ConvBiasForwardImpl::AlgoBase::get_param_convolution(
        const SizeArgs& args) const {
    param::Convolution::Mode mode;
    param::Convolution::Sparse sparse = args.filter_meta.group > 1
                                              ? param::Convolution::Sparse::GROUP
                                              : param::Convolution::Sparse::DENSE;
    if (args.filter_meta.should_flip) {
        mode = param::Convolution::Mode::CONVOLUTION;
    } else {
        mode = param::Convolution::Mode::CROSS_CORRELATION;
    }
    return param::Convolution{
            mode,
            args.filter_meta.padding[0],
            args.filter_meta.padding[1],
            args.filter_meta.stride[0],
            args.filter_meta.stride[1],
            args.filter_meta.dilation[1],
            args.filter_meta.dilation[0],
            sparse,
            args.filter_meta.format,
            args.opr->param().compute_mode};
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
            {AlgoInt8CHWN4IMMAImplicitGemmReorderFilter::MMATileSize::IMMA16x16x16});
    int8_chwn4_imma_reorder_filter.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmReorderFilter::MMATileSize::IMMA32x8x16});
    int8_chwn4_imma_reorder_filter.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmReorderFilter::MMATileSize::IMMA8x32x16});
    int8_chwn4_imma_unroll_width.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth::MMATileSize::IMMA16x16x16});
    int8_chwn4_imma_unroll_width.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth::MMATileSize::IMMA32x8x16});
    int8_chwn4_imma_unroll_width.push_back(
            {AlgoInt8CHWN4IMMAImplicitGemmUnrollWidth::MMATileSize::IMMA8x32x16});
#if CUDA_VERSION >= 10020
    {
        using AlgoParam = AlgoInt8NCHW32IMMAImplicitGemm::AlgoParam;
        int8_nchw32_imma.emplace_back(AlgoParam{128, 256, 64, 64, 64, 64, 8, 8, 16, 2});
        int8_nchw32_imma.emplace_back(AlgoParam{256, 128, 64, 64, 64, 64, 8, 8, 16, 2});
        int8_nchw32_imma.emplace_back(AlgoParam{128, 128, 64, 64, 64, 64, 8, 8, 16, 2});
        int8_nchw32_imma.emplace_back(AlgoParam{128, 64, 64, 64, 32, 64, 8, 8, 16, 2});
        int8_nchw32_imma.emplace_back(AlgoParam{64, 128, 64, 32, 64, 64, 8, 8, 16, 2});
        int8_nchw32_imma.emplace_back(AlgoParam{128, 64, 32, 64, 32, 32, 8, 8, 16, 1});
        int8_nchw32_imma.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 8, 8, 16, 1});
        int8_nchw32_imma.emplace_back(AlgoParam{64, 128, 32, 32, 64, 32, 8, 8, 16, 1});
        int8_nchw32_imma.emplace_back(AlgoParam{32, 128, 32, 32, 64, 32, 8, 8, 16, 1});
    }
    {
        using AlgoParam = AlgoInt8NHWCIMMAImplicitGemm::AlgoParam;
        int8_nhwc_imma.emplace_back(AlgoParam{64, 16, 32, 64, 16, 32, 8, 8, 16, 2, 16});
        int8_nhwc_imma.emplace_back(AlgoParam{64, 16, 32, 64, 16, 32, 8, 8, 16, 2, 8});
        int8_nhwc_imma.emplace_back(AlgoParam{64, 16, 32, 64, 16, 32, 8, 8, 16, 2, 4});
        int8_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 32, 64, 32, 32, 8, 8, 16, 1, 16});
        int8_nhwc_imma.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 8, 8, 16, 1, 8});
        int8_nhwc_imma.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 8, 8, 16, 1, 4});
    }
    {
        using AlgoParam = AlgoInt4Int4NCHW64IMMAImplicitGemm::AlgoParam;
        int4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 128, 128, 64, 64, 128, 8, 8, 32, 2});
        int4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 256, 128, 64, 64, 128, 8, 8, 32, 2});
        int4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 64, 128, 64, 64, 128, 8, 8, 32, 2});
        int4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1});
    }
    {
        using AlgoParam = AlgoUInt4Int4NCHW64IMMAImplicitGemm::AlgoParam;
        uint4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 128, 128, 64, 64, 128, 8, 8, 32, 2});
        uint4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 256, 128, 64, 64, 128, 8, 8, 32, 2});
        uint4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 64, 128, 64, 64, 128, 8, 8, 32, 2});
        uint4_int4_nchw64_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1});
    }
    {
        using AlgoParam = AlgoInt4Int4NHWCIMMAImplicitGemm::AlgoParam;
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 16, 64, 128, 16, 64, 8, 8, 32, 2, 32});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 16, 64, 128, 16, 64, 8, 8, 32, 2, 16});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 16, 64, 128, 16, 64, 8, 8, 32, 2, 8});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 64, 64, 32, 64, 8, 8, 32, 1, 32});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 64, 64, 32, 64, 8, 8, 32, 1, 16});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 64, 64, 32, 64, 8, 8, 32, 1, 8});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1, 32});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1, 16});
        int4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1, 8});
    }
    {
        using AlgoParam = AlgoUInt4Int4NHWCIMMAImplicitGemm::AlgoParam;
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 16, 64, 128, 16, 64, 8, 8, 32, 2, 32});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 16, 64, 128, 16, 64, 8, 8, 32, 2, 16});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 16, 64, 128, 16, 64, 8, 8, 32, 2, 8});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 64, 64, 32, 64, 8, 8, 32, 1, 32});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 64, 64, 32, 64, 8, 8, 32, 1, 16});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 32, 64, 64, 32, 64, 8, 8, 32, 1, 8});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1, 32});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1, 16});
        uint4_int4_nhwc_imma.emplace_back(
                AlgoParam{128, 64, 64, 64, 64, 64, 8, 8, 32, 1, 8});
    }
#endif
}
#endif

void ConvBiasForwardImpl::AlgoPack::fill_dwconv_algos() {
    using AlgoParam = AlgoCutlassConvolutionBase::AlgoParam;
    /// preferred algo
    f32_implicit_bmm.emplace_back(AlgoParam{64, 128, 8, 32, 64, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{128, 128, 8, 32, 64, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{128, 64, 8, 64, 32, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{128, 32, 8, 64, 32, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{32, 128, 8, 32, 64, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{64, 64, 8, 32, 64, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{32, 64, 8, 32, 64, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{32, 32, 8, 32, 32, 8, 1, 1, 1, 2});
    f32_implicit_bmm.emplace_back(AlgoParam{64, 32, 8, 64, 32, 8, 1, 1, 1, 2});
    for (auto&& algo : f32_implicit_bmm) {
        all_algos.push_back(&algo);
    }
#if CUDA_VERSION >= 10010
    /// preferred algo
    f16_implicit_bmm.emplace_back(AlgoParam{64, 128, 32, 32, 32, 32, 8, 8, 4, 2});
//  TODO: optimize the method to avoid too many resources requested.
//! Remove this config to avoid too many resources requested with cuda118 when align is
//! 8 or 1.
#if CUDA_VERSION != 11080
    f16_implicit_bmm.emplace_back(AlgoParam{128, 128, 32, 32, 32, 32, 8, 8, 4, 2});
#endif
    f16_implicit_bmm.emplace_back(AlgoParam{128, 256, 32, 64, 64, 32, 8, 8, 4, 2});
    f16_implicit_bmm.emplace_back(AlgoParam{128, 64, 32, 32, 32, 32, 8, 8, 4, 2});
    f16_implicit_bmm.emplace_back(AlgoParam{64, 64, 32, 32, 32, 32, 8, 8, 4, 2});
    for (auto&& algo : f16_implicit_bmm) {
        all_algos.push_back(&algo);
    }
#endif
}

void ConvBiasForwardImpl::AlgoPack::fill_dp4a_algos() {
    using AlgoParam = AlgoInt8NCHW4DotProdImplicitGemm::AlgoParam;
    int8_nchw4_dotprod.emplace_back(AlgoParam{128, 128, 32, 64, 32, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{128, 64, 32, 64, 32, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{64, 128, 32, 64, 32, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 128, 32, 32, 64, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{128, 32, 32, 64, 32, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 64, 32, 32, 64, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{64, 32, 32, 64, 32, 32, 1, 1, 4, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 128, 16, 16, 128, 16, 1, 1, 4, 1});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8, 1, 1, 4, 2});
}

void ConvBiasForwardImpl::AlgoPack::fill_ptx_algos() {
    algo_ptx_conv2d_u4_s4.emplace_back(
            AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm{128, 256, 256});
    algo_ptx_conv2d_u4_s4.emplace_back(
            AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm{128, 128, 128});
    // FIXME: destroy event error on NVIDIA A2 after execute the algo
    //     algo_ptx_conv2d_u4_s4.emplace_back(
    //             AlgoPTXUInt4Int4NCHW64IMMAImplicitGemm{256, 64, 128});
}

ConvBiasForwardImpl::AlgoBase* ConvBiasForwardImpl::AlgoPack::cudnn_conv_from_enum(
        cudnnConvolutionFwdAlgo_t algo) {
    for (auto&& i : cudnn_convs) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(ssprintf(
            "can not find cudnn conv fwd algorithm %d", static_cast<int>(algo)));
}

ConvBiasForwardImpl::AlgoBase* ConvBiasForwardImpl::AlgoPack::
        cudnn_conv_bias_act_from_enum(cudnnConvolutionFwdAlgo_t algo) {
    for (auto&& i : cudnn_conv_bias_activations) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(ssprintf(
            "can not find cudnn conv bias act algorithm %d", static_cast<int>(algo)));
}

// vim: syntax=cpp.doxygen
