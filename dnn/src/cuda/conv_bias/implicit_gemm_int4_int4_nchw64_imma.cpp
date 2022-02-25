#include "src/cuda/conv_bias/algo.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#if CUDA_VERSION >= 10020
size_t ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return 0;
    } else {
        return args.filter_layout->span().dist_byte();
    }
}

size_t ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    megdnn_assert(args.preprocessed_filter->tensors.size() == 1);
    void* filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr();
    reorder_filter(args, filter_ptr);
}

std::tuple<void*, void*> ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::
        prepare_filter_bias(const ExecArgs& args) const {
    void* filter_ptr = nullptr;
    if (args.preprocessed_filter) {
        megdnn_assert(args.preprocessed_filter->tensors.size() == 1);
        filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr();
    } else {
        filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
        reorder_filter(args, filter_ptr);
    }
    void* bias_ptr = args.bias_tensor->raw_ptr();
    return {filter_ptr, bias_ptr};
}

std::tuple<float, float, float, float, float> ConvBiasForwardImpl::
        AlgoInt4Int4NCHW64IMMAImplicitGemm::get_constants(const ExecArgs& args) const {
    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS4>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS4>().scale,
          bias_scale = args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;

    float alpha = src_scale * filter_scale / dst_scale, beta = bias_scale / dst_scale,
          gamma = 0.f, delta = 0.f, theta = 0.f;

    if (args.z_layout->ndim > 0) {
        float z_scale = args.z_layout->dtype.param<dtype::QuantizedS4>().scale;
        gamma = z_scale / dst_scale;
    }

    return {alpha, beta, gamma, delta, theta};
}
#endif

// vim: syntax=cpp.doxygen
