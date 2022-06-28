#include "src/cuda/conv_bias/helper.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

ConvBiasDesc::ConvBiasDesc() {
    cudnn_check(cudnnCreateActivationDescriptor(&act_desc));
    cudnn_check(cudnnCreateConvolutionDescriptor(&conv_desc));
#if CUDNN_VERSION >= 7000
    cudnn_check(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
}

ConvBiasDesc::~ConvBiasDesc() {
    cudnn_check(cudnnDestroyConvolutionDescriptor(conv_desc));
    cudnn_check(cudnnDestroyActivationDescriptor(act_desc));
}

void ConvBiasDesc::set_conv_bias(
        DType data_type, const param::ConvBias& param, size_t nr_group) {
#if CUDNN_VERSION < 7100
    megdnn_throw("ConvBias(CUDNN_ACTIVATION_IDENTITY) require cudnn 7.1 or higher");
#else
    cudnnConvolutionMode_t mode;
    using Param = param::ConvBias;
    switch (param.mode) {
        case Param::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case Param::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw("conv mode must be conv or xcorr.");
    }
    cudnn_check(cudnnSetConvolutionGroupCount(conv_desc, nr_group));
    cudnnDataType_t compute_type;
    switch (data_type.category()) {
        case DTypeCategory::FLOAT:
            compute_type = CUDNN_DATA_FLOAT;
            break;
        case DTypeCategory::INT:
        case DTypeCategory::QUANTIZED:
            compute_type = CUDNN_DATA_INT32;
            break;
        default:
            megdnn_throw("unspport data type for conv bias");
    }
    if (data_type.enumv() == DTypeEnum::Float16) {
        auto comp_mode = param.compute_mode;
        compute_type = get_compute_type_fp16(comp_mode);
    }
    cudnn_check(cudnnSetConvolution2dDescriptor(
            conv_desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode, compute_type));

    switch (param.nonlineMode) {
        case Param::NonlineMode::IDENTITY:
        case Param::NonlineMode::SIGMOID:
        case Param::NonlineMode::H_SWISH:
            cudnn_check(cudnnSetActivationDescriptor(
                    act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        case Param::NonlineMode::RELU:
            cudnn_check(cudnnSetActivationDescriptor(
                    act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        default:
            megdnn_throw("unsupported non linear mode");
    }
#endif
}

void ConvBiasDesc::set_conv(
        DType data_type, const param::ConvBias& param, const size_t nr_group) {
    using Param = param::ConvBias;
    cudnnConvolutionMode_t mode;
    switch (param.mode) {
        case Param::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case Param::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw("conv mode must be conv or xcorr.");
    }
    cudnnDataType_t compute_type;
    MEGDNN_MARK_USED_VAR(compute_type);
    if (data_type.enumv() == DTypeEnum::Float32) {
        // FLOAT_CONFIG
        compute_type = CUDNN_DATA_FLOAT;
    } else if (data_type.enumv() == DTypeEnum::Float16) {
        auto comp_mode = param.compute_mode;
        compute_type = get_compute_type_fp16(comp_mode);
#if CUDNN_MAJOR >= 7
    } else if (
            data_type.category() == DTypeCategory::INT ||
            data_type.category() == DTypeCategory::QUANTIZED) {
        compute_type = CUDNN_DATA_INT32;
#endif
    } else {
        megdnn_throw("unspport data type for conv bias");
    }
#if CUDNN_MAJOR >= 7
    cudnn_check(cudnnSetConvolutionGroupCount(conv_desc, nr_group));
#else
    megdnn_assert(nr_group == 1);
#endif

#if CUDNN_MAJOR >= 6
    cudnn_check(cudnnSetConvolution2dDescriptor(
            conv_desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode, compute_type));
#else
    cudnn_check(cudnnSetConvolution2dDescriptor(
            conv_desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode));
#endif
}

namespace conv_bias {

bool is_cudnn_supported(const BiasForwardSizeArgs& args) {
    if (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS1)
        return false;

    if ((args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS4 ||
         args.src_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
        args.filter_layout->dtype.enumv() == DTypeEnum::QuantizedS4)
        return false;

    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }

    // CUDNN_STATUS_EXECUTION_FAILED on Tegra K1, so disable CUDNN
    // on Tegra K1.
    if (args.handle->is_tegra_k1())
        return false;

    if (args.filter_meta.format == param::Convolution::Format::NCHW4 ||
        args.filter_meta.format == param::Convolution::Format::NCHW32) {
        if (args.dst_layout->dtype.enumv() != DTypeEnum::Int8 &&
            args.dst_layout->dtype.enumv() != DTypeEnum::QuantizedS8) {
            return false;
        }
    } else if (
            args.filter_meta.format != param::Convolution::Format::NCHW &&
            args.filter_meta.format != param::Convolution::Format::NHWC) {
        return false;
    }
    auto& fm = args.filter_meta;
    bool supported = true;
    supported &= (fm.spatial_ndim == 2);
#if CUDNN_VERSION < 7000
    supported &= (fm.group == 1);
#endif
#if CUDNN_VERSION < 7500
    supported &= (fm.dilation[0] == 1 && fm.dilation[1] == 1);
#endif
    return supported;
}

SmallVector<size_t> matmul_get_workspace_bundle(const BiasForwardSizeArgs& args) {
    auto dtype = args.src_layout->dtype;
    auto&& fm = args.filter_meta;
    megdnn_assert(fm.group == 1);
    auto N = args.src_layout->shape[0];
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto OH = args.dst_layout->shape[2], OW = args.dst_layout->shape[3];
    SmallVector<size_t> sizes{
            dtype.size() * args.dst_layout->total_nr_elems(),
            dtype.size() * IC * FH * FW * OH * OW * N};
    if (args.filter_meta.should_flip) {
        sizes.push_back(dtype.size() * OC * IC * FH * FW);
    }
    return sizes;
}

void flip_filter(
        const BiasForwardSizeArgs& args, const Workspace& workspace, RefPtr& ref_ptr) {
    auto&& fm = args.filter_meta;
    megdnn_assert(fm.group == 1 && fm.spatial_ndim == 2);
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto dtype = fm.dtype;
    megdnn_assert(workspace.size >= dtype.size() * OC * IC * FH * FW);

    TensorND src{{{OC, IC, FH, FW}, dtype}, ref_ptr},
            dst{workspace.raw_ptr + (FH * FW - 1) * dtype.size(), src.layout};
    dst.layout.stride[2] = -dst.layout.stride[2];
    dst.layout.stride[3] = -dst.layout.stride[3];
    args.handle->relayout_opr()->exec(src, dst);
    ref_ptr.reset(workspace.raw_ptr);
}

std::pair<float, float> cudnn_get_conv_bias_act_scale_param(
        const TensorLayout& x, const TensorLayout& y, const TensorLayout& w,
        const TensorLayout& b, const TensorLayout& z) {
    float alpha = 1.f, beta = 0.f;
    if (z.ndim > 0)
        beta = 1.f;

    auto get_scale = [](const DType& dtype) -> float {
        megdnn_assert(dtype.category() == DTypeCategory::QUANTIZED);
        switch (dtype.enumv()) {
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return dtype.param<_dt>().scale;
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    };

    auto x_dtype = x.dtype, y_dtype = y.dtype, w_dtype = w.dtype;
    megdnn_assert(
            (x_dtype.category() == y_dtype.category()) ||
            (x_dtype.enumv() == DTypeEnum::QuantizedS8 &&
             y_dtype.enumv() == DTypeEnum::Float32));
    megdnn_assert(x_dtype.category() == w_dtype.category());

    if (x_dtype.category() == DTypeCategory::QUANTIZED) {
        auto expected_bias_scale = get_scale(x_dtype) * get_scale(w_dtype);
        alpha = expected_bias_scale;
        if (y_dtype.category() == DTypeCategory::QUANTIZED)
            alpha /= get_scale(y_dtype);
        if (z.ndim > 0 && z.dtype.category() == DTypeCategory::QUANTIZED) {
            beta = get_scale(z.dtype) / get_scale(y_dtype);
        }
        if (b.dtype.category() == DTypeCategory::QUANTIZED) {
            megdnn_assert(fabs(expected_bias_scale - get_scale(b.dtype)) < 1e-4);
        }
    }
    return {alpha, beta};
}

#if CUDNN_VERSION >= 7500
void cudnn_reorder_filter_and_bias_nchw32(
        const cudnnHandle_t& handle, const void* filter_ptr,
        const CanonizedFilterMeta& fm, const void* bias_ptr, void* reordered_filter_ptr,
        void* reordered_bias_ptr) {
    FilterDesc<param::ConvBias> filter_desc;
    filter_desc.set(fm);
    int reorder_bias = bias_ptr != nullptr;
    cudnn_check(cudnnReorderFilterAndBias(
            handle, filter_desc.desc, CUDNN_DEFAULT_REORDER, filter_ptr,
            reordered_filter_ptr, reorder_bias, bias_ptr, reordered_bias_ptr));
}
#endif

}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
