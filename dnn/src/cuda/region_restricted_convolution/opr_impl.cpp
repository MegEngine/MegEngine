#include "src/cuda/region_restricted_convolution/opr_impl.h"
#include "src/cuda/cutlass/singleton.h"
#include "src/cuda/region_restricted_convolution/chanwise/depthwise_large_filter.cuh"
#include "src/cuda/region_restricted_convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace region_restricted_convolution;
using namespace cutlass::library;

/* ============== RegionRestrictedConvolutionForwardImpl ============== */
void RegionRestrictedConvolutionForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto fm = check_exec(
            src.layout, filter.layout, rin.layout, rout.layout, dst.layout,
            workspace.size);
    auto kparam = chanwise::Param::load(
            src.layout, dst.layout, fm,
            param().compute_mode == Param::ComputeMode::DEFAULT);
    megdnn_assert(
            fm.group > 1 && src.layout.dtype.category() == DTypeCategory::FLOAT &&
            param().compute_mode == Param::ComputeMode::DEFAULT &&
            fm.spatial_ndim == 2 && fm.icpg == 1 && fm.ocpg == 1 &&
            fm.dilation[0] == 1 && fm.dilation[1] == 1 && !fm.should_flip &&
            param().stride_h == 1 && param().stride_w == 1);
    if (rin.layout.dtype == dtype::Uint8()) {
        megdnn_assert((src.layout.shape[3] & 3) == 0 && (dst.layout.shape[3] & 3) == 0);
    }

    auto stream = cuda_stream(handle());

    if (filter.layout.dtype == dtype::Float32() && rin.layout.dtype == dtype::Int32() &&
        rout.layout.dtype == dtype::Int32()) {
        chanwise::run_fwd_depthwise_large_filter(
                dst.ptr<float>(), src.ptr<float>(), filter.ptr<float>(), rin.ptr<int>(),
                rout.ptr<int>(), kparam, stream);
    } else if (
            filter.layout.dtype == dtype::Float32() &&
            rin.layout.dtype == dtype::Uint8() && rout.layout.dtype == dtype::Uint8()) {
        chanwise::run_fwd_depthwise_large_filter(
                dst.ptr<float>(), src.ptr<float>(), filter.ptr<float>(),
                rin.ptr<uint8_t>(), rout.ptr<uint8_t>(), kparam, stream);
    } else {
        megdnn_assert_internal(0);
    }
}

size_t RegionRestrictedConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff, const TensorLayout& rin,
        const TensorLayout& rout, const TensorLayout& grad) {
    return 0;
}

/* ============== RegionRestrictedConvolutionBackwardDataImpl ============== */
void RegionRestrictedConvolutionBackwardDataImpl::exec(
        _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    auto fm = check_exec(
            filter.layout, diff.layout, rin.layout, rout.layout, grad.layout,
            workspace.size);
    // XXX: a naive impl to set deconv padding to param, needs optimization in future.
    [&]() -> void {
        size_t stride = fm.stride[0];
        size_t src_size = grad.layout.shape[2];
        size_t fwd_pad = fm.padding[0];
        size_t filter_size = fm.spatial[0];
        size_t deconv_pad = (stride * src_size - stride + stride * filter_size -
                             src_size - 2 * fwd_pad + filter_size - 1) /
                            (2 * stride);
        fm.padding[0] = fm.padding[1] = deconv_pad;
        return;
    }();
    auto kparam = chanwise::Param::load(
            diff.layout, grad.layout, fm,
            param().compute_mode == Param::ComputeMode::DEFAULT);
    megdnn_assert(
            fm.group > 1 && diff.layout.dtype.category() == DTypeCategory::FLOAT &&
            param().compute_mode == Param::ComputeMode::DEFAULT &&
            fm.spatial_ndim == 2 && fm.icpg == 1 && fm.ocpg == 1 &&
            fm.dilation[0] == 1 && fm.dilation[1] == 1 && !fm.should_flip &&
            param().stride_h == 1 && param().stride_w == 1);
    // NOTE: uint8 dtype region mask requires the spatial size of src&dst is 4*N
    if (rin.layout.dtype == dtype::Uint8()) {
        megdnn_assert(
                (grad.layout.shape[3] & 3) == 0 && (diff.layout.shape[3] & 3) == 0);
    }
    auto stream = cuda_stream(handle());
    if (filter.layout.dtype == dtype::Float32() && rin.layout.dtype == dtype::Int32() &&
        rout.layout.dtype == dtype::Int32()) {
        chanwise::run_bwd_depthwise_large_filter(
                grad.ptr<dt_float32>(), diff.ptr<dt_float32>(),
                filter.ptr<dt_float32>(), rin.ptr<dt_int32>(), rout.ptr<dt_int32>(),
                kparam, stream);
    } else if (
            filter.layout.dtype == dtype::Float32() &&
            rin.layout.dtype == dtype::Uint8() && rout.layout.dtype == dtype::Uint8()) {
        chanwise::run_bwd_depthwise_large_filter(
                grad.ptr<dt_float32>(), diff.ptr<dt_float32>(),
                filter.ptr<dt_float32>(), rin.ptr<dt_uint8>(), rout.ptr<dt_uint8>(),
                kparam, stream);
    } else {
        megdnn_throw("undefined or unimplemented region restricted conv mode");
    }
}

size_t RegionRestrictedConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout&,
        const TensorLayout&, const TensorLayout& grad) {
    return 0;
}

/* ============== RegionRestrictedConvolutionBackwardFilterImpl ============== */
void RegionRestrictedConvolutionBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    auto fm = check_exec(
            src.layout, diff.layout, rin.layout, rout.layout, grad.layout,
            workspace.size);

    megdnn_assert(
            fm.group > 1 && src.layout.dtype.category() == DTypeCategory::FLOAT &&
            param().compute_mode == Param::ComputeMode::DEFAULT &&
            fm.spatial_ndim == 2 && fm.icpg == 1 && fm.ocpg == 1 &&
            fm.dilation[0] == 1 && fm.dilation[1] == 1 && !fm.should_flip &&
            param().stride_h == 1 && param().stride_w == 1);

    int hi = src.layout.operator[](2), wi = src.layout.operator[](3);
    int n = diff.layout.operator[](0), ho = diff.layout.operator[](2),
        wo = diff.layout.operator[](3);
    int co = fm.group, ci = co, groups = co;
    int fh = fm.spatial[0], fw = fm.spatial[1];
    int sh = fm.stride[0], sw = fm.stride[1];
    int ph = fm.padding[0], pw = fm.padding[1];
    int dh = 0, dw = 0;

    // check if channelwise convolution
    megdnn_assert(fm.icpg == 1 && fm.ocpg == 1);
    auto stream = cuda_stream(handle());

    float alpha = 1.f;
    float beta = 0.f;

    ConvolutionKey key;

    int threadblock_shape_n = 128;
    int warp_shape_m = 32;
    int warp_shape_n = 64;
    if (grad.layout.operator[](3) % 8 < 4) {
        threadblock_shape_n = 64;
        warp_shape_m = 64;
        warp_shape_n = 32;
    }

    if (rin.layout.dtype == dtype::Int32() && rout.layout.dtype == dtype::Int32()) {
        key = {
                cutlass::conv::Operator::kWgrad,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                cutlass::conv::ConvType::kDepthwiseConvolution,
                128,
                threadblock_shape_n,
                8,
                warp_shape_m,
                warp_shape_n,
                8,
                1,
                1,
                1,
                cutlass::epilogue::EpilogueType::kLinearCombination,
                1,
                cutlass::conv::SpecialOptimizeDesc::NONE,
                1,
                1,
                false,
                NumericTypeID::kS32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kS32,
                LayoutTypeID::kTensorNCHW,
        };
    } else if (
            rin.layout.dtype == dtype::Uint8() && rout.layout.dtype == dtype::Uint8()) {
        key = {
                cutlass::conv::Operator::kWgrad,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kF32,
                cutlass::conv::ConvType::kDepthwiseConvolution,
                128,
                threadblock_shape_n,
                8,
                warp_shape_m,
                warp_shape_n,
                8,
                1,
                1,
                1,
                cutlass::epilogue::EpilogueType::kLinearCombination,
                1,
                cutlass::conv::SpecialOptimizeDesc::NONE,
                1,
                1,
                false,
                NumericTypeID::kS8,
                LayoutTypeID::kTensorNCHW,
                NumericTypeID::kS8,
                LayoutTypeID::kTensorNCHW,
        };
    } else {
        megdnn_throw(ssprintf(
                             "don't support region restricted type rin: %s, rout: %s",
                             rin.layout.dtype.name(), rout.layout.dtype.name())
                             .c_str());
    }

    const Operation* op =
            (const Operation*)Singleton::get().operation_table.find_op(key);

    cutlass::conv::Conv2dProblemSize problem_size{
            n,      hi, wi, ci, co, fh, fw, ho,
            wo,     ph, pw, sh, sw, dh, dw, cutlass::conv::Mode::kCrossCorrelation,
            1,       // split k slices, always 1
            groups,  // groups
    };

    cutlass::library::ConvolutionArguments conv_args{
            problem_size, src.raw_ptr(),  diff.raw_ptr(), nullptr,
            nullptr,      grad.raw_ptr(), &alpha,         &beta,
            nullptr,      nullptr,        nullptr,        nullptr,
            nullptr,      nullptr,        rin.raw_ptr(),  rout.raw_ptr()};

    cutlass_check(op->run(&conv_args, nullptr, stream));

    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
