#include "src/cuda/region_restricted_convolution/opr_impl.h"
#include "src/cuda/region_restricted_convolution/chanwise/depthwise_large_filter.cuh"
#include "src/cuda/region_restricted_convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace region_restricted_convolution;

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
    megdnn_throw(ssprintf(
            "unsupported RegionRestrictedConvolutionBackwardData(%s, %s, %s, %s) -> %s",
            filter.layout.dtype.name(), diff.layout.dtype.name(),
            rin.layout.dtype.name(), rout.layout.dtype.name(),
            grad.layout.dtype.name()));
}

size_t RegionRestrictedConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout&,
        const TensorLayout&, const TensorLayout& grad) {
    size_t workspace_size = 0;
    return workspace_size;
}

/* ============== RegionRestrictedConvolutionBackwardFilterImpl ============== */
void RegionRestrictedConvolutionBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_in rin,
        _megdnn_tensor_in rout, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    megdnn_assert_internal(0);
}

// vim: syntax=cpp.doxygen
