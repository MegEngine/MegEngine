#include "src/cuda/conv_bias/chanwise/depthwise_large_filter.cuh"
#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

namespace {
inline bool is_available_depthwise_large_filter(const chanwise::Param& param) {
    if ((param.stride_h == 1 && param.stride_w == 1) ||
        (param.stride_h == 2 && param.stride_w == 2)) {
        auto&& device_prop = cuda::current_device_prop();
        static int const unroll_oh = 1, unroll_fh = 1;
        CHECK(BWD)
    }
    return false;
}
}  // anonymous namespace

bool ConvolutionBackwardDataImpl::AlgoDepthwiseLargeFilter::is_available(
        const SizeArgs& args) const {
    if (!args.grad_layout->is_contiguous() || !args.diff_layout->is_contiguous()) {
        return false;
    }
    if (args.diff_layout->dtype != args.filter_layout->dtype &&
        (args.diff_layout->dtype != dtype::Float32()
#if CUDA_VERSION >= 9000
         || args.diff_layout->dtype != dtype::Float16()
#endif
                 )) {
        return false;
    }

    auto param = chanwise::Param::from_fwd_args(
            {args.handle, args.diff_layout, args.filter_layout, args.filter_meta,
             args.grad_layout});
    auto&& fm = args.filter_meta;
    return fm.group > 1 && args.filter_meta.format == Param::Format::NCHW &&
           args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           args.opr->param().compute_mode == Param::ComputeMode::DEFAULT &&
           fm.spatial_ndim == 2 && fm.icpg == 1 && fm.ocpg == 1 &&
           fm.dilation[0] == 1 && fm.dilation[1] == 1 && !fm.should_flip &&
           is_available_depthwise_large_filter(param);
}

size_t ConvolutionBackwardDataImpl::AlgoDepthwiseLargeFilter::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return 0;
}

void ConvolutionBackwardDataImpl::AlgoDepthwiseLargeFilter::exec(
        const ExecArgs& args) const {
    auto kparam = chanwise::Param::from_fwd_args(
            {args.handle, args.diff_layout, args.filter_layout, args.filter_meta,
             args.grad_layout});
    auto stream = cuda_stream(args.handle);
    switch (args.diff_layout->dtype.enumv()) {
        case DTypeEnum::Float32:
            chanwise::run_bwd_depthwise_large_filter(
                    args.grad_tensor->ptr<float>(), args.diff_tensor->ptr<float>(),
                    args.filter_tensor->ptr<float>(), kparam, stream);
            break;
#if CUDA_VERSION >= 9000
        case DTypeEnum::Float16:
            chanwise::run_bwd_depthwise_large_filter(
                    static_cast<half*>(args.grad_tensor->raw_ptr()),
                    static_cast<half*>(args.diff_tensor->raw_ptr()),
                    static_cast<half*>(args.filter_tensor->raw_ptr()), kparam, stream);
            break;
#endif
        default:
            megdnn_assert_internal(0);
    }
}

// vim: syntax=cpp.doxygen
