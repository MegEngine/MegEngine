#include "./algo.h"
#include "src/cuda/convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

bool ConvolutionBackwardDataImpl::AlgoChanwise::is_available(
        const SizeArgs& args) const {
    auto kparam = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto&& device_prop = cuda::current_device_prop();
    if (device_prop.sharedMemPerBlock <
        kparam.chl_mul * kparam.flt_h * kparam.flt_w * args.diff_layout->dtype.size()) {
        return false;
    }
    if (!args.grad_layout->is_contiguous() || !args.diff_layout->is_contiguous()) {
        return false;
    }
    if ((args.diff_layout->dtype == args.filter_layout->dtype &&
         args.diff_layout->dtype == dtype::BFloat16()) ||
        (args.diff_layout->dtype == args.filter_layout->dtype &&
         args.diff_layout->dtype == dtype::QuantizedS8())) {
        return false;
    }
    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           fm.spatial_ndim == 2 && fm.icpg == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && !fm.should_flip;
}

size_t ConvolutionBackwardDataImpl::AlgoChanwise::get_workspace_in_bytes(
        const SizeArgs&) const {
    return 0;
}

void ConvolutionBackwardDataImpl::AlgoChanwise::exec(const ExecArgs& args) const {
    auto kparam = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto stream = cuda_stream(args.handle);
    switch (args.diff_layout->dtype.enumv()) {
        case DTypeEnum::Float32:
            return chanwise::run_bwd_data(
                    args.grad_tensor->ptr<float>(), args.diff_tensor->ptr<float>(),
                    args.filter_tensor->ptr<float>(), kparam, stream);

        case DTypeEnum::Float16:
#if CUDA_VERSION >= 9000
            if (is_compute_capability_required(5, 3)) {
                return chanwise::run_bwd_data(
                        static_cast<__half*>(args.grad_tensor->raw_ptr()),
                        static_cast<__half*>(args.diff_tensor->raw_ptr()),
                        static_cast<__half*>(args.filter_tensor->raw_ptr()), kparam,
                        stream);
            } else {
                return chanwise::run_bwd_data(
                        args.grad_tensor->ptr<dt_float16>(),
                        args.diff_tensor->ptr<dt_float16>(),
                        args.filter_tensor->ptr<dt_float16>(), kparam, stream);
            }
#else
            return chanwise::run_bwd_data(
                    args.grad_tensor->ptr<dt_float16>(),
                    args.diff_tensor->ptr<dt_float16>(),
                    args.filter_tensor->ptr<dt_float16>(), kparam, stream);
#endif

        default:
            break;
    }
    megdnn_assert_internal(0);
}

// vim: syntax=cpp.doxygen
