#include "./algo.h"
#include "src/rocm/convolution/chanwise/kern.h.hip"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;
using namespace convolution;

bool ConvolutionBackwardDataImpl::AlgoChanwise::is_available(
        const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           args.opr->param().compute_mode != Param::ComputeMode::FLOAT32 &&
           fm.spatial_ndim == 2 && fm.icpg == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && !fm.should_flip;
}

size_t ConvolutionBackwardDataImpl::AlgoChanwise::get_workspace_in_bytes(
        const SizeArgs&) const {
    return 0;
}

void ConvolutionBackwardDataImpl::AlgoChanwise::exec(const ExecArgs& args) const {
    auto kparam = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto stream = hip_stream(args.handle);
    switch (args.diff_layout->dtype.enumv()) {
#define cb(_dt)                                                                 \
    case DTypeTrait<_dt>::enumv: {                                              \
        using ctype = DTypeTrait<_dt>::ctype;                                   \
        return chanwise::run_bwd_data(                                          \
                args.grad_tensor->ptr<ctype>(), args.diff_tensor->ptr<ctype>(), \
                args.filter_tensor->ptr<ctype>(), kparam, stream);              \
    }
        MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
        default:
            break;
    }
    megdnn_assert_internal(0);
}

// vim: syntax=cpp.doxygen
