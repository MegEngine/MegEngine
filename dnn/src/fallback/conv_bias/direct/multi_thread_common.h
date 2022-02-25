#pragma once

#include "megbrain_build_config.h"

#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"

#if MEGDNN_X86
#include "src/x86/conv_bias/postprocess_helper.h"
#elif (MEGDNN_ARMV7 || MEGDNN_AARCH64)
#include "src/arm_common/conv_bias/postprocess_helper.h"
#else
//! TODO: optimize common postprocess_helper with general intrinsic
#include "src/common/postprocess_helper.h"
#endif

namespace megdnn {
namespace fallback {

template <class io_ctype, class compute_ctype>
class MultithreadDirectConvCommon {
public:
    using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
    using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
    using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

    using kern_direct_conv_f32 = std::function<void(
            const compute_ctype* src, const compute_ctype* filter, compute_ctype* dst,
            size_t, size_t, size_t, size_t, size_t, size_t)>;
    using kern_direct_conv_f32_stride = std::function<void(
            const compute_ctype* src, const compute_ctype* filter, compute_ctype* dst,
            size_t, size_t, size_t, size_t, size_t)>;

    static WorkspaceBundle get_bundle(
            const NCBKernSizeParam& param, bool m_large_group);
    static WorkspaceBundle get_bundle_stride(
            const NCBKernSizeParam& param, bool m_large_group);
    static void weight_flip_kern(
            const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
            const NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids);
    static void copy_padding_kern(
            const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
            const NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids);
    static void copy_padding_kern_stride(
            const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
            const NCBKernIndex& ncb_index, const CpuNDRange& workspace_ids);
    static void do_conv_kern(
            const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
            const NCBKernIndex& ncb_index, const kern_direct_conv_f32& fun,
            const CpuNDRange& workspace_ids);
    static void do_conv_kern_stride(
            const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
            const NCBKernIndex& ncb_index, const kern_direct_conv_f32_stride& fun,
            const CpuNDRange& workspace_ids);
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
