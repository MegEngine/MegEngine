#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"

namespace megdnn {
namespace arm_common {
namespace channel_wise_nchw44_8x8x16 {

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;
using NCBKernIndex = fallback::ConvBiasImpl::NCBKernIndex;

using conv_fun = std::function<void(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index)>;

namespace stride1 {

bool is_available(const NCBKernSizeParam& param);

WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

template <size_t filter, BiasMode bias_mode>
void do_conv_kern(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index);

SmallVector<ConvBiasImpl::NCBKern> get_kimpls(const NCBKernSizeParam& param);
}  // namespace stride1

namespace stride2 {
bool is_available(const NCBKernSizeParam& param);

WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

template <size_t filter, BiasMode bias_mode>
void do_conv_kern(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index);

SmallVector<ConvBiasImpl::NCBKern> get_kimpls(const NCBKernSizeParam& param);

}  // namespace stride2
}  // namespace channel_wise_nchw44_8x8x16
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
