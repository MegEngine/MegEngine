#pragma once

#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
namespace direct_conv_avx2_stride2 {

using NCBKern = fallback::ConvBiasImpl::NCBKern;
using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;

SmallVector<NCBKern> get_kimpls(
        const NCBKernSizeParam& param, const WorkspaceBundle& bundle);

}  // namespace direct_conv_avx2_stride2
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
