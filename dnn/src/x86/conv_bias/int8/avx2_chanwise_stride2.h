#pragma once

#include "src/x86/conv_bias/int8/common_helper.h"
#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
namespace avx2_chanwise_stride2 {
using conv_fun = std::function<void(
        const WorkspaceBundle& bundle, const NCBKernParam& kern_param,
        const NCBKernIndex& ncb_index)>;
SmallVector<NCBKern> get_kimpls(
        const NCBKernSizeParam& param, const WorkspaceBundle& bundle);

}  // namespace avx2_chanwise_stride2
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
