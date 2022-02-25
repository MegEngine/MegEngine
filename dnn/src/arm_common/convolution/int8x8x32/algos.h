#pragma once

#include "src/arm_common/convolution/opr_impl.h"

namespace megdnn {
namespace arm_common {

#if MGB_ENABLE_DOT
/* ===================== ConvolutionBackwardData ===================== */

class ConvolutionBackwardDataImpl::AlgoSdot8DirectStride1 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "AARCH32_I8x8x32_DECONV_STRIDE1"; }

    bool usable(fallback::ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param)
            const override;

    size_t get_workspace(
            fallback::ConvolutionBackwardDataImpl*,
            const NCBKernSizeParam& param) const override;

    ncb_kern_t dispatch_kern(
            fallback::ConvolutionBackwardDataImpl*,
            const NCBKernSizeParam&) const override;
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD1_DOT_INT8X8X32)
};

class ConvolutionBackwardDataImpl::AlgoSdot8DirectStride2 final : public AlgoBase {
public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "AARCH32_I8x8x32_DECONV_STRIDE2"; }

    bool usable(fallback::ConvolutionBackwardDataImpl*, const NCBKernSizeParam& param)
            const override;

    size_t get_workspace(
            fallback::ConvolutionBackwardDataImpl*,
            const NCBKernSizeParam& param) const override;

    ncb_kern_t dispatch_kern(
            fallback::ConvolutionBackwardDataImpl*,
            const NCBKernSizeParam&) const override;
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD2_DOT_INT8X8X32)
};

#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
