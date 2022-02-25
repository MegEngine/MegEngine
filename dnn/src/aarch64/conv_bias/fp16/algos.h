#pragma once

#include "src/aarch64/conv_bias/opr_impl.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace aarch64 {
/* ===================== stride-2 algo ===================== */
class ConvBiasImpl::AlgoF16DirectStride2 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "ARMV8F16STRD2"; }
    bool usable(
            const NCBKernSizeParam& param,
            AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(AARCH64_DIRECT_STRD2_FP16)
};
}  // namespace aarch64
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
