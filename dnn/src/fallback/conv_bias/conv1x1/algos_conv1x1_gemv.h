#pragma once

#include "megdnn/thin/small_vector.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {

class ConvBiasImpl::AlgoConv1x1Gemv final : public AlgoBase {
public:
    AlgoConv1x1Gemv() = default;

    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }

    const char* name() const override { return "CONV1x1_GEMV"; }

    bool usable(
            const NCBKernSizeParam& param,
            AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam& param) const override;

    bool is_preferred(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        auto support_data_type = static_cast<AlgoDataType>(
                static_cast<uint32_t>(AlgoDataType::FLOAT16) |
                static_cast<uint32_t>(AlgoDataType::FLOAT32) |
                static_cast<uint32_t>(AlgoDataType::INT8X8X16) |
                static_cast<uint32_t>(AlgoDataType::QINT8X8X32) |
                static_cast<uint32_t>(AlgoDataType::QUINT8X8X32));
        return {support_data_type, AlgoCategory::IM2COL};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_CONV1x1_GEMV)

protected:
    size_t get_oc_tile_size_heuristic(const NCBKernSizeParam& param) const;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
