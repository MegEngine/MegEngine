#pragma once
#include "src/fallback/general_intrinsic/gi_common.h"

#if defined(GI_SUPPORT_F16)

#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"

namespace megdnn {
namespace fallback {
class ConvBiasImpl::AlgoFP16WinogradF43_8x8_NCHW88 final : public AlgoBase {
public:
    AlgoFP16WinogradF43_8x8_NCHW88(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo, uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {8, 4, m_tile_size, 3},
                    param::ConvBias::Format::NCHW88);
        }
        return m_name.c_str();
    }
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT16);
    MEGDNN_DECL_ALGO_TYPE(GI_COMMON_WINOGRAD_F43_8X8_NCHW88_F16)
};

}  // namespace fallback
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
