#include "src/fallback/conv_bias/gi/fp16/algos.h"

#if defined(GI_SUPPORT_F16)

#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/direct/multi_thread_common.h"
#include "src/fallback/conv_bias/gi/fp16/strategy.h"
#include "src/fallback/conv_bias/gi/postprocess_helper.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_winograd_fp16_nchw88)

using namespace megdnn;
using namespace fallback;

/* =================== AlgoFP16WinogradF43_8x8_NCHW88 ===================== */

bool ConvBiasImpl::AlgoFP16WinogradF43_8x8_NCHW88::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(
            megdnn_fallback_winograd_fp16_nchw88,
            midout_iv("AlgoFP16WinogradF43_8x8_NCHW88"_hash)) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = winograd::winograd_F43_mk8_f16_nchw88;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy, param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() ==
                       fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK &&
               param.filter_meta.format == param::ConvBias::Format::NCHW88 &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] == param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(
        AlgoFP16WinogradF43_8x8_NCHW88, winograd::winograd_F43_mk8_f16_nchw88,
        megdnn_fallback_winograd_fp16_nchw88, param::MatrixMul::Format::MK8);

/* =================== AlgoFP16WinogradF23_8x8_NCHW88 ===================== */

bool ConvBiasImpl::AlgoFP16WinogradF23_8x8_NCHW88::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(
            megdnn_fallback_winograd_fp16_nchw88,
            midout_iv("AlgoFP16WinogradF23_8x8_NCHW88"_hash)) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = winograd::winograd_F23_mk8_f16_nchw88;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy, param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() ==
                       fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK &&
               param.filter_meta.format == param::ConvBias::Format::NCHW88 &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] == param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(
        AlgoFP16WinogradF23_8x8_NCHW88, winograd::winograd_F23_mk8_f16_nchw88,
        megdnn_fallback_winograd_fp16_nchw88, param::MatrixMul::Format::MK8);

/* =================== AlgoFP16WinogradF63_8x8_NCHW88 ===================== */

bool ConvBiasImpl::AlgoFP16WinogradF63_8x8_NCHW88::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(
            megdnn_fallback_winograd_fp16_nchw88,
            midout_iv("AlgoFP16WinogradF63_8x8_NCHW88"_hash)) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = winograd::winograd_F63_mk8_f16_nchw88;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy, param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() ==
                       fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK &&
               param.filter_meta.format == param::ConvBias::Format::NCHW88 &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] == param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(
        AlgoFP16WinogradF63_8x8_NCHW88, winograd::winograd_F63_mk8_f16_nchw88,
        megdnn_fallback_winograd_fp16_nchw88, param::MatrixMul::Format::MK8);

#endif
// vim: syntax=cpp.doxygen
