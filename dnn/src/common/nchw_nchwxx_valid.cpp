#include "src/common/nchw_nchwxx_valid.h"
#include "megdnn/oprs/nn.h"
using namespace megdnn;
namespace {
using NchwNchwxxFuncInterface = std::function<bool(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const ConvBiasForward::BiasMode bias_mode,
        const param::ConvBias::NonlineMode nonline_mode)>;
static SmallVector<NchwNchwxxFuncInterface> g_func_vec {
    nchw_nchwxx_valid<NchwNchwxxType::NCHW44_FP32>,
            nchw_nchwxx_valid<NchwNchwxxType::NCHW44_INT8>,
            nchw_nchwxx_valid<NchwNchwxxType::NCHW44_INT8_INT8_INT16>,
            nchw_nchwxx_valid<NchwNchwxxType::NCHW44_INT8_DOT>,
            nchw_nchwxx_valid<NchwNchwxxType::NCHW88>,
#if !MEGDNN_DISABLE_FLOAT16
            nchw_nchwxx_valid<NchwNchwxxType::NCHW88_FP16>,
#endif
};
}  // namespace
bool ConvBiasForward::is_nchw_nchwxx_optimized(
        const DTypeEnum src_dtype, const DTypeEnum filter_dtype,
        const DTypeEnum dst_dtype,
        const ConvolutionBase<param::Convolution>::CanonizedFilterMeta& fm,
        const ConvBiasForward::BiasMode bias_mode,
        const param::ConvBias::NonlineMode nonline_mode) {
    for (auto& func : g_func_vec) {
        if (func(src_dtype, filter_dtype, dst_dtype, fm, bias_mode, nonline_mode)) {
            return true;
        }
    }
    return false;
}