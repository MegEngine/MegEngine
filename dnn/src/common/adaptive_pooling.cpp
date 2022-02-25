#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

#include "src/common/utils.h"
namespace megdnn {

param::Pooling AdaptivePoolingBase::deduce_pooling_param(
        const TensorLayout& src, const TensorLayout& dst) {
    megdnn_assert(param().format == param::AdaptivePooling::Format::NCHW);
    size_t IH = src.shape[2], IW = src.shape[3], OH = dst.shape[2], OW = dst.shape[3];

    param::Pooling ret;
    ret.mode = param().mode;
    ret.format = param().format;
    ret.pad_h = ret.pad_w = 0;
    ret.stride_h = floor(IH / OH);
    ret.stride_w = floor(IW / OW);
    ret.window_h = IH - (OH - 1) * ret.stride_h;
    ret.window_w = IW - (OW - 1) * ret.stride_w;

    return ret;
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen
