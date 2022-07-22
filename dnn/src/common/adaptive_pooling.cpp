#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"

#include "src/common/utils.h"
namespace megdnn {

param::Pooling AdaptivePoolingBase::deduce_pooling_param(
        const TensorLayout& src, const TensorLayout& dst) {
    auto param_format = param().format;
    size_t IH, IW, OH, OW;
    if (param_format == param::AdaptivePooling::Format::NCHW ||
        param_format == param::AdaptivePooling::Format::NCHW44 ||
        param_format == param::AdaptivePooling::Format::NCHW88) {
        IH = src.shape[2];
        IW = src.shape[3];
        OH = dst.shape[2];
        OW = dst.shape[3];
    } else if (param_format == param::AdaptivePooling::Format::NHWC) {
        IH = src.shape[1];
        IW = src.shape[2];
        OH = dst.shape[1];
        OW = dst.shape[2];
    } else {
        megdnn_throw(
                "AdaptivePooling only support NCHW or NHWC or NCHW44 or NCHW88 format");
    }

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
