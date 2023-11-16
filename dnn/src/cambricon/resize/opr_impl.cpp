#include "src/cambricon/resize/opr_impl.h"
#include <numeric>
#include "cnnl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

struct ResizeCnnlDescs {
    CnnlTensorDescriptor input_desc, output_desc;
    cnnlInterpMode_t imode;
    cnnlInterpBackwardMode_t bmode;
    bool align_corners = false;
    bool align_center = false;
    using InterpolationMode = param::Resize::InterpolationMode;
    ResizeCnnlDescs(
            const TensorLayout& src, const TensorLayout& dst, InterpolationMode& mode) {
        cnnlTensorLayout_t layout = CNNL_LAYOUT_NHWC;
        input_desc.set(src, layout);
        output_desc.set(dst, layout);
        switch (mode) {
            case InterpolationMode::INTER_LINEAR:
                if (src.ndim == 3) {
                    imode = CNNL_INTERP_LINEAR;
                    bmode = CNNL_INTERP_BACKWARD_LINEAR;
                } else if (src.ndim == 4) {
                    imode = CNNL_INTERP_BILINEAR;
                    bmode = CNNL_INTERP_BACKWARD_BILINEAR;
                }
                align_center = true;
                break;
            case InterpolationMode::NEAREST:
                megdnn_assert(src.ndim == 3 || src.ndim == 4);
                imode = CNNL_INTERP_NEAREST;
                bmode = CNNL_INTERP_BACKWARD_NEAREST;
                break;
            case InterpolationMode::INTER_CUBIC:
                imode = CNNL_INTERP_BICUBIC;
                bmode = CNNL_INTERP_BACKWARD_BICUBIC;
                align_center = true;
                break;
            default:
                megdnn_throw("unsupported mode in ResizeImpl");
                break;
        }
    }
};

void ResizeImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    auto handle = cnnl_handle(this->handle());
    check_exec(src.layout, dst.layout, workspace.size);
    // cnnl interp only support float and half
    check_dtype_float(src.layout.dtype.enumv());
    megdnn_assert(
            param().format == param::Resize::Format::NHWC,
            "CNNL resize only support NHWC when input tensor is 4D");
    ResizeCnnlDescs descs(src.layout, dst.layout, param().imode);
    cnnl_check(cnnlInterp_v2(
            handle, descs.align_corners, descs.align_center, descs.imode, NULL, true,
            descs.input_desc.desc(), src.raw_ptr(), descs.output_desc.desc(),
            dst.raw_ptr()));
}

// ***************************Backward*************************** //
void ResizeBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    check_exec(diff.layout, grad.layout, workspace.size);
    check_dtype_float(diff.layout.dtype.enumv());
    megdnn_assert(
            param().format == param::Resize::Format::NHWC,
            "CNNL resize only support NHWC");

    auto handle = cnnl_handle(this->handle());
    ResizeCnnlDescs descs(diff.layout, grad.layout, param().imode);
    cnnl_check(cnnlInterpBackward_v2(
            handle, descs.align_corners, descs.align_center, descs.bmode, NULL, true,
            descs.input_desc.desc(), diff.raw_ptr(), descs.output_desc.desc(),
            grad.raw_ptr()));
}

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
