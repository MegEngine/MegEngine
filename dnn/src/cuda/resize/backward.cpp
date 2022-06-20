#include "src/cuda/resize/opr_impl.h"

#include "src/cuda/resize/common.h"
#include "src/cuda/resize/helper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void ResizeBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    check_exec(diff.layout, grad.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    bool is_nhwc = param().format == param::Resize::Format::NHWC;
    size_t N, C, IH, IW, OH, OW;
    if (is_nhwc) {
        if (param().imode != Param::InterpolationMode::LINEAR &&
            is_nhwc_contig_wc(grad.layout)) {
            megdnn_assert(
                    0,
                    "unsupport mode in resizeBackward, only support param().imode = "
                    "LINEAR");
        }
        N = grad.layout.shape[0];
        C = grad.layout.shape[3];
        IH = grad.layout.shape[1];
        IW = grad.layout.shape[2];
        OH = diff.layout.shape[1];
        OW = diff.layout.shape[2];
    } else {
        N = grad.layout.shape[0], C = grad.layout.shape[1], IH = grad.layout.shape[2],
        IW = grad.layout.shape[3], OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    }
    size_t max_batch_x_channel = max_batch_x_channel_size();
    size_t max_batch_size = max_batch_x_channel / C;
    while (N > 0) {
        size_t curr_batch_size = N > max_batch_size ? max_batch_size : N;
        switch (grad.layout.dtype.enumv()) {
#define cb(_t)                                                                 \
    case DTypeTrait<_t>::enumv: {                                              \
        typedef DTypeTrait<_t>::ctype ct;                                      \
        ct* diff_ptr = diff.ptr<ct>();                                         \
        ct* grad_ptr = grad.ptr<ct>();                                         \
        resize::backward_data_proxy(                                           \
                is_nhwc, resize::get_imode(param().imode), diff_ptr, grad_ptr, \
                curr_batch_size, C, IH, IW, OH, OW, stream);                   \
        if (N <= max_batch_size) {                                             \
            return;                                                            \
        } else {                                                               \
            N -= max_batch_size;                                               \
            diff_ptr += curr_batch_size * diff.layout.stride[0];               \
            grad_ptr += curr_batch_size * grad.layout.stride[0];               \
        }                                                                      \
        break;                                                                 \
    }
            cb(megdnn::dtype::Float32);
            DNN_INC_FLOAT16(cb(megdnn::dtype::Float16));
            default:
                megdnn_throw(ssprintf(
                        "unsupported dtype: %s in resize backward",
                        grad.layout.dtype.name()));
        }
#undef cb
    }
}

size_t ResizeBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& diff, const TensorLayout& grad) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(grad);
    return 0;
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
