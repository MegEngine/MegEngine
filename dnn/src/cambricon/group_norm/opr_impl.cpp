#include "src/cambricon/group_norm/opr_impl.h"
#include <algorithm>
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/handle.h"
#include "src/cambricon/utils.h"
#include "src/cambricon/utils.mlu.h"

using namespace megdnn;
using namespace cambricon;

namespace megdnn {
namespace cambricon {

size_t GroupNormForwardImpl::get_workspace_in_bytes(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        const TensorLayout& dst, const TensorLayout& mean, const TensorLayout& rstd) {
    return get_workspace_bundle(data, nullptr).total_size_in_bytes();
}

WorkspaceBundle GroupNormForwardImpl::get_workspace_bundle(
        const TensorLayout& data, void* raw_ptr) {
    auto handle = concrete_handle(this->handle());
    int group_num = static_cast<int>(param().group);
    size_t channel = data.shape[1];
    size_t workspace_size = 0;
    CnnlTensorDescriptor x_desc;
    x_desc.set(data, CNNL_LAYOUT_NCHW);
    cnnl_check(cnnlGetGroupNormForwardWorkspaceSize(
            handle->cnnl_handle(), group_num, x_desc.desc(), &workspace_size));
    SmallVector<size_t> sizes_in_bytes;
    sizes_in_bytes.push_back(workspace_size);
    return {raw_ptr, sizes_in_bytes, handle->alignment_requirement()};
}

void GroupNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    check_exec(
            data.layout, weight.layout, bias.layout, dst.layout, mean.layout,
            rstd.layout, workspace.size);
    megdnn_assert(data.layout.dtype == dtype::Float32());
    auto handle = concrete_handle(this->handle());
    auto bundle = get_workspace_bundle(data.layout, workspace.raw_ptr);
    size_t channel = data.layout.shape[1];
    void *weight_ptr = nullptr, *bias_ptr = nullptr;
    CnnlTensorDescriptor x_desc, scale_bias_desc, y_desc, mean_rstd_desc;
    x_desc.set(data.layout, CNNL_LAYOUT_NCHW);
    if (param().affine) {
        scale_bias_desc.set(weight.layout, CNNL_LAYOUT_ARRAY);
        weight_ptr = weight.raw_ptr();
        bias_ptr = bias.raw_ptr();
    }
    y_desc.set(dst.layout, CNNL_LAYOUT_NCHW);
    mean_rstd_desc.set(mean.layout, CNNL_LAYOUT_ARRAY);
    cnnl_check(cnnlGroupNormForward_v3(
            handle->cnnl_handle(), param().eps, static_cast<int>(param().group),
            x_desc.desc(), data.raw_ptr(), scale_bias_desc.desc(), weight_ptr, bias_ptr,
            bundle.get(0), bundle.get_size(0), y_desc.desc(), dst.raw_ptr(),
            mean_rstd_desc.desc(), mean.raw_ptr(), rstd.raw_ptr()));
}

size_t GroupNormBackwardImpl::get_workspace_in_bytes(
        const TensorLayout&, const TensorLayout& data, const TensorLayout&,
        const TensorLayout&, const TensorLayout& rstd, const TensorLayout&,
        const TensorLayout&, const TensorLayout&) {
    size_t N = data.shape[0];
    size_t C = data.shape[1];
    size_t G = rstd.shape[1];
    return get_workspace_bundle(N, C, G, data.dtype.size()).total_size_in_bytes();
}

WorkspaceBundle GroupNormBackwardImpl::get_workspace_bundle(
        size_t N, size_t C, size_t G, size_t dtype_size, void* raw_ptr) {
    auto handle = concrete_handle(this->handle());
    size_t workspace_size = 0;
    cnnl_check(cnnlGetGroupNormBackwardWorkspaceSize(
            handle->cnnl_handle(), N * C, &workspace_size));
    return {raw_ptr,
            {
                    workspace_size,
            },
            handle->alignment_requirement()};
}

void GroupNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
    megdnn_assert(check_dtype_float(data.layout.dtype.enumv()));

    auto handle = concrete_handle(this->handle());
    size_t N = data.layout.shape[0];
    size_t C = data.layout.shape[1];
    size_t G = rstd.layout.shape[1];
    auto bundle = get_workspace_bundle(N, C, G, data.layout.dtype.size());
    bundle.set(workspace.raw_ptr);
    CnnlTensorDescriptor x_desc, diff_z_desc, gamma_desc, mean_desc, rstd_desc,
            diff_x_desc, diff_scale_desc, diff_bias_desc;
    diff_z_desc.set(diff.layout, CNNL_LAYOUT_NCHW);
    x_desc.set(data.layout, CNNL_LAYOUT_NCHW);
    mean_desc.set(mean.layout, CNNL_LAYOUT_ARRAY);
    rstd_desc.set(rstd.layout, CNNL_LAYOUT_ARRAY);
    diff_x_desc.set(ddata.layout, CNNL_LAYOUT_NCHW);
    void *dweight_ptr = nullptr, *dbias_ptr = nullptr, *weight_ptr = nullptr;
    if (param().affine) {
        diff_scale_desc.set(dweight.layout, CNNL_LAYOUT_ARRAY);
        diff_bias_desc.set(dbias.layout, CNNL_LAYOUT_ARRAY);
        gamma_desc.set(weight.layout, CNNL_LAYOUT_ARRAY);
        dweight_ptr = dweight.raw_ptr();
        dbias_ptr = dbias.raw_ptr();
        weight_ptr = weight.raw_ptr();
    }
    cnnl_check(cnnlGroupNormBackward(
            handle->cnnl_handle(), x_desc.desc(), data.raw_ptr(), diff_z_desc.desc(),
            diff.raw_ptr(), gamma_desc.desc(), weight_ptr, mean_desc.desc(),
            mean.raw_ptr(), rstd_desc.desc(), rstd.raw_ptr(), param().group,
            diff_x_desc.desc(), ddata.raw_ptr(), diff_scale_desc.desc(), dweight_ptr,
            diff_bias_desc.desc(), dbias_ptr, bundle.get(0), bundle.get_size(0)));
}

}  // namespace cambricon
}  // namespace megdnn
// vim: syntax=cpp.doxygen
