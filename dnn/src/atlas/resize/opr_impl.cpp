#include "opr_impl.h"
#include "aclnnop/aclnn_resize.h"
#include "aclnnop/aclnn_upsample_bilinear_2d_backward.h"
#include "aclnnop/aclnn_upsample_nearest_2d_backward.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;

// *************************** Backward *************************** //
void ResizeBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    check_exec(diff.layout, grad.layout, workspace.size);
    auto handle = concrete_handle(this->handle());

    using InterpolationMode = param::Resize::InterpolationMode;
    megdnn_assert(
            param().format == param::Resize::Format::NCHW ||
                    param().format == param::Resize::Format::NHWC,
            "atlas resize backward only support NHWC and NCHW");
    // output size
    int64_t grad_size[4] =
            {static_cast<int64_t>(grad.layout.shape[0]),
             static_cast<int64_t>(grad.layout.shape[1]),
             static_cast<int64_t>(grad.layout.shape[2]),
             static_cast<int64_t>(grad.layout.shape[3])},
            diff_size[2];
    aclFormat format;
    switch (param().format) {
        case param::Resize::Format::NCHW: {
            diff_size[0] = static_cast<int64_t>(diff.layout.shape[2]);
            diff_size[1] = static_cast<int64_t>(diff.layout.shape[3]);
            format = aclFormat::ACL_FORMAT_NCHW;
            break;
        }
        case param::Resize::Format::NHWC: {
            diff_size[0] = static_cast<int64_t>(diff.layout.shape[1]);
            diff_size[1] = static_cast<int64_t>(diff.layout.shape[2]);
            format = aclFormat::ACL_FORMAT_NHWC;
            break;
        }
        default:
            break;
    }
    AclIntArray acl_grad_size(grad_size, 4), acl_diff_size(diff_size, 2);
    AclTensor acl_grad(grad, format), acl_diff(diff, format);
    switch (param().imode) {
        case InterpolationMode::NEAREST: {
            uint64_t ws_size = 0;
            aclOpExecutor* executor = nullptr;
            aclnn_check(aclnnUpsampleNearest2dBackwardGetWorkspaceSize(
                    acl_diff.get(), acl_diff_size.get(), acl_grad_size.get(), 0.0, 0.0,
                    acl_grad.get(), &ws_size, &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnUpsampleNearest2dBackward(
                    ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        case InterpolationMode::LINEAR: {
            uint64_t ws_size = 0;
            aclOpExecutor* executor = nullptr;
            aclnn_check(aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(
                    acl_diff.get(), acl_diff_size.get(), acl_grad_size.get(), false,
                    0.0, 0.0, acl_grad.get(), &ws_size, &executor));
            AclMem ws(ws_size, handle);
            aclnn_check(aclnnUpsampleBilinear2dBackward(
                    ws.ptr(), ws_size, executor, handle->stream()));
            break;
        }
        default:
            megdnn_throw("unsupported mode");
    }
}

// *************************** Forward *************************** //
void ResizeForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    //! FIXME: Although ascend official docs stated that resize supports non-contiguous
    //! tensor, but getWorkspace will crash when layout is not contiguous, may be
    //! related to aclCreateTensor's storageDim parameter.
    megdnn_assert_contiguous(src.layout);
    megdnn_assert_contiguous(dst.layout);
    auto&& _param = param();

    megdnn_assert(
            _param.format == Param::Format::NCHW, "ascend resize only support NCHW");
    aclFormat fmt = aclFormat::ACL_FORMAT_NCHW;
    AclTensor acl_src(src.raw_ptr(), src.layout, fmt);
    AclTensor acl_dst(dst.raw_ptr(), dst.layout, fmt);

    const char* imode = "";
    if (_param.imode == param::Resize::InterpolationMode::INTER_NEAREST) {
        imode = "nearest";
    } else if (_param.imode == param::Resize::InterpolationMode::INTER_LINEAR) {
        imode = "bilinear";
    } else {
        megdnn_assert(false, "ascend resize only support nearest and bilinear");
    }

    SmallVector<float> scale{
            static_cast<float>(dst.layout.shape[0]) / src.layout.shape[0],
            static_cast<float>(dst.layout.shape[1]) / src.layout.shape[1],
            static_cast<float>(dst.layout.shape[2]) / src.layout.shape[2],
            static_cast<float>(dst.layout.shape[3]) / src.layout.shape[3]};
    AclFloatArray acl_scale(scale);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnn_check(aclnnResizeGetWorkspaceSize(
            acl_src.get(), acl_scale.get(), imode, acl_dst.get(), &ws_size, &executor));
    auto ws = AclMem(ws_size, concrete_handle(handle()));
    aclnn_check(aclnnResize(
            ws.ptr(), ws_size, executor, concrete_handle(handle())->stream()));
}

// vim: syntax=cpp.doxygen