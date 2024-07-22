#include <memory>

#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_batch_norm.h"
#include "aclnnop/aclnn_batch_norm_backward.h"
#include "aclnnop/aclnn_batch_norm_elemt.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "aclnnop/aclnn_sign.h"
#include "opr_impl.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;
using Mode = param::Elemwise::Mode;

void BNForwardImpl::exec_infer(
        _megdnn_tensor_in src, _megdnn_tensor_in bn_scale, _megdnn_tensor_in bn_bias,
        _megdnn_tensor_in mean, _megdnn_tensor_in variance, _megdnn_tensor_out dst) {
    megdnn_assert(
            m_param.param_dim == param::BN::ParamDim::DIM_1C11,
            "atlas batchnorm only support (1,C,1,1)");

    auto handle = concrete_handle(this->handle());
    AclTensor acl_input(src, ACL_FORMAT_NCHW);
    AclTensor acl_output(dst, ACL_FORMAT_NCHW);

    TensorLayout flattened_weight_lyt = bn_scale.layout.collapse_contiguous();
    AclTensor acl_weight(bn_scale.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_bias(bn_bias.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_mean(mean.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_var(variance.raw_ptr(), flattened_weight_lyt);

    AclMem invstd_mem(flattened_weight_lyt.span().dist_byte(), handle);
    TensorND invstd(invstd_mem.ptr(), flattened_weight_lyt);
    AclTensor acl_invstd(invstd);

    double eps = m_param.epsilon, alpha = 1.0;
    AclScalar acl_eps(eps), acl_aplha(alpha);

    // perform var + eps
    {
        uint64_t adds_ws_size = 0;
        aclOpExecutor* adds_executor = nullptr;
        aclnn_check(aclnnAddsGetWorkspaceSize(
                acl_var.get(), acl_eps.get(), acl_aplha.get(), acl_invstd.get(),
                &adds_ws_size, &adds_executor));
        AclMem adds_ws(adds_ws_size, handle);
        aclnn_check(aclnnAdds(
                adds_ws.ptr(), adds_ws_size, adds_executor, handle->stream()));
    }

    // get inv std
    {
        uint64_t rsqrt_ws_size = 0;
        aclOpExecutor* rsqrt_executor = nullptr;
        aclnn_check(aclnnInplaceRsqrtGetWorkspaceSize(
                acl_invstd.get(), &rsqrt_ws_size, &rsqrt_executor));
        AclMem rsqrt_ws(rsqrt_ws_size, handle);
        aclnn_check(aclnnInplaceRsqrt(
                rsqrt_ws.ptr(), rsqrt_ws_size, rsqrt_executor, handle->stream()));
    }

    // perform batch_norm
    {
        uint64_t bn_ws_size = 0;
        aclOpExecutor* bn_executor = nullptr;
        aclnn_check(aclnnBatchNormElemtGetWorkspaceSize(
                acl_input.get(), acl_weight.get(), acl_bias.get(), acl_mean.get(),
                acl_invstd.get(), eps, acl_output.get(), &bn_ws_size, &bn_executor));
        AclMem bn_ws(bn_ws_size, handle);
        aclnn_check(aclnnBatchNormElemt(
                bn_ws.ptr(), bn_ws_size, bn_executor, handle->stream()));
    }
}

void BNForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in bn_scale, _megdnn_tensor_in bn_bias,
        _megdnn_tensor_out mean, _megdnn_tensor_out variance,
        _megdnn_tensor_out batch_mean, _megdnn_tensor_out batch_inv_variance,
        _megdnn_tensor_out, _megdnn_tensor_out dst, _megdnn_workspace) {
    if (m_param.fwd_mode == Param::FwdMode::INFERENCE) {
        exec_infer(src, bn_scale, bn_bias, mean, variance, dst);
        return;
    }
    megdnn_assert(
            m_param.param_dim == param::BN::ParamDim::DIM_1C11,
            "atlas batchnorm only support (1,C,1,1)");

    auto handle = concrete_handle(this->handle());
    AclTensor acl_input(src, ACL_FORMAT_NCHW);
    AclTensor acl_output(dst, ACL_FORMAT_NCHW);

    TensorLayout flattened_weight_lyt = bn_scale.layout.collapse_contiguous();
    AclTensor acl_weight(bn_scale.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_bias(bn_bias.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_running_mean(mean.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_running_var(variance.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_save_mean(batch_mean.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_save_invvar(batch_inv_variance.raw_ptr(), flattened_weight_lyt);

    bool training = true;
    double eps = m_param.epsilon;
    double momentum = m_param.avg_factor;

    {
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        aclnn_check(aclnnBatchNormGetWorkspaceSize(
                acl_input.get(), acl_weight.get(), acl_bias.get(),
                acl_running_mean.get(), acl_running_var.get(), training, momentum, eps,
                acl_output.get(), acl_save_mean.get(), acl_save_invvar.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnBatchNorm(ws.ptr(), ws_size, executor, handle->stream()));
    }

    // atlas compute std, megengine need (1 / sqrt(std))
    {
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        aclnn_check(aclnnInplaceRsqrtGetWorkspaceSize(
                acl_save_invvar.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(aclnnInplaceRsqrt(ws.ptr(), ws_size, executor, handle->stream()));
    }
}

void BNBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in dy, _megdnn_tensor_in saved_batch_mean,
        _megdnn_tensor_in saved_batch_inv_variance, _megdnn_tensor_in bn_scale,
        _megdnn_tensor_in, _megdnn_tensor_out d_bn_scale, _megdnn_tensor_out d_bn_bias,
        _megdnn_tensor_out dx, _megdnn_workspace) {
    using FMode = param::BN::FwdMode;
    auto mode = m_param.fwd_mode;
    megdnn_assert(
            m_param.param_dim == param::BN::ParamDim::DIM_1C11,
            "atlas batchnorm only support (1,C,1,1)");
    megdnn_assert(mode == FMode::TRAINING, "only support training now");

    auto handle = concrete_handle(this->handle());
    AclTensor acl_input(x, ACL_FORMAT_NCHW);
    AclTensor acl_grad_out(dy, ACL_FORMAT_NCHW);
    AclTensor acl_grad_inp(dx, ACL_FORMAT_NCHW);

    TensorLayout flattened_weight_lyt = bn_scale.layout.collapse_contiguous();
    AclTensor acl_save_mean(saved_batch_mean.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_save_invvar(saved_batch_inv_variance.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_weight(bn_scale.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_grad_weight(d_bn_scale.raw_ptr(), flattened_weight_lyt);
    AclTensor acl_grad_bias(d_bn_bias.raw_ptr(), flattened_weight_lyt);

    // megengine give inv_var, ascend need std = (1 / inv_var^2). in dnn test, inv_var
    // can be negative but std = (1 / inv_var^2) is always positive. we solve this
    // problem at last of this function
    AclMem acl_save_invstd_buf(flattened_weight_lyt.access_bytes(), handle);
    AclTensor acl_save_invstd(acl_save_invstd_buf.ptr(), flattened_weight_lyt);
    {
        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;
        AclScalar exponent(-2.0f);
        aclnn_check(aclnnPowTensorScalarGetWorkspaceSize(
                acl_save_invvar.get(), exponent.get(), acl_save_invstd.get(), &ws_size,
                &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnPowTensorScalar(ws.ptr(), ws_size, executor, handle->stream()));
    }

    {
        bool training = (m_param.fwd_mode == param::BN::FwdMode::TRAINING);
        double eps = m_param.epsilon;
        size_t elems = d_bn_bias.layout.total_nr_elems();

        uint64_t ws_size = 0;
        aclOpExecutor* executor = nullptr;

        SmallVector<uint8_t> mask(elems, 1);
        AclBoolArray output_mask(mask);

        aclnn_check(aclnnBatchNormBackwardGetWorkspaceSize(
                acl_grad_out.get(), acl_input.get(), acl_weight.get(), nullptr, nullptr,
                acl_save_mean.get(), acl_save_invstd.get(), training, eps,
                output_mask.get(), acl_grad_inp.get(), acl_grad_weight.get(),
                acl_grad_bias.get(), &ws_size, &executor));
        AclMem ws(ws_size, handle);
        aclnn_check(
                aclnnBatchNormBackward(ws.ptr(), ws_size, executor, handle->stream()));
    }

    // in dnn test, the elements of 'saved_batch_inv_variance' can be negative because
    // it is generated as by RandomNumGenerator rather than computed by BNForward. At
    // this time, the std = (1 / inv_var^2) is still positive, which may cause the sign
    // problem of dx and dweight, we can solve the sign problem as below.
    // But the saved_batch_inv_variance computed by BNForward is always positive. This
    // code is only used to pass dnn test. It should be removed to improve performance
    {
        AclMem acl_sign_buf(flattened_weight_lyt.access_bytes(), handle);
        AclTensor acl_sign(acl_sign_buf.ptr(), flattened_weight_lyt);

        uint64_t sign_ws_size = 0;
        aclOpExecutor* sign_executor = nullptr;
        aclnn_check(aclnnSignGetWorkspaceSize(
                acl_save_invvar.get(), acl_sign.get(), &sign_ws_size, &sign_executor));
        AclMem sign_ws(sign_ws_size, handle);
        aclnn_check(aclnnSign(
                sign_ws.ptr(), sign_ws_size, sign_executor, handle->stream()));

        uint64_t weight_ws_size = 0;
        aclOpExecutor* weight_executor = nullptr;
        aclnn_check(aclnnInplaceMulGetWorkspaceSize(
                acl_grad_weight.get(), acl_sign.get(), &weight_ws_size,
                &weight_executor));
        AclMem weight_ws(weight_ws_size, handle);
        aclnn_check(aclnnInplaceMul(
                weight_ws.ptr(), weight_ws_size, weight_executor, handle->stream()));

        AclTensor acl_sign_unflatten(acl_sign_buf.ptr(), bn_scale.layout);
        uint64_t dx_ws_size = 0;
        aclOpExecutor* dx_executor = nullptr;
        aclnn_check(aclnnInplaceMulGetWorkspaceSize(
                acl_grad_inp.get(), acl_sign_unflatten.get(), &dx_ws_size,
                &dx_executor));
        AclMem dx_ws(dx_ws_size, handle);
        aclnn_check(aclnnInplaceMul(
                dx_ws.ptr(), dx_ws_size, dx_executor, handle->stream()));
    }
}