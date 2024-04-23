#include <memory>

#include "aclnnop/aclnn_batch_norm.h"
#include "aclnnop/aclnn_batch_norm_backward.h"
#include "aclnnop/aclnn_pow.h"
#include "aclnnop/aclnn_rsqrt.h"
#include "opr_impl.h"
#include "src/atlas/atlas_wrapper.h"
#include "src/atlas/utils.h"

using namespace megdnn;
using namespace atlas;
using Mode = param::Elemwise::Mode;

TensorND reshape_tensor_to_C_with_param(
        const TensorND& in, HandleImpl* handle, param::BN& bn) {
    if (in.layout.ndim == 4) {
        int c_pos = 1;
        if (bn.param_dim == param::BN::ParamDim::DIM_111C) {
            c_pos = 3;
        }
        SmallVector<size_t> sizes;
        sizes.push_back(in.layout.shape[c_pos]);
        TensorShape dst_shape(sizes);
        TensorLayout dst_layout = in.layout.reshape(dst_shape);
        TensorND dst(in.raw_ptr(), dst_layout);
        auto relayout_opr = handle->create_operator<RelayoutForward>();
        relayout_opr->exec(in, dst);
        return dst;
    } else {
        return in;
    }
    return in;
}

TensorND BNForwardImpl::reshape_tensor_to_C(const TensorND& in) {
    auto handle = concrete_handle(this->handle());
    return reshape_tensor_to_C_with_param(in, handle, m_param);
}

TensorND BNBackwardImpl::reshape_tensor_to_C(const TensorND& in) {
    auto handle = concrete_handle(this->handle());
    return reshape_tensor_to_C_with_param(in, handle, m_param);
}

void BNForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in bn_scale, _megdnn_tensor_in bn_bias,
        _megdnn_tensor_out mean, _megdnn_tensor_out variance,
        _megdnn_tensor_out batch_mean, _megdnn_tensor_out batch_inv_variance,
        _megdnn_tensor_out, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    megdnn_assert(m_param.param_dim != param::BN::ParamDim::DIM_11HW);
    auto handle = concrete_handle(this->handle());
    AclTensor acl_input(src, ACL_FORMAT_NCHW),
            acl_weight(this->reshape_tensor_to_C(bn_scale)),
            acl_bias(this->reshape_tensor_to_C(bn_bias)),
            acl_runningMean(this->reshape_tensor_to_C(mean)),
            acl_runningVar(this->reshape_tensor_to_C(variance)),
            acl_saveMean(this->reshape_tensor_to_C(batch_mean)),
            acl_saveInvstd(this->reshape_tensor_to_C(batch_inv_variance)),
            acl_output(dst, ACL_FORMAT_NCHW);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    bool training = (m_param.fwd_mode == param::BN::FwdMode::TRAINING);
    double eps = m_param.epsilon;
    double momentum = m_param.avg_factor;
    aclnn_check(aclnnBatchNormGetWorkspaceSize(
            acl_input.get(), acl_weight.get(), acl_bias.get(), acl_runningMean.get(),
            acl_runningVar.get(), training, momentum, eps, acl_output.get(),
            acl_saveMean.get(), acl_saveInvstd.get(), &ws_size, &executor));
    AclMem ws(ws_size, handle);
    aclnn_check(aclnnBatchNorm(ws.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(aclnnInplaceRsqrtGetWorkspaceSize(
            acl_saveInvstd.get(), &ws_size, &executor));
    AclMem ws_2(ws_size, handle);
    aclnn_check(aclnnInplaceRsqrt(ws_2.ptr(), ws_size, executor, handle->stream()));
}

void BNBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in dy, _megdnn_tensor_in saved_batch_mean,
        _megdnn_tensor_in saved_batch_inv_variance, _megdnn_tensor_in bn_scale,
        _megdnn_tensor_in, _megdnn_tensor_out d_bn_scale, _megdnn_tensor_out d_bn_bias,
        _megdnn_tensor_out dx, _megdnn_workspace workspace) {
    megdnn_assert(m_param.param_dim != param::BN::ParamDim::DIM_11HW);
    auto handle = concrete_handle(this->handle());
    AclTensor acl_input(x, ACL_FORMAT_NCHW), acl_gradOut(dy, ACL_FORMAT_NCHW),
            acl_saveMean(this->reshape_tensor_to_C(saved_batch_mean)),
            acl_saveInvstd(this->reshape_tensor_to_C(saved_batch_inv_variance)),
            acl_weight(this->reshape_tensor_to_C(bn_scale)),
            acl_gradWeight(this->reshape_tensor_to_C(d_bn_scale)),
            acl_gradBias(this->reshape_tensor_to_C(d_bn_bias)),
            acl_gradInput(dx, ACL_FORMAT_NCHW);
    bool training = (m_param.fwd_mode == param::BN::FwdMode::TRAINING);
    double eps = m_param.epsilon;
    size_t elems = d_bn_bias.layout.total_nr_elems();
    std::unique_ptr<bool> bool_ptr(new bool[elems]);
    for (int idx = 0; idx < elems; idx++)
        bool_ptr.get()[idx] = true;
    AclBoolArray output_mask(bool_ptr.get(), elems);

    uint64_t ws_size = 0;
    aclOpExecutor* executor = nullptr;

    aclnn_check(aclnnInplaceRsqrtGetWorkspaceSize(
            acl_saveInvstd.get(), &ws_size, &executor));
    AclMem ws_2(ws_size, handle);
    aclnn_check(aclnnInplaceRsqrt(ws_2.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(aclnnBatchNormBackwardGetWorkspaceSize(
            acl_gradOut.get(), acl_input.get(), acl_weight.get(), nullptr, nullptr,
            acl_saveMean.get(), acl_saveInvstd.get(), training, eps, output_mask.get(),
            acl_gradInput.get(), acl_gradWeight.get(), acl_gradBias.get(), &ws_size,
            &executor));

    float exponent_value = 4.0f;
    AclScalar exponent(exponent_value);
    aclnn_check(aclnnInplacePowTensorScalarGetWorkspaceSize(
            acl_saveInvstd.get(), exponent.get(), &ws_size, &executor));
    AclMem ws_3(ws_size, handle);
    aclnn_check(aclnnInplacePowTensorScalar(
            ws_3.ptr(), ws_size, executor, handle->stream()));

    AclMem ws(ws_size, handle);
    aclnn_check(aclnnBatchNormBackward(ws.ptr(), ws_size, executor, handle->stream()));

    aclnn_check(aclnnInplaceRsqrtGetWorkspaceSize(
            acl_saveInvstd.get(), &ws_size, &executor));
    AclMem ws_4(ws_size, handle);
    aclnn_check(aclnnInplaceRsqrt(ws_4.ptr(), ws_size, executor, handle->stream()));
}