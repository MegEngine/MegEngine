#include <algorithm>

#include "cnnl.h"
#include "src/cambricon/batch_normalization/opr_impl.h"
#include "src/cambricon/cnnl_wrapper/cnnl_op_descriptor.h"
#include "src/cambricon/cnnl_wrapper/cnnl_tensor_descriptor.h"
#include "src/cambricon/utils.h"

namespace megdnn {
namespace cambricon {

size_t BNForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&) {
    auto handle = cnnl_handle(this->handle());
    CnnlTensorDescriptor input_desc;
    input_desc.set(src, CNNL_LAYOUT_NHWC);
    size_t bn_workspace_size = 0;
    cnnl_check(cnnlGetBatchNormForwardWorkspaceSize(
            handle, input_desc.desc(), &bn_workspace_size));
    return bn_workspace_size;
}

void BNForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in bn_scale, _megdnn_tensor_in bn_bias,
        _megdnn_tensor_out mean, _megdnn_tensor_out variance,
        _megdnn_tensor_out batch_mean, _megdnn_tensor_out batch_inv_variance,
        _megdnn_tensor_out reserve, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(
            src.layout, bn_scale.layout, bn_bias.layout, mean.layout, variance.layout,
            batch_mean.layout, batch_inv_variance.layout, dst.layout, workspace.size,
            reserve.layout.access_bytes());
    megdnn_assert(m_param.param_dim == param::BN::ParamDim::DIM_111C);
    // only support NHWC
    auto handle = cnnl_handle(this->handle());

    cnnlTensorLayout_t cnnl_layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor x_desc, y_desc, weight_bias_mean_var_desc;
    x_desc.set(src.layout, cnnl_layout);
    y_desc.set(dst.layout, cnnl_layout);
    std::vector<size_t> dims = {src.layout.shape[src.layout.ndim - 1]};
    weight_bias_mean_var_desc.set(
            1, dims, convert_to_cnnl_datatype(bn_scale.layout.dtype.enumv()),
            CNNL_LAYOUT_ARRAY);

    CnnlActivationDescriptor activation_desc;
    activation_desc.set(
            CNNL_ACTIVATION_IDENTITY, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);

    switch (m_param.fwd_mode) {
        case param::BN::FwdMode::TRAINING:
            cnnl_check(cnnlBatchNormForwardTrainingV2(
                    handle, activation_desc.desc(), CNNL_BATCHNORM_SPATIAL,
                    CNNL_BATCHNORM_OPS_BN, nullptr, nullptr,  // alpha, beta
                    x_desc.desc(), src.raw_ptr(),             // xDesc & x
                    nullptr, nullptr,                         // zDesc & z
                    weight_bias_mean_var_desc.desc(), bn_scale.raw_ptr(),
                    bn_bias.raw_ptr(),                   // sclae & bias
                    mean.raw_ptr(), variance.raw_ptr(),  // moving_mean & moving_var
                    m_param.epsilon,
                    m_param.avg_factor,            // momentum
                    y_desc.desc(), dst.raw_ptr(),  // yDesc & y
                    batch_mean.raw_ptr(),
                    batch_inv_variance.raw_ptr(),  // saved_mean, saved_invstd
                    workspace.raw_ptr, workspace.size, reserve.raw_ptr(),
                    reserve.layout.access_bytes()));
            break;
        case param::BN::FwdMode::INFERENCE:
            cnnl_check(cnnlBatchNormForwardInferenceV2(
                    handle, activation_desc.desc(), CNNL_BATCHNORM_SPATIAL,
                    CNNL_BATCHNORM_OPS_BN, nullptr, nullptr,  // alpha, beta
                    x_desc.desc(), src.raw_ptr(),             // xDesc & x
                    weight_bias_mean_var_desc.desc(), bn_scale.raw_ptr(),
                    bn_bias.raw_ptr(),                   // sclae & bias
                    nullptr, nullptr,                    // zDesc & z
                    mean.raw_ptr(), variance.raw_ptr(),  // moving_mean & moving_var
                    m_param.epsilon, y_desc.desc(), dst.raw_ptr()  // yDesc & y
                    ));
            break;
        default:
            megdnn_throw("Unknown forward mode type of batch normalization.");
    }
}

size_t BNBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout&, const TensorLayout&) {
    auto handle = cnnl_handle(this->handle());
    CnnlTensorDescriptor input_desc;
    input_desc.set(x, CNNL_LAYOUT_NHWC);
    size_t bn_workspace_size = 0;
    cnnl_check(cnnlGetBatchNormBackwardWorkspaceSize(
            handle, input_desc.desc(), &bn_workspace_size));
    return bn_workspace_size;
}

void BNBackwardImpl::exec(
        _megdnn_tensor_in x, _megdnn_tensor_in dy, _megdnn_tensor_in saved_batch_mean,
        _megdnn_tensor_in saved_batch_inv_variance, _megdnn_tensor_in bn_scale,
        _megdnn_tensor_in reserve, _megdnn_tensor_out d_bn_scale,
        _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx,
        _megdnn_workspace workspace) {
    check_exec(
            x.layout, dy.layout, saved_batch_mean.layout,
            saved_batch_inv_variance.layout, bn_scale.layout, d_bn_scale.layout,
            d_bn_bias.layout, dx.layout, workspace.size, reserve.layout.access_bytes());
    megdnn_assert(m_param.param_dim == param::BN::ParamDim::DIM_111C);
    auto handle = cnnl_handle(this->handle());

    cnnlTensorLayout_t cnnl_layout = CNNL_LAYOUT_NHWC;
    CnnlTensorDescriptor x_desc, dy_desc, dx_desc, filter_bias_mean_var_desc;
    x_desc.set(x.layout, cnnl_layout);
    dy_desc.set(dy.layout, cnnl_layout);
    dx_desc.set(x.layout, cnnl_layout);
    std::vector<size_t> dims = {x.layout.shape[x.layout.ndim - 1]};
    filter_bias_mean_var_desc.set(
            1, dims, convert_to_cnnl_datatype(bn_scale.layout.dtype.enumv()),
            CNNL_LAYOUT_ARRAY);

    CnnlActivationDescriptor activation_desc;
    activation_desc.set(
            CNNL_ACTIVATION_IDENTITY, /*prefer=*/CNNL_ACTIVATION_HIGH_PRECISION,
            /*nanProp=*/CNNL_NOT_PROPAGATE_NAN, /*ceof=*/1.0);
    cnnl_check(cnnlBatchNormBackward_v2(
            handle, activation_desc.desc(), CNNL_BATCHNORM_SPATIAL,
            CNNL_BATCHNORM_OPS_BN, nullptr, nullptr, nullptr, nullptr, x_desc.desc(),
            x.raw_ptr(),                       // x_desc, x
            nullptr, nullptr,                  // y_desc, y
            dy_desc.desc(), dy.raw_ptr(),      // dy_desc, dy
            filter_bias_mean_var_desc.desc(),  // filter_bias_mean_var_desc
            bn_scale.raw_ptr(),                // filter
            nullptr,                           // bias
            saved_batch_mean.raw_ptr(), saved_batch_inv_variance.raw_ptr(),
            m_param.epsilon, nullptr, nullptr,  // diff_z_desc, diff_z
            dx_desc.desc(), dx.raw_ptr(),       // diff_x_desc, diff_x
            d_bn_scale.raw_ptr(),               // diff_filter
            d_bn_bias.raw_ptr(),                // diff_bias
            workspace.raw_ptr, workspace.size, reserve.raw_ptr(),
            reserve.layout.access_bytes()));
}

}  // namespace cambricon
}  // namespace megdnn
   // vim: syntax=cpp.doxygen