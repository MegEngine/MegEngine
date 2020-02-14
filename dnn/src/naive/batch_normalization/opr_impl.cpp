/**
 * \file dnn/src/naive/batch_normalization/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/batch_normalization/opr_impl.h"

#include <cmath>
#include <cstring>
#include "src/naive/handle.h"

#define rep_4d(dims, offsets)                                     \
    src_pos = 0;                                                  \
    for (size_t n = 0; n < dims[0]; ++n) {                        \
        for (size_t c = 0; c < dims[1]; ++c) {                    \
            for (size_t h = 0; h < dims[2]; ++h) {                \
                for (size_t w = 0; w < dims[3]; ++w) {            \
                    param_pos = n * offsets[0] + c * offsets[1] + \
                                h * offsets[2] + w * offsets[3];
#define rep_4d_end \
    ++src_pos;     \
    }              \
    }              \
    }              \
    }

namespace megdnn {
namespace naive {

namespace {

template <typename T0, typename T1 = T0>
void bn_forward_exec(_megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
                     _megdnn_tensor_in bn_bias, _megdnn_tensor_inout mean,
                     _megdnn_tensor_inout variance,
                     _megdnn_tensor_out batch_mean,
                     _megdnn_tensor_out batch_inv_variance,
                     _megdnn_tensor_out dst, param::BN param) {
    size_t src_shape[4], dim_offset[4], param_pos = 0, src_pos = 0,
                                        batch_size = 1;

    float sigma_p, tmp, epsilon = (float)param.epsilon, denominator = 1.f;

    T0 *src_p = src.ptr<T0>(), *dst_p = dst.ptr<T0>();
    T1 *bn_scale_p = bn_scale.ptr<T1>(), *bn_bias_p = bn_bias.ptr<T1>(),
       *mean_p = mean.ptr<T1>(), *variance_p = variance.ptr<T1>(),
       *batch_mean_p = batch_mean.ptr<T1>(),
       *batch_inv_variance_p = batch_inv_variance.ptr<T1>();

    rep(i, src.layout.ndim) {
        src_shape[i] = src.layout.shape[i];
        if (bn_scale.layout.shape[i] == 1) {
            dim_offset[i] = 0;
            batch_size *= src_shape[i];
        } else {
            dim_offset[i] = 1;
        }
    }

    int curr_stride = 0;
    for (int i = 3; i >= 0; --i) {
        if (dim_offset[i] != 0) {
            if (curr_stride == 0) {
                dim_offset[i] = 1;
                curr_stride = src_shape[i];
            } else {
                dim_offset[i] = curr_stride;
                curr_stride *= src_shape[i];
            }
        }
    }
    denominator = 1.0 / batch_size;

    if (param.fwd_mode == param::BN::FwdMode::TRAINING) {
        // Calculate the means of this batch (Mu)
        memset(batch_mean.raw_ptr, 0,
               batch_mean.layout.total_nr_elems() * sizeof(float));
        rep_4d(src_shape, dim_offset) batch_mean_p[param_pos] += src_p[src_pos];
        rep_4d_end

        rep(i, batch_mean.layout.total_nr_elems()) {
            batch_mean_p[i] *= denominator;
            if (!mean.layout.is_empty()) {
                mean_p[i] = (1 - param.avg_factor) * mean_p[i] +
                            param.avg_factor * batch_mean_p[i];
            }
        }

        // Calculate the variances of this batch (Sigma)
        memset(batch_inv_variance.raw_ptr, 0,
               batch_inv_variance.layout.total_nr_elems() * sizeof(float));
        rep_4d(src_shape, dim_offset) sigma_p =
                src_p[src_pos] - batch_mean_p[param_pos];
        batch_inv_variance_p[param_pos] += sigma_p * sigma_p;
        rep_4d_end

        rep(i, batch_inv_variance.layout.total_nr_elems()) {
            tmp = batch_inv_variance_p[i] * denominator;
            batch_inv_variance_p[i] = 1 / sqrt(tmp + epsilon);
            if (!variance.layout.is_empty()) {
                variance_p[i] =
                        (1 - param.avg_factor) * variance_p[i] +
                        param.avg_factor * tmp * batch_size / (batch_size - 1);
            }
        }
        // Calculate Normalization of the input data.
        size_t dst_pos = 0;
        rep_4d(src_shape, dim_offset) tmp =
                (src_p[dst_pos] - batch_mean_p[param_pos]) *
                batch_inv_variance_p[param_pos];
        dst_p[dst_pos] = bn_scale_p[param_pos] * tmp + bn_bias_p[param_pos];

        ++dst_pos;
        rep_4d_end
    } else if (param.fwd_mode == param::BN::FwdMode::INFERENCE) {
        size_t dst_pos = 0;
        rep_4d(src_shape, dim_offset) tmp =
                (src_p[dst_pos] - mean_p[param_pos]) /
                sqrt(variance_p[param_pos] + epsilon);
        dst_p[dst_pos] = bn_scale_p[param_pos] * tmp + bn_bias_p[param_pos];

        ++dst_pos;
        rep_4d_end
    }
}

template <typename T0, typename T1 = T0>
void bn_backward_exec(_megdnn_tensor_in x_in, _megdnn_tensor_in dy_in,
                      _megdnn_tensor_in saved_batch_mean,
                      _megdnn_tensor_in saved_batch_inv_variance,
                      _megdnn_tensor_in bn_scale, _megdnn_tensor_out d_bn_scale,
                      _megdnn_tensor_out d_bn_bias, _megdnn_tensor_out dx_out,
                      const WorkspaceBundle& bundle) {
    size_t src_shape[4], dim_offset[4],
            param_pos = 0, src_pos = 0, batch_size = 1,
            param_size = bn_scale.layout.total_nr_elems();

    float denominator = 1.f;

    T0 *x = x_in.ptr<T0>(), *dx = dx_out.ptr<T0>(), *dy = dy_in.ptr<T0>();
    T1 *gamma = bn_scale.ptr<T1>(), *mu = saved_batch_mean.ptr<T1>(),
       *ivar = saved_batch_inv_variance.ptr<T1>(),
       *dgamma = d_bn_scale.ptr<T1>(), *dbeta = d_bn_bias.ptr<T1>();

    rep(i, dy_in.layout.ndim) {
        src_shape[i] = dy_in.layout.shape[i];
        if (bn_scale.layout.shape[i] == 1) {
            dim_offset[i] = 0;
            batch_size *= src_shape[i];
        } else {
            dim_offset[i] = 1;
        }
    }

    int curr_stride = 0;
    for (int i = 3; i >= 0; --i) {
        if (dim_offset[i] != 0) {
            if (curr_stride == 0) {
                dim_offset[i] = 1;
                curr_stride = src_shape[i];
            } else {
                dim_offset[i] = curr_stride;
                curr_stride *= src_shape[i];
            }
        }
    }
    denominator = 1.0 / batch_size;

    // step1. dbeta, dgamma
    memset(dbeta, 0, param_size * sizeof(T1));
    memset(dgamma, 0, param_size * sizeof(T1));
    rep_4d(src_shape, dim_offset) float xhat =
            (x[src_pos] - mu[param_pos]) * ivar[param_pos];
    dbeta[param_pos] += dy[src_pos];
    dgamma[param_pos] += dy[src_pos] * xhat;
    rep_4d_end

            // step2. dxhat = dy * gamma
            float* dxhat = static_cast<float*>(bundle.get(0));
    rep_4d(src_shape, dim_offset) dxhat[src_pos] =
            dy[src_pos] * gamma[param_pos];
    rep_4d_end

            // step3. dvar = sigma[dxhat * xmu] * [-1/2 * (ivar)^(3/2)]
            //        dmu = sigma[dxhat] * (-ivar)
            float* dvar = static_cast<float*>(bundle.get(1));
    float* dmu = static_cast<float*>(bundle.get(2));
    memset(dvar, 0, param_size * sizeof(float));
    memset(dmu, 0, param_size * sizeof(float));
    rep_4d(src_shape, dim_offset) float xmu = (x[src_pos] - mu[param_pos]);
    dvar[param_pos] += dxhat[src_pos] * xmu;
    dmu[param_pos] += dxhat[src_pos];
    rep_4d_end

    rep(i, param_size) {
        // dvar[i] *= ( -0.5 * ivar[i] * sqrt(ivar[i]) );
        float sqrtivar = ivar[i];
        dvar[i] *= (-0.5 * sqrtivar * sqrtivar * sqrtivar);
        dmu[i] *= (-ivar[i]);
    }

    // step4. dx
    rep_4d(src_shape, dim_offset) float xmu = (x[src_pos] - mu[param_pos]);
    dx[src_pos] = dxhat[src_pos] * ivar[param_pos] +
                  2.0 * dvar[param_pos] * xmu * denominator +
                  dmu[param_pos] * denominator;
    rep_4d_end
}

};  // anonymous namespace

void BNForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in bn_scale,
                         _megdnn_tensor_in bn_bias, _megdnn_tensor_inout mean,
                         _megdnn_tensor_inout variance,
                         _megdnn_tensor_out batch_mean,
                         _megdnn_tensor_out batch_inv_variance,
                         _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, bn_scale.layout, bn_bias.layout, mean.layout,
               variance.layout, batch_mean.layout, batch_inv_variance.layout,
               dst.layout, workspace.size);

    MEGDNN_INC_FLOAT16(if (src.layout.dtype == dtype::Float16() &&
                           bn_scale.layout.dtype == dtype::Float32()) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(({
            using T0 = typename DTypeTrait<dtype::Float16>::ctype;
            using T1 = typename DTypeTrait<dtype::Float32>::ctype;
            bn_forward_exec<T0, T1>(src, bn_scale, bn_bias, mean, variance,
                                    batch_mean, batch_inv_variance, dst,
                                    m_param);
        }));
    } else) {
        megdnn_assert(src.layout.dtype == bn_scale.layout.dtype);
        switch (src.layout.dtype.enumv()) {
#define cb(_dt)                                                     \
    case DTypeTrait<_dt>::enumv: {                                  \
        using T = typename DTypeTrait<_dt>::ctype;                  \
        MEGDNN_DISPATCH_CPU_KERN_OPR((bn_forward_exec<T>(           \
                src, bn_scale, bn_bias, mean, variance, batch_mean, \
                batch_inv_variance, dst, m_param)));                \
        break;                                                      \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    }
}

WorkspaceBundle BNBackwardImpl::get_workspace_bundle(size_t x_size,
                                                     size_t param_size,
                                                     void* raw_ptr) {
    return {raw_ptr,
            {sizeof(float) * x_size, sizeof(float) * param_size,
             sizeof(float) * param_size}};
}

size_t BNBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& x, const TensorLayout&, const TensorLayout&,
        const TensorLayout&, const TensorLayout& bn_scale, const TensorLayout&,
        const TensorLayout&, const TensorLayout&) {
    auto x_size = x.total_nr_elems(), param_size = bn_scale.total_nr_elems();
    return get_workspace_bundle(x_size, param_size).total_size_in_bytes();
}

void BNBackwardImpl::exec(_megdnn_tensor_in x_in, _megdnn_tensor_in dy_in,
                          _megdnn_tensor_in saved_batch_mean,
                          _megdnn_tensor_in saved_batch_inv_variance,
                          _megdnn_tensor_in bn_scale,
                          _megdnn_tensor_out d_bn_scale,
                          _megdnn_tensor_out d_bn_bias,
                          _megdnn_tensor_out dx_out,
                          _megdnn_workspace workspace) {
    check_exec(x_in.layout, dy_in.layout, saved_batch_mean.layout,
               saved_batch_inv_variance.layout, bn_scale.layout,
               d_bn_scale.layout, d_bn_bias.layout, dx_out.layout,
               workspace.size);

    auto&& bundle = get_workspace_bundle(x_in.layout.total_nr_elems(),
                                         bn_scale.layout.total_nr_elems(),
                                         workspace.raw_ptr);

    MEGDNN_INC_FLOAT16(if (x_in.layout.dtype == dtype::Float16() &&
                           bn_scale.layout.dtype == dtype::Float32()) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(({
            using T0 = typename DTypeTrait<dtype::Float16>::ctype;
            using T1 = typename DTypeTrait<dtype::Float32>::ctype;
            bn_backward_exec<T0, T1>(x_in, dy_in, saved_batch_mean,
                                     saved_batch_inv_variance, bn_scale,
                                     d_bn_scale, d_bn_bias, dx_out, bundle);
        }));
    } else) {
        megdnn_assert(x_in.layout.dtype == bn_scale.layout.dtype);
        switch (x_in.layout.dtype.enumv()) {
#define cb(_dt)                                                          \
    case DTypeTrait<_dt>::enumv: {                                       \
        using T = typename DTypeTrait<_dt>::ctype;                       \
        MEGDNN_DISPATCH_CPU_KERN_OPR((bn_backward_exec<T>(               \
                x_in, dy_in, saved_batch_mean, saved_batch_inv_variance, \
                bn_scale, d_bn_scale, d_bn_bias, dx_out, bundle)));      \
        break;                                                           \
    }
            MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    }
}

#undef rep_4d
#undef rep_4d_end

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
