/**
 * \file dnn/src/naive/layer_norm/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/naive/layer_norm/opr_impl.h"
#include <algorithm>
#include "src/common/utils.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace naive;

namespace {

using Param = megdnn::LayerNorm::Param;

template <typename T, typename T_ACC = float>
void forward(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        const Param& param) {
    float eps = param.eps;
    bool affine = param.affine;
    uint64_t slice_length = param.normalized_size;
    uint64_t slice_dim = param.normalized_dim;
    uint64_t n_slices = 1;
    for (size_t i = 0; i < data.layout.ndim - slice_dim; ++i) {
        n_slices = n_slices * data.layout.shape[i];
    }

    for (size_t i = 0; i < n_slices; i++) {
        T_ACC slice_sum = static_cast<T>(0.0f);
        for (size_t j = 0; j < slice_length; j++) {
            auto value = data.ptr<T>()[i * slice_length + j];
            slice_sum += value;
        }
        T_ACC slice_mean = static_cast<T>(slice_sum / slice_length);

        T_ACC slice_var = static_cast<T>(0.0f);
        for (size_t j = 0; j < slice_length; j++) {
            slice_var += (data.ptr<T>()[i * slice_length + j] - slice_mean) *
                         (data.ptr<T>()[i * slice_length + j] - slice_mean);
        }
        slice_var = slice_var / slice_length;

        T_ACC slice_std = static_cast<T>(sqrt(slice_var + eps));
        for (size_t j = 0; j < slice_length; j++) {
            dst.ptr<T>()[i * slice_length + j] =
                    (data.ptr<T>()[i * slice_length + j] - slice_mean) / slice_std;
            if (affine) {
                dst.ptr<T>()[i * slice_length + j] =
                        dst.ptr<T>()[i * slice_length + j] * weight.ptr<T>()[j] +
                        bias.ptr<T>()[j];
            }
        }
        mean.ptr<T_ACC>()[i] = static_cast<T_ACC>(slice_mean);
        rstd.ptr<T_ACC>()[i] = static_cast<T_ACC>(1.0 / slice_std);
    }
}

template <typename T, typename T_ACC = float>
void backward(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias, const Param& param) {
    bool affine = param.affine;
    uint64_t slice_length = param.normalized_size;
    uint64_t slice_dim = param.normalized_dim;
    uint64_t n_slices = 1;
    for (size_t i = 0; i < data.layout.ndim - slice_dim; ++i) {
        n_slices = n_slices * data.layout.shape[i];
    }

    if (affine) {
        for (size_t i = 0; i < slice_length; ++i) {
            dweight.ptr<T>()[i] = 0;
            dbias.ptr<T>()[i] = 0;
        }

        for (size_t i = 0; i < n_slices; ++i) {
            for (size_t j = 0; j < slice_length; ++j) {
                dweight.ptr<T>()[j] +=
                        (data.ptr<T>()[i * slice_length + j] - mean.ptr<T_ACC>()[i]) *
                        rstd.ptr<T_ACC>()[i] * diff.ptr<T>()[i * slice_length + j];

                dbias.ptr<T>()[j] += diff.ptr<T>()[i * slice_length + j];
            }
        }
    }

    for (size_t i = 0; i < n_slices; ++i) {
        T_ACC ds = static_cast<T_ACC>(0.0f);
        T_ACC db = static_cast<T_ACC>(0.0f);
        T_ACC a = static_cast<T_ACC>(0.0f);
        T_ACC b = static_cast<T_ACC>(0.0f);
        T_ACC c = static_cast<T_ACC>(0.0f);

        for (size_t j = 0; j < slice_length; ++j) {
            auto value = data.ptr<T>()[i * slice_length + j];
            auto diff_v = diff.ptr<T>()[i * slice_length + j];
            auto weight_v = affine ? weight.ptr<T>()[j] : static_cast<T>(1.0f);
            db += diff_v * weight_v;
            ds += diff_v * value * weight_v;
        }

        a = rstd.ptr<T_ACC>()[i];
        b = (db * mean.ptr<T_ACC>()[i] - ds) * a * a * a / slice_length;
        c = -b * mean.ptr<T_ACC>()[i] - db * a / slice_length;

        for (uint64_t j = 0; j < slice_length; j++) {
            auto weight_v = affine ? weight.ptr<T>()[j] : static_cast<T>(1.0f);
            ddata.ptr<T>()[i * slice_length + j] =
                    diff.ptr<T>()[i * slice_length + j] * a * weight_v +
                    data.ptr<T>()[i * slice_length + j] * b + c;
        }
    }
}

}  // namespace

namespace megdnn {
namespace naive {

void LayerNormForwardImpl::exec(
        _megdnn_tensor_in data, _megdnn_tensor_in weight, _megdnn_tensor_in bias,
        _megdnn_tensor_out dst, _megdnn_tensor_out mean, _megdnn_tensor_out rstd,
        _megdnn_workspace workspace) {
    check_exec(
            data.layout, weight.layout, bias.layout, dst.layout, mean.layout,
            rstd.layout, workspace.size);
#define cb(DType)                                                                \
    if (data.layout.dtype == DType()) {                                          \
        MEGDNN_DISPATCH_CPU_KERN_OPR(forward<typename DTypeTrait<DType>::ctype>( \
                data, weight, bias, dst, mean, rstd, param()));                  \
        return;                                                                  \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

void LayerNormBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_in data, _megdnn_tensor_in weight,
        _megdnn_tensor_in mean, _megdnn_tensor_in rstd, _megdnn_tensor_out ddata,
        _megdnn_tensor_out dweight, _megdnn_tensor_out dbias,
        _megdnn_workspace workspace) {
    check_exec(
            diff.layout, data.layout, weight.layout, mean.layout, rstd.layout,
            ddata.layout, dweight.layout, dbias.layout, workspace.size);
#define cb(DType)                                                                 \
    if (data.layout.dtype == DType()) {                                           \
        MEGDNN_DISPATCH_CPU_KERN_OPR(backward<typename DTypeTrait<DType>::ctype>( \
                diff, data, weight, mean, rstd, ddata, dweight, dbias, param())); \
        return;                                                                   \
    }
    MEGDNN_FOREACH_COMPUTING_DTYPE_FLOAT(cb)
#undef cb
    megdnn_throw("bad dtype");
}

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
