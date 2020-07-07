/**
 * \file dnn/src/cuda/remap/forward.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/config/config.h"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/remap/common.h"
#include "src/cuda/remap/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

void RemapImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out map_xy,
                     _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, map_xy.layout, dst.layout, workspace.size);
    megdnn_assert(map_xy.layout.dtype.enumv() ==
                  DTypeTrait<dtype::Float32>::enumv);
    auto stream = cuda_stream(this->handle());
    int N, C, IH, IW, OH, OW;
    OH = map_xy.layout.shape[1];
    OW = map_xy.layout.shape[2];

    megdnn_assert(param().imode == param::Remap::InterpolationMode::LINEAR,
                  "only support LINEAR interpolationMode");

    if (param().format == param::Remap::Format::NCHW) {
        N = src.layout.shape[0];
        C = src.layout.shape[1];
        IH = src.layout.shape[2];
        IW = src.layout.shape[3];
    } else if (param().format == param::Remap::Format::NHWC) {
        N = src.layout.shape[0];
        C = src.layout.shape[3];
        IH = src.layout.shape[1];
        IW = src.layout.shape[2];
    } else {
        megdnn_throw("unsupported format, cuda remap");
    }

#define cb(dt, _format, bmode)                                           \
    if (param().format == param::Remap::Format::_format &&               \
        param().border_type == param::Remap::BorderMode::bmode) {        \
        using ctype = DTypeTrait<dt>::ctype;                             \
        remap::forward_proxy<ctype, param_enumv::Remap::Format::_format, \
                             ::BorderMode::BORDER_##bmode>(              \
                src.compatible_ptr<ctype>(),                             \
                map_xy.compatible_ptr<dt_float32>(),                     \
                dst.compatible_ptr<ctype>(), N, C, IH, IW, OH, OW,       \
                param().scalar, stream);                                 \
        break;                                                           \
    }

#define support_dtype(dt)                                      \
    case DTypeTrait<dt>::enumv: {                              \
        cb(dt, NCHW, CONSTANT);                                \
        cb(dt, NCHW, REPLICATE);                               \
        cb(dt, NCHW, REFLECT);                                 \
        cb(dt, NCHW, REFLECT_101);                             \
        cb(dt, NCHW, WRAP);                                    \
        cb(dt, NHWC, CONSTANT);                                \
        cb(dt, NHWC, REPLICATE);                               \
        cb(dt, NHWC, REFLECT);                                 \
        cb(dt, NHWC, REFLECT_101);                             \
        cb(dt, NHWC, WRAP);                                    \
        megdnn_throw("unsupported border type in remap cuda"); \
    }

    switch (src.layout.dtype.enumv()) {
        support_dtype(dtype::Float32);
        MEGDNN_INC_FLOAT16(support_dtype(dtype::Float16));
        MEGDNN_INC_FLOAT16(support_dtype(dtype::BFloat16));
        support_dtype(dtype::Int8);
        support_dtype(dtype::Uint8);
        default:
            megdnn_throw("unsupported dtype in remap cuda");
    }

#undef support_dtype
#undef cb
}

// vim: syntax=cpp.doxygen
