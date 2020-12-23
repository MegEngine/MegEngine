/**
 * \file dnn/src/cuda/tqt/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"

#if MEGDNN_CC_HOST
#include "megdnn/oprs.h"
#endif

namespace megdnn {
namespace cuda {

template <typename ctype>
struct TQTKernOp {
    ctype* input;
    ctype* output;
    ctype qmin, qmax;

    __device__ void operator()(uint32_t idx, ctype scale) {
        ctype t = powf(2, scale);
        ctype x = round(input[idx] / t);
        x = fmaxf(fminf(x, qmax), qmin);
        output[idx] = x * t;
    }

#if MEGDNN_CC_HOST
    TQTKernOp(const TensorND& input, const TensorND& output,
              const TQT::Param& param)
            : input{input.ptr<ctype>()},
              output{output.ptr<ctype>()},
              qmin(param.qmin),
              qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct TQTBwdKernOp {
    ctype* diff;
    ctype* input;
    ctype* grad_x;
    ctype* grad_s;
    ctype qmin, qmax;

    __device__ void operator()(uint32_t idx, ctype scale) {
        ctype t = powf(2, scale);
        ctype scaled = input[idx] / t;
        ctype rounded = round(scaled);
        rounded = fmaxf(fminf(rounded, qmax), qmin);
        bool mask_clip = scaled < -0.5 + qmin && scaled > 0.5 + qmax;
        bool mask_quant = !mask_clip;

        grad_x[idx] = diff[idx] * mask_quant;
        ctype grad_quant =
                diff[idx] * mask_quant * (rounded - scaled) * t * log(2.0);
        ctype grad_clip = diff[idx] * mask_clip * rounded * t * log(2.0);
        grad_s[idx] = grad_quant + grad_clip;
    }

#if MEGDNN_CC_HOST
    TQTBwdKernOp(const TensorND& diff, const TensorND& input,
                 const TensorND& grad_x, const TensorND& grad_s,
                 const TQT::Param& param)
            : diff{diff.ptr<ctype>()},
              input{input.ptr<ctype>()},
              grad_x{grad_x.ptr<ctype>()},
              grad_s{grad_s.ptr<ctype>()},
              qmin(param.qmin),
              qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct TQTKernOpNonContig {
    ctype qmin;
    ctype qmax;

    __device__ void operator()(uint32_t, ctype& input, ctype& scale,
                               ctype& output) {
        ctype t = powf(2, scale);
        ctype x = round(input / t);
        x = fmaxf(fminf(x, qmax), qmin);
        output = x * t;
    }
#if MEGDNN_CC_HOST
    TQTKernOpNonContig(const TQT::Param& param)
            : qmin(param.qmin), qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct TQTBwdKernOpNonContig {
    ctype qmin;
    ctype qmax;

    __device__ void operator()(uint32_t, ctype& diff, ctype& input,
                               ctype& scale, ctype& grad_x, ctype& grad_s) {
        ctype t = powf(2, scale);
        ctype scaled = input / t;
        ctype rounded = round(scaled);
        rounded = fmaxf(fminf(rounded, qmax), qmin);
        bool mask_clip = scaled < -0.5 + qmin && scaled > 0.5 + qmax;
        bool mask_quant = !mask_clip;

        grad_x = diff * mask_quant;
        ctype grad_quant =
                diff * mask_quant * (rounded - scaled) * t * log(2.0);
        ctype grad_clip = diff * mask_clip * rounded * t * log(2.0);
        grad_s = grad_quant + grad_clip;
    }
#if MEGDNN_CC_HOST
    TQTBwdKernOpNonContig(const TQT::Param& param)
            : qmin(param.qmin), qmax(param.qmax) {}
#endif
};

}  // namespace cuda
}  // namespace megdnn
