/**
 * \file dnn/src/cuda/lsq/kern.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
struct LSQKernOp {
    ctype* input;
    ctype* output;
    ctype qmin, qmax;

    __device__ void operator()(
            uint32_t idx, ctype scale, ctype zero_point, ctype grad_scale) {
        ctype x = input[idx] / scale + zero_point;
        x = fmaxf(fminf(x, qmax), qmin);
        x = round(x);
        output[idx] = (x - zero_point) * scale;
    }

#if MEGDNN_CC_HOST
    LSQKernOp(const TensorND& input, const TensorND& output, const LSQ::Param& param)
            : input{input.ptr<ctype>()},
              output{output.ptr<ctype>()},
              qmin(param.qmin),
              qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct LSQBwdKernOp {
    ctype* diff;
    ctype* input;
    ctype* grad_x;
    ctype* grad_s;
    ctype qmin, qmax;

    __device__ void operator()(
            uint32_t idx, ctype scale, ctype zero_point, ctype grad_scale) {
        ctype x = input[idx] / scale + zero_point;
        bool ind_small = x < qmin;
        bool ind_big = x > qmax;
        bool ind_middle = ind_small ^ ind_big;
        ind_middle = !ind_middle;
        grad_s[idx] = ind_small * qmin + ind_big * qmax + ind_middle * (-x + round(x));
        grad_s[idx] = grad_s[idx] * grad_scale * diff[idx];
        grad_x[idx] = ind_middle * diff[idx];
    }

#if MEGDNN_CC_HOST
    LSQBwdKernOp(
            const TensorND& diff, const TensorND& input, const TensorND& grad_x,
            const TensorND& grad_s, const LSQ::Param& param)
            : diff{diff.ptr<ctype>()},
              input{input.ptr<ctype>()},
              grad_x{grad_x.ptr<ctype>()},
              grad_s{grad_s.ptr<ctype>()},
              qmin(param.qmin),
              qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct LSQKernOpNonContig {
    ctype qmin;
    ctype qmax;

    __device__ void operator()(
            uint32_t, ctype& output, ctype& input, ctype& scale, ctype& zero_point,
            ctype grad_scale) {
        ctype x = input / scale + zero_point;
        x = fmaxf(fminf(x, qmax), qmin);
        x = round(x);
        output = (x - zero_point) * scale;
    }
#if MEGDNN_CC_HOST
    LSQKernOpNonContig(const LSQ::Param& param) : qmin(param.qmin), qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct LSQBwdKernOpNonContig {
    ctype qmin;
    ctype qmax;

    __device__ void operator()(
            uint32_t, ctype& grad_x, ctype& grad_s, ctype& diff, ctype& input,
            ctype& scale, ctype& zero_point, ctype grad_scale) {
        ctype x = input / scale + zero_point;
        bool ind_small = x < qmin;
        bool ind_big = x > qmax;
        bool ind_middle = ind_small ^ ind_big;
        ind_middle = !ind_middle;
        grad_s = ind_small * qmin + ind_big * qmax + ind_middle * (-x + round(x));
        grad_s = grad_s * grad_scale * diff;
        grad_x = ind_middle * diff;
    }
#if MEGDNN_CC_HOST
    LSQBwdKernOpNonContig(const LSQ::Param& param)
            : qmin(param.qmin), qmax(param.qmax) {}
#endif
};

}  // namespace cuda
}  // namespace megdnn
