#pragma once

#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"

#if MEGDNN_CC_HOST
#include "megdnn/oprs.h"
#endif

namespace megdnn {
namespace cuda {

template <typename ctype>
struct FakeQuantKernOp {
    ctype* input;
    ctype* output;
    ctype qmin, qmax;

    __device__ void operator()(uint32_t idx, ctype scale, ctype zero_point) {
        ctype x = round(input[idx] / scale) + zero_point;
        if (isnan(x)) {
            output[idx] = NAN;
            return;
        }
        x = fmaxf(fminf(x, qmax), qmin);
        output[idx] = (x - zero_point) * scale;
    }

#if MEGDNN_CC_HOST
    FakeQuantKernOp(
            const TensorND& input, const TensorND& output,
            const FakeQuant::Param& param)
            : input{input.ptr<ctype>()},
              output{output.ptr<ctype>()},
              qmin(param.qmin),
              qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct FakeQuantBwdKernOp {
    ctype* diff;
    ctype* input;
    ctype* grad;
    ctype qmin, qmax;

    __device__ void operator()(uint32_t idx, ctype scale, ctype zero_point) {
        ctype x = round(input[idx] / scale) + zero_point;
        grad[idx] = isnan(x) ? NAN : x <= qmax && x >= qmin ? diff[idx] : 0.0;
    }

#if MEGDNN_CC_HOST
    FakeQuantBwdKernOp(
            const TensorND& diff, const TensorND& input, const TensorND& grad,
            const FakeQuant::Param& param)
            : diff{diff.ptr<ctype>()},
              input{input.ptr<ctype>()},
              grad{grad.ptr<ctype>()},
              qmin(param.qmin),
              qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct FakeQuantKernOpNonContig {
    ctype qmin;
    ctype qmax;

    __device__ void operator()(
            uint32_t, ctype& output, ctype input, ctype scale, ctype zero_point) {
        ctype x = round(input / scale) + zero_point;
        if (isnan(x)) {
            output = NAN;
            return;
        }
        x = fmaxf(fminf(x, qmax), qmin);
        output = (x - zero_point) * scale;
    }

#if MEGDNN_CC_HOST
    FakeQuantKernOpNonContig(const FakeQuant::Param& param)
            : qmin(param.qmin), qmax(param.qmax) {}
#endif
};

template <typename ctype>
struct FakeQuantBwdKernOpNonContig {
    ctype qmin;
    ctype qmax;

    __device__ void operator()(
            uint32_t, ctype& grad, ctype diff, ctype input, ctype scale,
            ctype zero_point) {
        ctype x = round(input / scale) + zero_point;
        grad = isnan(x) ? NAN : x <= qmax && x >= qmin ? diff : 0.0;
    }

#if MEGDNN_CC_HOST
    FakeQuantBwdKernOpNonContig(const FakeQuant::Param& param)
            : qmin(param.qmin), qmax(param.qmax) {}
#endif
};

}  // namespace cuda
}  // namespace megdnn
