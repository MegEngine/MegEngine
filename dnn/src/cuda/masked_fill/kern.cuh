#pragma once

#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"

#if MEGDNN_CC_HOST
#include "megdnn/oprs.h"
#endif

namespace megdnn {
namespace cuda {

template <typename ctype>
struct MaskedFillScalarKernOp {
    using VectTypeTrait = elemwise_intl::VectTypeTrait<ctype>;
    typedef typename VectTypeTrait::vect_type vect_type;
    ctype* output;
    bool* mask;
    ctype value;
    uint32_t mask_stride;

    __device__ __forceinline__ void operator()(uint32_t idx, ctype orig) {
        output[idx] = mask[idx / mask_stride]
                            ? value
                            : orig;  //! mask[idx] * orig + mask[idx]* *value;
    }
    __device__ __forceinline__ void operator()(uint32_t idx, vect_type orig) {
        ctype a = mask[(idx) / mask_stride] ? value : orig.x;
        ctype b = mask[(idx + 1) / mask_stride] ? value : orig.y;
        ctype g = mask[(idx + 2) / mask_stride] ? value : orig.z;
        ctype r = mask[(idx + 3) / mask_stride] ? value : orig.w;
        *(vect_type*)(&output[idx]) = VectTypeTrait::make_vector(a, b, g, r);
    }

#if MEGDNN_CC_HOST
    MaskedFillScalarKernOp(
            const TensorND& output, const TensorND& mask, ctype value,
            uint32_t mask_stride)
            : output{output.ptr<ctype>()},
              mask{mask.ptr<bool>()},
              value{value},
              mask_stride{mask_stride} {}
#endif
};

}  // namespace cuda
}  // namespace megdnn
