#pragma once
#include <cuda_runtime_api.h>
#include "megcore_cdefs.h"
#include "src/common/cv/enums.h"
#include "src/common/opr_param_defs_enumv.cuh"

namespace megdnn {
namespace cuda {
namespace remap {

// all these kernels use LINEAR interpolation

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void forward_proxy(
        const ctype* src, const float* map_xy, ctype* dst, int N, int C, int IH, int IW,
        int OH, int OW, float scalar, cudaStream_t stream);

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void backwarddata_proxy(
        ctype* grad, const float* map_xy, const ctype* diff, int N, int C, int IH,
        int IW, int OH, int OW, cudaStream_t stream);

template <
        typename ctype, const uint32_t format, ::BorderMode bmode,
        ::InterpolationMode imode>
void backwardmat_proxy(
        const ctype* src, const float* map_xy, const ctype* diff, float* grad, int N,
        int C, int IH, int IW, int OH, int OW, float scalar, cudaStream_t stream);

}  // namespace remap
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
