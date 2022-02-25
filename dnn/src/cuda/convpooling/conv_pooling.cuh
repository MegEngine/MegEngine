#pragma once
#include <cuda_runtime_api.h>
#include "./conv_pooling.h"

namespace megdnn {
namespace cuda {
namespace conv_pool {

template <
        int kern_h, int kern_w, int pool_shape_h, int pool_shape_w, class Nonlin,
        class Pooler, class IdxGetter>
__global__ void kern_xcorr_smallkern_pool(
        float* input, const float* filter, float* output, const float* output_bias,
        cudaTextureObject_t m_tex, int IC, int IH, int IW, int OH, int OW);

}  // namespace conv_pool
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
