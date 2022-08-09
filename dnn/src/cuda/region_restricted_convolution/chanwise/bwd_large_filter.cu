#include "cuda.h"
#include "cuda_fp16.h"
#include "src/cuda/fp16_help.cuh"
#include "src/cuda/region_restricted_convolution/chanwise/kern.cuh"

using namespace megdnn;
using namespace cuda;
using namespace region_restricted_convolution;
using namespace chanwise;

#include "src/cuda/region_restricted_convolution/chanwise/depthwise_large_filter_algo.cuh"

namespace megdnn {
namespace cuda {
namespace region_restricted_convolution {
namespace chanwise {

// =====================================bwd=====================================

template <>
void run_bwd_depthwise_large_filter(
        float* dst, const float* src, const float* flt, const int* rin, const int* rout,
        const Param& param, cudaStream_t stream) {
    INSTANCE_INT(float, int, DepthwiseConv2dDirection::DIRECTION_BACKWARD)
}

template <>
void run_bwd_depthwise_large_filter(
        float* dst, const float* src, const float* flt, const uint8_t* rin,
        const uint8_t* rout, const Param& param, cudaStream_t stream) {
    INSTANCE_UINT8(float, uint8_t, DepthwiseConv2dDirection::DIRECTION_BACKWARD)
}

}  // namespace chanwise
}  // namespace region_restricted_convolution
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
