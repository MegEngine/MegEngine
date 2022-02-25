#include "cuda.h"
#include "cuda_fp16.h"
#include "src/cuda/conv_bias/chanwise/kern.cuh"
#include "src/cuda/conv_bias/chanwise/kern_helper.cuh"
#include "src/cuda/conv_bias/chanwise/launch_config.cuh"
#include "src/cuda/fp16_help.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;
using namespace chanwise;

#include "src/cuda/conv_bias/chanwise/depthwise_large_filter_algo.cuh"

namespace megdnn {
namespace cuda {
namespace conv_bias {
namespace chanwise {

// =====================================fwd=====================================

#define check

template <>
void run_fwd_depthwise_large_filter(
        float* dst, const float* src, const float* flt, const Param& param,
        cudaStream_t stream) {
    INSTANCE(float, float2, DepthwiseConv2dDirection::DIRECTION_FORWARD)
}

#if CUDA_VERSION >= 9000
template <>
void run_fwd_depthwise_large_filter(
        __half* dst, const __half* src, const __half* flt, const Param& param,
        cudaStream_t stream) {
    INSTANCE(__half, __half2, DepthwiseConv2dDirection::DIRECTION_FORWARD)
}
#endif

}  // namespace chanwise
}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
