#pragma once
#include "../conv_pooling.cuh"

namespace megdnn {
namespace cuda {
namespace conv_pool {

typedef void (*kern_corr_pointer)(
        float* input, const float* filter, float* output, const float* output_bias,
        cudaTextureObject_t m_tex, int IC, int IH, int IW, int OH, int OW);

#include "./kern_corr_func_macro.inc"

#define DISPATCH_POOLMODE(nonlin, kern_size, pool_size, idx_getter)         \
    KERN_CORR_DEFINE(                                                       \
            nonlin, kern_size, kern_size, pool_size, pool_size, idx_getter, \
            MeanPooler)                                                     \
    KERN_CORR_DEFINE(                                                       \
            nonlin, kern_size, kern_size, pool_size, pool_size, idx_getter, MaxPooler)

#define DISPATCH_CONVMODE(nonlin, kern_size, pool_size)                   \
    DISPATCH_POOLMODE(nonlin, kern_size, pool_size, IdxGetterConvolution) \
    DISPATCH_POOLMODE(nonlin, kern_size, pool_size, IdxGetterCorrRel)

#define DISPATCH_POOLSHAPE(nonlin, kern_size) \
    DISPATCH_CONVMODE(nonlin, kern_size, 1)   \
    DISPATCH_CONVMODE(nonlin, kern_size, 2)   \
    DISPATCH_CONVMODE(nonlin, kern_size, 3)   \
    DISPATCH_CONVMODE(nonlin, kern_size, 4)

}  // namespace conv_pool
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
