#include "../conv_pooling_utils.cuh"
#include "./kernel_impl.h"

namespace megdnn {
namespace cuda {
namespace conv_pool {

DISPATCH_POOLSHAPE(Relu, 6)

}  // namespace conv_pool
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
