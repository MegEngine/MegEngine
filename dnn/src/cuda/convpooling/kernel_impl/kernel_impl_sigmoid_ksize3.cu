#include "../conv_pooling_utils.cuh"
#include "./kernel_impl.h"

namespace megdnn {
namespace cuda {
namespace conv_pool {

DISPATCH_POOLSHAPE(Sigmoid, 3)

}  // namespace conv_pool
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen
