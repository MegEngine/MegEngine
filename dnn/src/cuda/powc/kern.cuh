#include "megdnn/basic_types.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

void powc_kern(
        const TensorND& dest, const TensorND& src, const float* exp_f, const int* exp_i,
        cudaStream_t stream);

}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
