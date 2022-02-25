#include "hip_header.h"
#include "megdnn/basic_types.h"
#include "src/rocm/utils.h.hip"

namespace megdnn {
namespace rocm {

void powc_kern(const TensorND& dest, const TensorND& src, const float* exp_f,
               const int* exp_i, hipStream_t stream);

}  // namespace rocm`
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
