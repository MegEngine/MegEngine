#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace test {

void pollute_shared_mem(cudaStream_t stream);

}  // namespace test
}  // namespace megdnn
