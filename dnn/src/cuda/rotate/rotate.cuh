#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>

namespace megdnn {
namespace cuda {
namespace rotate {

template <typename T, bool clockwise>
void rotate(
        const T* src, T* dst, size_t N, size_t IH, size_t IW, size_t CH,
        size_t istride0, size_t istride1, size_t istride2, size_t OH, size_t OW,
        size_t ostride0, size_t ostride1, size_t ostride2, cudaStream_t stream);

}  // namespace rotate
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
