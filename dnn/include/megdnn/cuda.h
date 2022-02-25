#pragma once
#include "megdnn/basic_types.h"

#include <cuda_runtime_api.h>
#include <memory>

#include "megdnn/internal/visibility_prologue.h"
namespace megdnn {

std::unique_ptr<Handle> make_cuda_handle_with_stream(
        cudaStream_t stream, int device_id = -1);
cudaStream_t get_cuda_stream(Handle* handle);

}  // namespace megdnn
#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
