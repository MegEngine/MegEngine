#pragma once

#include "megbrain/custom/op.h"

#include <cuda_runtime_api.h>

namespace custom {

class CudaRuntimeArgs {
private:
    int m_device;
    cudaStream_t m_stream;

public:
    CudaRuntimeArgs(int device, cudaStream_t stream)
            : m_device(device), m_stream(stream) {}

    int device() const { return m_device; }

    cudaStream_t stream() const { return m_stream; }
};

const CudaRuntimeArgs get_cuda_runtime_args(const RuntimeArgs& rt_args);

}  // namespace custom
