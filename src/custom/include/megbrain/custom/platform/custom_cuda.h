#pragma once

#include "megbrain/custom/op.h"

#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace custom {

class MGE_WIN_DECLSPEC_FUC CudaRuntimeArgs {
private:
    int m_device;
    cudaStream_t m_stream;

public:
    CudaRuntimeArgs(int device, cudaStream_t stream)
            : m_device(device), m_stream(stream) {}

    int device() const { return m_device; }

    cudaStream_t stream() const { return m_stream; }
};

MGE_WIN_DECLSPEC_FUC const CudaRuntimeArgs
get_cuda_runtime_args(const RuntimeArgs& rt_args);
MGE_WIN_DECLSPEC_FUC int get_cuda_device_id(Device device);
MGE_WIN_DECLSPEC_FUC const cudaDeviceProp* get_cuda_device_props(Device device);
MGE_WIN_DECLSPEC_FUC cudaStream_t get_cuda_stream(Device device);

}  // namespace custom
