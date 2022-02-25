#pragma once

#include "src/common/megcore/common/device_context.hpp"
#include <cuda_runtime_api.h>

namespace megcore {
namespace cuda {

class CUDADeviceContext: public DeviceContext {
    public:
        CUDADeviceContext(int device_id, unsigned int flags);
        ~CUDADeviceContext() noexcept;

        size_t mem_alignment_in_bytes() const noexcept override;

        void activate() override;
        void *malloc(size_t size_in_bytes) override;
        void free(void *ptr) override;
    private:
        cudaDeviceProp prop_;
};

} // namespace cuda
} // namespace megcore

// vim: syntax=cpp.doxygen
