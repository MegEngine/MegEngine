#pragma once

#include "src/common/megcore/common/computing_context.hpp"
#include "megcore_cuda.h"
#include <cuda_runtime_api.h>

namespace megcore {
namespace cuda {

class CUDAComputingContext final: public ComputingContext {
    public:
        CUDAComputingContext(megcoreDeviceHandle_t dev_handle,
                unsigned int flags, const CudaContext &ctx = {});
        ~CUDAComputingContext();

        void memcpy(void *dst, const void *src, size_t size_in_bytes,
                megcoreMemcpyKind_t kind) override;
        void memset(void *dst, int value, size_t size_in_bytes) override;
        void synchronize() override;

        const CudaContext& context() const {
            return context_;
        }

        cudaStream_t stream() const {
            return context().stream;
        }

    private:
        bool own_stream_;
        CudaContext context_;
};

} // namespace cuda
} // namespace megcore

// vim: syntax=cpp.doxygen
