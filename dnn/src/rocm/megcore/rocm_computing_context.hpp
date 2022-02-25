#pragma once

#include "src/common/megcore/common/computing_context.hpp"
#include "megcore_rocm.h"

namespace megcore {
namespace rocm {

class ROCMComputingContext final : public ComputingContext {
public:
    ROCMComputingContext(megcoreDeviceHandle_t dev_handle, unsigned int flags,
                         const ROCMContext& ctx = {});
    ~ROCMComputingContext();

    void memcpy(void* dst, const void* src, size_t size_in_bytes,
                megcoreMemcpyKind_t kind) override;
    void memset(void* dst, int value, size_t size_in_bytes) override;
    void synchronize() override;

    const ROCMContext& context() const { return context_; }
    hipStream_t stream() const { return context().stream; }

private:
    bool own_stream_;
    ROCMContext context_;
};

} // namespace rocm
} // namespace megcore

// vim: syntax=cpp.doxygen
