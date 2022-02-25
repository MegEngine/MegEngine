#pragma once

#include "src/common/megcore/common/device_context.hpp"

namespace megcore {
namespace rocm {

class ROCMDeviceContext: public DeviceContext {
    public:
        ROCMDeviceContext(int device_id, unsigned int flags);
        ~ROCMDeviceContext() noexcept;

        size_t mem_alignment_in_bytes() const noexcept override;

        void activate() override;
        void *malloc(size_t size_in_bytes) override;
        void free(void *ptr) override;
    private:
        hipDeviceProp_t prop_;
};

} // namespace rocm
} // namespace megcore

// vim: syntax=cpp.doxygen
