#pragma once

#include "../common/device_context.hpp"

namespace megcore {
namespace cpu {

/**
 * \brief A thin wrapper class over malloc and free.
 *
 * No magic thing happens here.
 */
class DefaultDeviceContext: public DeviceContext {
    public:
        DefaultDeviceContext(int device_id, unsigned int flags);
        ~DefaultDeviceContext() noexcept;

        size_t mem_alignment_in_bytes() const noexcept override;

        void activate() noexcept override;
        void *malloc(size_t size_in_bytes) override;
        void free(void *ptr) override;
};

} // namespace cpu
} // namespace megcore

// vim: syntax=cpp.doxygen
