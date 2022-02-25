#pragma once

#include "../common/computing_context.hpp"

namespace megcore {
namespace cpu {

/**
 * \brief A thin wrapper over memcpy and memset.
 *
 * No magic thing happens here.
 */
class DefaultComputingContext: public ComputingContext {
    std::shared_ptr<MegcoreCPUDispatcher> m_dispatcher;

    public:
        DefaultComputingContext(megcoreDeviceHandle_t dev_handle,
                unsigned int flags);
        ~DefaultComputingContext() noexcept;

        void set_dispatcher(
                const std::shared_ptr<MegcoreCPUDispatcher>& dispatcher) {
            m_dispatcher = dispatcher;
        }

        MegcoreCPUDispatcher* get_dispatcher() const {
            return m_dispatcher.get();
        }

        void memcpy(void *dst, const void *src, size_t size_in_bytes,
                megcoreMemcpyKind_t kind) override;
        void memset(void *dst, int value, size_t size_in_bytes) override;
        void synchronize() override;
};

} // namespace cpu
} // namespace megcore

// vim: syntax=cpp.doxygen
