#pragma once

#include "megcore_cambricon.h"
#include "src/common/megcore/common/computing_context.hpp"

namespace megcore {
namespace cambricon {

class CambriconComputingContext final : public ComputingContext {
public:
    CambriconComputingContext(
            megcoreDeviceHandle_t dev_handle, unsigned int flags,
            const CambriconContext& ctx = {});
    ~CambriconComputingContext();

    void memcpy(
            void* dst, const void* src, size_t size_in_bytes,
            megcoreMemcpyKind_t kind) override;
    void memcpy_peer_async_d2d(
            void* dst, int dst_dev, const void* src, int src_dev,
            size_t size_int_bytes);
    void memset(void* dst, int value, size_t size_in_bytes) override;
    void synchronize() override;

    const CambriconContext& context() const { return context_; }

    cnrtQueue_t queue() const { return context().queue; }

private:
    bool own_queue, own_cnnl_handle;
    CambriconContext context_;
};

}  // namespace cambricon
}  // namespace megcore

// vim: syntax=cpp.doxygen
