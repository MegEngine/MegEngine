#pragma once

#include <mutex>
#include "megcore_cambricon.h"
#include "src/common/megcore/common/device_context.hpp"
#include "src/common/utils.h"

namespace megcore {
namespace cambricon {

class CambriconDeviceContext : public DeviceContext {
public:
    CambriconDeviceContext(int device_id, unsigned int flags);
    ~CambriconDeviceContext() noexcept;

    size_t mem_alignment_in_bytes() const noexcept override;

    void activate() override;
    void* malloc(size_t size_in_bytes) override;
    void free(void* ptr) override;

private:
    cnrtDeviceProp_t device_info;
};

}  // namespace cambricon
}  // namespace megcore

// vim: syntax=cpp.doxygen
