#pragma once

#include "src/common/megcore/common/device_context.hpp"
#include <memory>

namespace megcore {
std::unique_ptr<DeviceContext> make_rocm_device_context(int deviceID, unsigned int flags);
}
