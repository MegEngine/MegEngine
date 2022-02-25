#pragma once

#include "src/common/megcore/common/computing_context.hpp"
#include <memory>

namespace megcore {
std::unique_ptr<ComputingContext> make_rocm_computing_context(megcoreDeviceHandle_t dev_handle, unsigned int flags);
}
