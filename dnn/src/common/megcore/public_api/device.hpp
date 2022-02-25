#pragma once
#include "megcore.h"
#include "../common/device_context.hpp"
#include <memory>

struct megcoreDeviceContext {
    std::unique_ptr<megcore::DeviceContext> content;
};

// vim: syntax=cpp.doxygen
