#pragma once

#include "megcore.h"
#include "../common/computing_context.hpp"
#include <memory>

struct megcoreComputingContext {
    std::unique_ptr<megcore::ComputingContext> content;
};

// vim: syntax=cpp.doxygen
