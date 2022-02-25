#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class SleepForwardImpl : public SleepForward {
public:
    using SleepForward::SleepForward;

    void exec() override;
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
