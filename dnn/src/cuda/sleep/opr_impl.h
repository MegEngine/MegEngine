#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace cuda {

class SleepForwardImpl : public SleepForward {
public:
    using SleepForward::SleepForward;

    void exec() override;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
