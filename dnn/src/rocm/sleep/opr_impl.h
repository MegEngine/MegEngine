#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace rocm {

class SleepForwardImpl : public SleepForward {
public:
    using SleepForward::SleepForward;

    void exec() override;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
