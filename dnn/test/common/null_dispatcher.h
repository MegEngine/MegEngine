#pragma once

#include "megcore.h"

namespace megdnn {
namespace test {

class NullDispatcher final : public MegcoreCPUDispatcher {
public:
    ~NullDispatcher() {}
    void dispatch(Task&&) override {}
    void dispatch(MultiThreadingTask&&, size_t) override {}
    void sync() override {}
    size_t nr_threads() override { return 1; }
};

}  // namespace test
}  // namespace megdnn
