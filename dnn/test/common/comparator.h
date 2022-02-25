#pragma once

namespace megdnn {
namespace test {

template <typename T>
class DefaultComparator {
public:
    bool is_same(T expected, T actual) const;
};

}  // namespace test
}  // namespace megdnn

#include "test/common/comparator.inl"

// vim: syntax=cpp.doxygen
