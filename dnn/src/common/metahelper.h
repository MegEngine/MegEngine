#pragma once

namespace megdnn {
/*!
 * \brief base class for non-copyable objects
 */
class NonCopyableObj {
    NonCopyableObj(const NonCopyableObj&) = delete;
    NonCopyableObj& operator=(const NonCopyableObj&) = delete;

public:
    NonCopyableObj() = default;
};

}  // namespace megdnn

// vim: syntax=cpp.doxygen
