#pragma once

#include <iomanip>
#include <memory>
#include <sstream>

#include "megbrain/utils/metahelper.h"

namespace mgb {

namespace imperative {

template <typename T = std::function<void()>>
class CleanupGuard : public NonCopyableObj {
private:
    T m_callback;

public:
    explicit CleanupGuard(T cb) : m_callback{std::move(cb)} {}
    ~CleanupGuard() { m_callback(); }
};

inline std::string quoted(std::string str) {
    std::stringstream ss;
    ss << std::quoted(str);
    return ss.str();
}

}  // namespace imperative

}  // namespace mgb
