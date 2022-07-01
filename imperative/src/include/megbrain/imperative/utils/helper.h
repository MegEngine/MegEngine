#pragma once

#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>

#include "megbrain/utils/metahelper.h"

namespace mgb {

namespace imperative {

template <typename T = std::function<void()>>
class CleanupGuard : public NonCopyableObj {
private:
    std::optional<T> m_callback;

public:
    CleanupGuard() = default;
    explicit CleanupGuard(T cb) : m_callback{std::move(cb)} {}
    ~CleanupGuard() { reset(); }
    CleanupGuard(CleanupGuard&& rhs) : m_callback(std::move(rhs.m_callback)) {
        rhs.m_callback.reset();
    }
    CleanupGuard& operator=(CleanupGuard&& rhs) {
        swap(m_callback, rhs.m_callback);
        rhs.reset();
        return *this;
    }

public:
    void reset() {
        if (m_callback) {
            (*m_callback)();
            m_callback.reset();
        }
    }
};

inline std::string quoted(std::string str) {
    std::stringstream ss;
    ss << std::quoted(str);
    return ss.str();
}

#define MGE_CALL_ONCE(...)                                \
    do {                                                  \
        static std::once_flag _once_flag;                 \
        std::call_once(_once_flag, [&] { __VA_ARGS__; }); \
    } while (false)

template <typename T>
struct is_small_vector {
    static constexpr bool value = false;
};

template <typename T>
struct is_small_vector<SmallVector<T>> {
    static constexpr bool value = true;
};

template <typename T>
static constexpr bool is_small_vector_v = is_small_vector<T>::value;

}  // namespace imperative

}  // namespace mgb
