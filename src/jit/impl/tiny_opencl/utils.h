#include "megbrain_build_config.h"

#if MGB_JIT && MGB_OPENCL

#include "megbrain/jit/compiler.h"

template <typename T, typename S>
T safe_icast(S val) {
    static_assert(
            std::is_integral<S>::value && std::is_integral<T>::value, "must be int");
    mgb_assert(
            val <= static_cast<S>(std::numeric_limits<T>::max()) &&
            val >= static_cast<S>(0));
    return static_cast<T>(val);
}

template <typename S>
int safe_int(S val) {
    return safe_icast<int>(val);
}

enum class LayoutType {
    SCALAR = 0,
    CHANNEL_BROADCAST = 1,
    VEC = 2,
};

/*!
 * \brief get inputs channel broadcast info
 * \param args of mgb::jit::JITExecutor::Args
 * \return input idx is channel boardcast
 */
std::vector<LayoutType> get_channel_broadcast_info(
        const mgb::jit::JITExecutor::Args& args);

#endif
