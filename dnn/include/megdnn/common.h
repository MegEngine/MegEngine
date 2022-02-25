#pragma once

#include "megbrain_build_config.h"
#include "megdnn/oprs/base.h"
#if MGB_ENABLE_GETENV
#define MGB_GETENV ::std::getenv
#else
#define MGB_GETENV(_name) static_cast<char*>(nullptr)
#endif

#ifdef WIN32
#define unsetenv(_name)                _putenv_s(_name, "");
#define setenv(name, value, overwrite) _putenv_s(name, value)
#endif

namespace megdnn {

/*!
 * \brief whether there is an algorithm from algo_pack() that is available for
 * current size
 */
template <class Opr, typename... Args>
bool has_available_algo(Opr* opr, Args&&... args) {
    auto&& all_algos = opr->get_all_algorithms_info(std::forward<Args>(args)...);
    return !all_algos.empty();
}

template <class Opr, typename... Args>
bool has_no_naive_heuristic_algo(Opr* opr, Args&&... args) {
    auto&& algo = opr->get_algorithm_info_heuristic(std::forward<Args>(args)...);
    return !static_cast<bool>(algo.attribute & detail::Algorithm::Attribute::NAIVE);
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
