#pragma once

#include <cstdlib>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {
template <typename Signature>
using thin_function = ::std::function<Signature>;

}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
