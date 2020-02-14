/**
 * \file dnn/include/megdnn/thin/function.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <type_traits>
#include <functional>
#include <utility>
#include <memory>
#include <cstdlib>

#include "megdnn/internal/visibility_prologue.h"

namespace megdnn {
template<typename Signature>
using thin_function = ::std::function<Signature>;

} // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
