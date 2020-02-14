/**
 * \file src/opr/test/loop/taylor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph.h"

#include <cmath>

using namespace mgb;

namespace mgb {
namespace test {
namespace loop {

    /*!
     *\brief calc sin(x) = sum((-1)^k * x^(1+2k) / (1+2k)!, k >= 0)
     */
    SymbolVar sin_by_taylor(SymbolVar x);

    /*!
     *\brief calc exp(x) = sum(x^k / k!, k >= 0)
     */
    SymbolVar exp_by_taylor(SymbolVar x);

} // namespace loop
} // namespace test
} // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

