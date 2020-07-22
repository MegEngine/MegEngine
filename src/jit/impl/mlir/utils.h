/**
 * \file src/jit/impl/mlir/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megbrain/common.h"
#include "megbrain/exception.h"
#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"

#include <string>

#include <mlir/ExecutionEngine/CRunnerUtils.h>

#include <llvm/Support/raw_ostream.h>

namespace mgb {
namespace jit {

template <typename T>
std::string to_string(T&& t) {
    std::string ret;
    llvm::raw_string_ostream stream(ret);
    t.print(stream);
    return ret;
}

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
