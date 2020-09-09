/**
 * \file src/jit/impl/mlir/ir/types.h
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
#include <mlir/IR/StandardTypes.h>

namespace mgb {
namespace jit {

inline bool is_elemwise_float(const mlir::Type& dt) {
    if (auto cast = dt.dyn_cast_or_null<mlir::MemRefType>()) {
        if (cast.getElementType().getKind() == mlir::StandardTypes::F32) {
            return true;
        }
    }
    if (dt.isa<mlir::FloatType>()) {
        return true;
    }
    return false;
}

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
