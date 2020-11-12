/**
 * \file src/jit/impl/mlir/ir/dialect.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megbrain/jit/mlir/ir/dialect.h"

#include "./types.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mgb;
using namespace jit;

MgbDialect::MgbDialect(mlir::MLIRContext* ctx)
        : mlir::Dialect("mgb", ctx, mlir::TypeID::get<MgbDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "megbrain/jit/mlir/ir/mgb_dialect.cpp.inc"
            >();
}

#define GET_OP_CLASSES
#include "megbrain/jit/mlir/ir/mgb_dialect.cpp.inc"

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
