/**
 * \file src/jit/impl/mlir/ir/dialect.h
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

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/Function.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "megbrain/jit/mlir/ir/shape_inference_interface.h"

namespace mgb {
namespace jit {

class MgbDialect : public ::mlir::Dialect {
public:
    explicit MgbDialect(::mlir::MLIRContext* ctx);

    //! We should register this function in dialect
    static llvm::StringRef getDialectNamespace() { return "mgb::jit"; }
};

#define GET_OP_CLASSES
using namespace mlir;
#include "megbrain/jit/mlir/ir/ops.h.inc"

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
