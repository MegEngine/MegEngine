/**
 * \file src/jit/impl/mlir/mlir_gen.h
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

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megbrain/jit/executor_opr.h"
#include "megbrain/jit/internal_graph.h"

#include <mlir/IR/Module.h>

namespace mgb {
namespace jit {

/**
 * \brief generate mlir from subgraph.
 *
 * \param context mlir context
 * \param internal_graph internal graph used to generate mlir
 * \param args input args for the internal graph
 * \return A pair of {kernel_name, module}
 **/
std::pair<llvm::StringRef, mlir::OwningModuleRef> mlir_gen(
        mlir::MLIRContext& context, const InternalGraph& internal_graph,
        const JITExecutor::Args& args);
}
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
