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
}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
