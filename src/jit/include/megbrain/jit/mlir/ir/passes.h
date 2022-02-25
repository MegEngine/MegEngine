#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include <mlir/IR/Module.h>
#include <mlir/Pass/Pass.h>
#include <memory>

namespace mgb {
namespace jit {

/**
 * \brief  Create a pass for lowering to operations in the `Affine` and `Std`
 * dialects, for a subset of the megbrain IR.
 */
std::unique_ptr<mlir::Pass> create_lower_to_affine_pass();

std::unique_ptr<mlir::Pass> create_lower_to_llvm_pass();

std::unique_ptr<mlir::Pass> create_lower_to_gpu_pass();

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
