/**
 * \file src/jit/impl/mlir/ir/passes.h
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

#include <mlir/IR/Module.h>
#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include <memory>

#include <mlir/Pass/Pass.h>

namespace mgb {
namespace jit {

std::unique_ptr<mlir::Pass> create_shape_inference_pass();

/**
 * \brief  Create a pass for lowering to operations in the `Affine` and `Std`
 * dialects, for a subset of the megbrain IR.
 */
std::unique_ptr<mlir::Pass> create_lower_to_affine_pass();

std::unique_ptr<mlir::Pass> create_lower_to_llvm_pass();

std::unique_ptr<mlir::Pass> create_lower_to_gpu_pass();

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
