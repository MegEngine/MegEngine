/**
 * \file src/jit/impl/mlir/ir/shape_inference_pass.cpp
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

#include "megbrain/common.h"
#include "megbrain/jit/mlir/ir/dialect.h"
#include "megbrain/jit/mlir/ir/passes.h"
#include "megbrain/jit/mlir/ir/shape_inference_interface.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/Pass.h>

using namespace mgb;
using namespace jit;

#include "megbrain/jit/mlir/ir/shape_inference_interface.cpp.inc"

namespace {
class ShapeInferencePass
        : public mlir::PassWrapper<ShapeInferencePass, FunctionPass> {
public:
    void runOnFunction() override {
        auto f = getFunction();

        llvm::SmallPtrSet<mlir::Operation*, 16> op_worklist;
        f.walk([&](mlir::Operation* op) {
            if (returns_dynamic_shape(op))
                op_worklist.insert(op);
        });

        // Iterate on the operations in the worklist until all operations have
        // been inferred or no change happened (fix point).
        while (!op_worklist.empty()) {
            // Find the next operation ready for inference, that is an operation
            // with all operands already resolved (non-generic).
            auto nextop = llvm::find_if(op_worklist, all_operands_inferred);
            if (nextop == op_worklist.end())
                break;

            Operation* op = *nextop;
            op_worklist.erase(op);

            if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
                shapeOp.infer_shapes();
            } else {
                mgb_log_error(
                        "unable to infer shape of operation without shape "
                        "inference interface");
                return signalPassFailure();
            }
        }

        // If the operation worklist isn't empty, this indicates a failure.
        if (!op_worklist.empty()) {
            mgb_log_error(
                    "Shape inference failed, %zu operations couldn't be "
                    "inferred",
                    op_worklist.size());
            signalPassFailure();
        }
    }

    //! A utility method that returns if the given operation has all of its
    //! operands inferred.
    static bool all_operands_inferred(Operation* op) {
        return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
            return operandType.isa<mlir::MemRefType>();
        });
    }

    //! A utility method that returns if the given operation has a dynamically
    //! shaped result.
    static bool returns_dynamic_shape(Operation* op) {
        return llvm::any_of(op->getResultTypes(), [](Type resultType) {
            return !resultType.isa<mlir::MemRefType>();
        });
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> mgb::jit::create_shape_inference_pass() {
    return std::make_unique<ShapeInferencePass>();
}

#endif  // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
