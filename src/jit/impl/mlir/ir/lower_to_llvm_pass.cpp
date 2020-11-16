/**
 * \file src/jit/impl/mlir/ir/lower_to_llvm_pass.cpp
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

#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>

using namespace mgb;
using namespace jit;

namespace {

class AffineToLLVMLoweringPass : public PassWrapper<AffineToLLVMLoweringPass,
                                                    OperationPass<ModuleOp>> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect>();
        registry.insert<mlir::scf::SCFDialect>();
    }

    void runOnOperation() final {
        LLVMConversionTarget target(getContext());
        target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

        LLVMTypeConverter typeConverter(&getContext());

        OwningRewritePatternList patterns;
        populateAffineToStdConversionPatterns(patterns, &getContext());
        populateLoopToStdConversionPatterns(patterns, &getContext());
        populateStdToLLVMConversionPatterns(typeConverter, patterns);
        populateExpandTanhPattern(patterns, &getContext());

        auto module = getOperation();
        if (failed(applyFullConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
}  // namespace

std::unique_ptr<mlir::Pass> mgb::jit::create_lower_to_llvm_pass() {
    return std::make_unique<AffineToLLVMLoweringPass>();
}

namespace mgb {
namespace jit {
void register_test_affine_to_llvm_lowering_pass() {
    PassRegistration<AffineToLLVMLoweringPass>(
            "mgb-codegen-convert-affine-to-llvm",
            "Perform final conversion from Affine to LLVMIR ",
            [] { return std::make_unique<AffineToLLVMLoweringPass>(); });
}
}  // namespace jit
}  // namespace mgb
#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
