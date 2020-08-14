/**
 * \file src/jit/impl/mlir/ir/lower_to_affine_pass.cpp
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

#include "./common.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/Sequence.h>

using namespace mgb;
using namespace jit;

namespace {

using LoopIterationFn = function_ref<Value(
        OpBuilder& rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

void lower_op_to_loops(Operation* op, ValueRange operands,
                       PatternRewriter& rewriter,
                       LoopIterationFn process_iteration) {
    auto memref_type = (*op->result_type_begin()).cast<MemRefType>();
    auto loc = op->getLoc();

    auto alloc = jit::insert_alloc_and_dealloc(memref_type, loc, rewriter);

    SmallVector<int64_t, 4> lower_bounds(memref_type.getRank(), 0);
    SmallVector<int64_t, 4> steps(memref_type.getRank(), 1);
    buildAffineLoopNest(
            rewriter, loc, lower_bounds, memref_type.getShape(), steps,
            [&](OpBuilder& nested_builder, Location loc, ValueRange ivs) {
                Value value_to_store =
                        process_iteration(nested_builder, operands, ivs);
                nested_builder.create<AffineStoreOp>(loc, value_to_store, alloc,
                                                     ivs);
            });

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
}

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
    BinaryOpLowering(MLIRContext* ctx)
            : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lower_op_to_loops(
                op, operands, rewriter,
                [loc](OpBuilder& builder, ValueRange memref_operands,
                      ValueRange loop_ivs) {
                    typename BinaryOp::Adaptor binary_adaptor(memref_operands);

                    auto loaded_lhs = builder.create<AffineLoadOp>(
                            loc, binary_adaptor.lhs(), loop_ivs);
                    auto loaded_rhs = builder.create<AffineLoadOp>(
                            loc, binary_adaptor.rhs(), loop_ivs);

                    return builder.create<LoweredBinaryOp>(loc, loaded_lhs,
                                                           loaded_rhs);
                });
        return success();
    }
};
using AddOpLowering = BinaryOpLowering<jit::AddOp, AddFOp>;

struct AssignOpLowering : public ConversionPattern {
    AssignOpLowering(MLIRContext* ctx)
            : ConversionPattern(jit::AssignOp::getOperationName(), 1, ctx) {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        auto memref_type = operands[0].getType().cast<MemRefType>();
        AssignOpAdaptor assign_adaptor(operands);

        SmallVector<int64_t, 4> lower_bounds(memref_type.getRank(), 0);
        SmallVector<int64_t, 4> steps(memref_type.getRank(), 1);
        buildAffineLoopNest(
                rewriter, loc, lower_bounds, memref_type.getShape(), steps,
                [&](OpBuilder& nested_builder, Location loc, ValueRange ivs) {
                    auto loaded_lhs = nested_builder.create<AffineLoadOp>(
                            loc, assign_adaptor.lhs(), ivs);
                    nested_builder.create<AffineStoreOp>(
                            loc, loaded_lhs, assign_adaptor.rhs(), ivs);
                });

        rewriter.eraseOp(op);
        return success();
    }
};

struct ReturnOpLowering : public OpRewritePattern<jit::ReturnOp> {
    using OpRewritePattern<jit::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(jit::ReturnOp op,
                                  PatternRewriter& rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
        return success();
    }
};

class MgbToAffineLoweringPass
        : public PassWrapper<MgbToAffineLoweringPass, FunctionPass> {
public:
    void runOnFunction() override final {
        auto function = getFunction();

        // Verify that the given main has no inputs and results.
        if (function.getType().getNumResults()) {
            mgb_log_error("expected 'main' to have 0 results");
            return signalPassFailure();
        }

        ConversionTarget target(getContext());
        target.addLegalDialect<AffineDialect, StandardOpsDialect>();
        target.addIllegalDialect<MgbDialect>();

        OwningRewritePatternList patterns;
        patterns.insert<AddOpLowering, ReturnOpLowering, AssignOpLowering>(
                &getContext());

        if (failed(applyPartialConversion(getFunction(), target, patterns))) {
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> mgb::jit::create_lower_to_affine_pass() {
    return std::make_unique<MgbToAffineLoweringPass>();
}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
