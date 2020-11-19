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

#include "./common.h"
#include "./each_mode.h"

#include "megbrain/common.h"
#include "megbrain/jit/mlir/ir/dialect.h"
#include "megbrain/jit/mlir/ir/passes.h"
#include "megbrain/jit/mlir/ir/utils.h"

#include <llvm/ADT/Sequence.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

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

    llvm::SmallVector<int64_t, 4> lower_bounds(memref_type.getRank(), 0);
    llvm::SmallVector<int64_t, 4> steps(memref_type.getRank(), 1);
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

struct ElemwiseLowering : public ConversionPattern {
    ElemwiseLowering(MLIRContext* ctx)
            : ConversionPattern(mgb::dialect::Elemwise::getOperationName(), 1,
                                ctx) {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        auto dst_memref_type = (*op->result_type_begin()).cast<MemRefType>();
        megdnn::TensorLayout dst_layout = mlir_type_to_layout(dst_memref_type);
        dst_layout.init_contiguous_stride();
        lower_op_to_loops(
                op, operands, rewriter,
                [dst_layout, loc, op](OpBuilder& builder,
                                      ValueRange memref_operands,
                                      ValueRange loop_ivs) {
                    auto inputs = llvm::to_vector<4>(llvm::map_range(
                            memref_operands, [&](mlir::Value val) {
                                return get_affine_load_op(builder, loc, val,
                                                          loop_ivs, dst_layout);
                            }));
                    return lower_elemwise_to_std(op, builder, loc, inputs);
                });
        return success();
    }
};

struct TypeCvtLowering : public ConversionPattern {
    TypeCvtLowering(MLIRContext* ctx)
            : ConversionPattern(mgb::dialect::TypeCvt::getOperationName(), 1,
                                ctx) {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        lower_op_to_loops(
                op, operands, rewriter,
                [loc, op](OpBuilder& builder, ValueRange memref_operands,
                          ValueRange loop_ivs) {
                    mlir::Value input = get_operand<AffineLoadOp>(
                            builder, loc, memref_operands[0], loop_ivs);
                    return lower_typecvt_to_std(op, builder, loc, input);
                });
        return success();
    }
};

struct DimshuffleLowering : public ConversionPattern {
    DimshuffleLowering(MLIRContext* ctx)
            : ConversionPattern(mgb::dialect::Dimshuffle::getOperationName(), 1,
                                ctx) {}

    static mlir::AffineMap get_affinemap_from_pattern(
            const std::vector<int32_t>& pattern, mlir::MLIRContext* ctx) {
        size_t ndim = *std::max_element(pattern.begin(), pattern.end()) + 1;
        std::vector<mlir::AffineExpr> exprs(ndim);
        for (size_t i = 0; i < pattern.size(); i++) {
            int32_t j = pattern[i];
            if (j >= 0) {
                exprs[j] = mlir::getAffineDimExpr(i, ctx);
            }
        }
        return mlir::AffineMap::get(pattern.size(), 0, exprs, ctx);
    }

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        auto pattern = llvm::dyn_cast<dialect::Dimshuffle>(op).pattern();
        auto map = get_affinemap_from_pattern(pattern, op->getContext());
        lower_op_to_loops(
                op, operands, rewriter,
                [loc, op, &map](OpBuilder& builder, ValueRange memref_operands,
                                ValueRange loop_ivs) {
                    return builder.create<AffineLoadOp>(loc, memref_operands[0],
                                                        map, loop_ivs);
                });
        return success();
    }
};

struct AssignOpLowering : public ConversionPattern {
    AssignOpLowering(MLIRContext* ctx)
            : ConversionPattern(dialect::AssignOp::getOperationName(), 1, ctx) {
    }

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();
        auto memref_type = operands[0].getType().cast<MemRefType>();
        dialect::AssignOpAdaptor assign_adaptor(operands);

        llvm::SmallVector<int64_t, 4> lower_bounds(memref_type.getRank(), 0);
        llvm::SmallVector<int64_t, 4> steps(memref_type.getRank(), 1);
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

struct ReturnOpLowering : public OpRewritePattern<dialect::ReturnOp> {
    using OpRewritePattern<dialect::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(dialect::ReturnOp op,
                                  PatternRewriter& rewriter) const final {
        // We lower "mgb.return" directly to "std.return".
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
        return success();
    }
};

struct ConstantScalarOpLowering
        : public OpRewritePattern<dialect::ConstantScalarOp> {
    using OpRewritePattern<dialect::ConstantScalarOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(dialect::ConstantScalarOp op,
                                  PatternRewriter& rewriter) const final {
        dialect::ConstantScalarOpAdaptor constant_scalar_adaptor(op);
        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(
                op, constant_scalar_adaptor.value());
        return success();
    }
};

class MgbToAffineLoweringPass
        : public PassWrapper<MgbToAffineLoweringPass, FunctionPass> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<mlir::AffineDialect>();
        registry.insert<mlir::StandardOpsDialect>();
    }

    void runOnFunction() override final {
        ConversionTarget target(getContext());
        target.addLegalDialect<AffineDialect, StandardOpsDialect>();
        // target.addLegalDialect<AffineDialect>();
        target.addIllegalDialect<MgbDialect>();

        OwningRewritePatternList patterns;
        patterns.insert<ElemwiseLowering, TypeCvtLowering, DimshuffleLowering,
                        ReturnOpLowering, AssignOpLowering,
                        ConstantScalarOpLowering>(&getContext());

        if (failed(applyPartialConversion(getFunction(), target,
                                          std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> mgb::jit::create_lower_to_affine_pass() {
    return std::make_unique<MgbToAffineLoweringPass>();
}

namespace mgb {
namespace jit {
void register_test_mgb_to_affine_lowering_pass() {
    PassRegistration<MgbToAffineLoweringPass>(
            "mgb-convert-to-affine",
            "Perform conversion from MGB Dialect to Affine Dialect ",
            [] { return std::make_unique<MgbToAffineLoweringPass>(); });
}
}  // namespace jit
}  // namespace mgb
#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
