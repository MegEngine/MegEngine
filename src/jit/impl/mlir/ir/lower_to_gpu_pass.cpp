/**
 * \file src/jit/impl/mlir/ir/lower_to_gpu_pass.cpp
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

#include <llvm/ADT/PointerUnion.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/IR/Type.h>
#include <mlir/Dialect/GPU/GPUDialect.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/EDSC/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace mgb;
using namespace jit;

namespace {

mlir::Value get_tid(ConversionPatternRewriter& rewriter, const Location& loc) {
    auto thread_idx = rewriter.create<gpu::ThreadIdOp>(
            loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    auto block_idx = rewriter.create<gpu::BlockIdOp>(
            loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    auto group_size = rewriter.create<gpu::BlockDimOp>(
            loc, rewriter.getIndexType(), rewriter.getStringAttr("x"));
    Value index = rewriter.create<AddIOp>(
            loc, thread_idx,
            rewriter.create<MulIOp>(loc, block_idx, group_size));

    return index;
}

megdnn::TensorLayout output_layout(gpu::LaunchOp& launch_op) {
    auto func_op = launch_op.getParentOfType<mlir::FuncOp>();
    mgb_assert(func_op, "Unexpexted launch op.");
    for (auto block_iter = func_op.rbegin(); block_iter != func_op.rend();
         block_iter++) {
        for (auto op_iter = block_iter->rbegin(); op_iter != block_iter->rend();
             op_iter++) {
            auto op = llvm::dyn_cast_or_null<dialect::AssignOp>(&(*op_iter));
            if (op && op.getNumOperands() > 0) {
                return mlir_type_to_layout(*(op.operand_type_begin()));
            }
        }
    }
    mgb_throw(MegBrainError, "Unexpexted launch op.");
}

std::vector<mlir::Value> get_multidim_tid(ConversionPatternRewriter& rewriter,
                                          const Location& loc,
                                          const mlir::Value& val,
                                          const megdnn::TensorLayout& dst) {
    Value index = get_tid(rewriter, loc);

    auto type = val.getType().dyn_cast_or_null<mlir::MemRefType>();
    if (type) {
        ValueBuilderHelper helper(rewriter, loc);
        std::vector<mlir::Value> idxs;
        idxs.resize(dst.ndim);
        mlir::Value dim_index = index;
        for (int i = dst.ndim - 1; i >= 0; i--) {
            auto cur_index = helper.modI(dim_index, helper.const_i32(dst[i]));
            idxs[i] = cur_index;
            dim_index = helper.divI(dim_index, helper.const_i32(dst[i]));
        }

        megdnn::TensorLayout src_layout = mlir_type_to_layout(type);
        src_layout.init_contiguous_stride();
        for (int i = 0; i < type.getRank(); ++i) {
            if (src_layout[i] == 1) {
                idxs[i] = helper.const_i32(0);
            }
        }
        return idxs;
    } else {
        return {index};
    }
}

struct ElemwiseLowering : public ConversionPattern {
    ElemwiseLowering(MLIRContext* ctx, gpu::LaunchOp& launch_op)
            : ConversionPattern(dialect::Elemwise::getOperationName(), 1, ctx),
              m_launch_op{launch_op} {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();

        rewriter.setInsertionPointToEnd(&(m_launch_op.body().front()));

        auto dst_layout = output_layout(m_launch_op);
        auto inputs = llvm::to_vector<4>(
                llvm::map_range(operands, [&](mlir::Value val) {
                    auto index =
                            get_multidim_tid(rewriter, loc, val, dst_layout);
                    return get_operand<LoadOp>(rewriter, loc, val, index);
                }));

        rewriter.replaceOp(op,
                           lower_elemwise_to_std(op, rewriter, loc, inputs));
        return success();
    }

private:
    gpu::LaunchOp& m_launch_op;
};

struct TypeCvtLowering : public ConversionPattern {
    TypeCvtLowering(MLIRContext* ctx, gpu::LaunchOp& launch_op)
            : ConversionPattern(dialect::TypeCvt::getOperationName(), 1, ctx),
              m_launch_op{launch_op} {}
    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();

        rewriter.setInsertionPointToEnd(&(m_launch_op.body().front()));

        auto dst_layout = output_layout(m_launch_op);
        auto index = get_multidim_tid(rewriter, loc, operands[0], dst_layout);
        auto input = get_operand<LoadOp>(rewriter, loc, operands[0], index);

        rewriter.replaceOp(op, lower_typecvt_to_std(op, rewriter, loc, input));
        return success();
    }

private:
    gpu::LaunchOp& m_launch_op;
};

struct ReturnOpLowering : public ConversionPattern {
    ReturnOpLowering(MLIRContext* ctx, gpu::LaunchOp& launch_op)
            : ConversionPattern(dialect::ReturnOp::getOperationName(), 1, ctx),
              m_launch_op{launch_op} {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value>,
            ConversionPatternRewriter& rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
        auto loc = op->getLoc();

        //! remove the first gpu.terminator
        m_launch_op.body().front().front().erase();

        //! if (tid >= nr_tid) {return;} in the begin of the block
        rewriter.setInsertionPointToStart(&(m_launch_op.body().front()));
        Block* cond_block = rewriter.getInsertionBlock();
        Block::iterator op_position = rewriter.getInsertionPoint();
        Block* remaining_ops_block =
                rewriter.splitBlock(cond_block, op_position);
        rewriter.setInsertionPointToEnd(cond_block);

        auto index = get_tid(rewriter, loc);
        auto comparison = rewriter.create<mlir::CmpIOp>(
                loc, CmpIPredicate::sge, index,
                m_launch_op.getParentOfType<mlir::FuncOp>()
                        .getArguments()
                        .back());

        Block* then_block =
                rewriter.splitBlock(cond_block, rewriter.getInsertionPoint());
        rewriter.setInsertionPointToEnd(then_block);
        rewriter.create<gpu::TerminatorOp>(loc);

        rewriter.setInsertionPointToEnd(cond_block);
        rewriter.create<mlir::CondBranchOp>(
                loc, comparison, then_block, ArrayRef<Value>(),
                remaining_ops_block, ArrayRef<Value>());

        rewriter.setInsertionPointToEnd(remaining_ops_block);
        rewriter.create<gpu::TerminatorOp>(loc);

        return success();
    }

private:
    gpu::LaunchOp& m_launch_op;
};

struct ConstantScalarOpLowering
        : public OpRewritePattern<dialect::ConstantScalarOp> {
    ConstantScalarOpLowering(MLIRContext* ctx, gpu::LaunchOp& launch_op)
            : OpRewritePattern<dialect::ConstantScalarOp>(ctx),
              m_launch_op{launch_op} {}

    LogicalResult matchAndRewrite(dialect::ConstantScalarOp op,
                                  PatternRewriter& rewriter) const final {
        dialect::ConstantScalarOpAdaptor constant_scalar_adaptor(op);
        rewriter.setInsertionPointToEnd(&(m_launch_op.body().front()));

        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(
                op, constant_scalar_adaptor.value());
        return success();
    }

private:
    gpu::LaunchOp& m_launch_op;
};

struct AssignOpLowering : public ConversionPattern {
    AssignOpLowering(MLIRContext* ctx, gpu::LaunchOp& launch_op)
            : ConversionPattern(dialect::AssignOp::getOperationName(), 2, ctx),
              m_launch_op{launch_op} {}

    LogicalResult matchAndRewrite(
            Operation* op, ArrayRef<Value> operands,
            ConversionPatternRewriter& rewriter) const final {
        auto loc = op->getLoc();

        dialect::AssignOpAdaptor assign_adaptor(operands);
        rewriter.setInsertionPointToEnd(&(m_launch_op.body().front()));

        auto dst_layout = output_layout(m_launch_op);
        auto index = get_multidim_tid(rewriter, loc, assign_adaptor.rhs(),
                                      dst_layout);

        auto loaded_lhs =
                get_operand<LoadOp>(rewriter, loc, assign_adaptor.lhs(), index);
        rewriter.create<StoreOp>(loc, loaded_lhs, assign_adaptor.rhs(), index);

        rewriter.eraseOp(op);
        return success();
    }

private:
    gpu::LaunchOp& m_launch_op;
};

class MgbToGpuLoweringPass
        : public PassWrapper<MgbToGpuLoweringPass, FunctionPass> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override {
        registry.insert<mlir::gpu::GPUDialect>();
        registry.insert<mlir::StandardOpsDialect>();
    }

    void runOnFunction() override final {
        auto func_op = getFunction();
        Location loc = func_op.getLoc();
        OpBuilder builder(&func_op.getBody());
        Value constantOne = builder.create<ConstantIndexOp>(loc, 1);
        gpu::LaunchOp launch_op = builder.create<gpu::LaunchOp>(
                loc, constantOne, constantOne, constantOne, constantOne,
                constantOne, constantOne);
        builder.setInsertionPointToEnd(&(launch_op.body().front()));
        builder.create<gpu::TerminatorOp>(loc);

        OwningRewritePatternList patterns;
        ConversionTarget target(getContext());
        target.addLegalDialect<StandardOpsDialect>();
        target.addLegalDialect<gpu::GPUDialect>();
        target.addIllegalDialect<MgbDialect>();

        patterns.insert<ElemwiseLowering, TypeCvtLowering, ReturnOpLowering,
                        ConstantScalarOpLowering, AssignOpLowering>(
                &getContext(), launch_op);

        if (failed(applyPartialConversion(func_op, target,
                                          std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}  // namespace

std::unique_ptr<mlir::Pass> mgb::jit::create_lower_to_gpu_pass() {
    return std::make_unique<MgbToGpuLoweringPass>();
}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
