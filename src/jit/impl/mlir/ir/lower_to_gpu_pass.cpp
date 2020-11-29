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

using Rewriter = ConversionPatternRewriter;
using Layout = megdnn::TensorLayout;

/* ===================== GpuLoweringHelper ===================== */

struct GpuLoweringHelper {
    GpuLoweringHelper(scf::ForOp* for_op, Value index, const Layout& dest)
            : m_for_op(for_op), m_index(index), m_dest(dest) {}

    void set_insertion_point(OpBuilder& builder) const {
        // insert before the last operation (scf.yield) in the loop body
        builder.setInsertionPoint(&(m_for_op->getLoopBody().front().back()));
    }

    std::vector<Value> map_indices(OpBuilder& builder, Location loc,
                                   Value value) const {
        auto type = value.getType().dyn_cast_or_null<MemRefType>();
        if (!type) {
            return {m_index};
        }

        std::vector<Value> indices(m_dest.ndim);
        ValueBuilderHelper helper(builder, loc);

        // map global index to multi-dimensional indices
        Value dim_index = m_index;
        for (int i = m_dest.ndim - 1; i >= 0; i--) {
            indices[i] = helper.modI(dim_index, helper.const_i32(m_dest[i]));
            dim_index = helper.divI(dim_index, helper.const_i32(m_dest[i]));
        }

        // allow broadcasting
        Layout src_layout = mlir_type_to_layout(type);
        src_layout.init_contiguous_stride();
        for (int i = 0; i < type.getRank(); ++i) {
            if (src_layout[i] == 1) {
                indices[i] = helper.const_i32(0);
            }
        }
        return indices;
    }

private:
    scf::ForOp* m_for_op;
    Value m_index;
    Layout m_dest;
};

/* ===================== conversion patterns ===================== */

struct AssignOpLowering : public ConversionPattern, public GpuLoweringHelper {
    AssignOpLowering(MLIRContext* ctx, scf::ForOp* for_op, mlir::Value index,
                     const Layout& dest)
            : ConversionPattern(dialect::AssignOp::getOperationName(), 2, ctx),
              GpuLoweringHelper(for_op, index, dest) {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                  Rewriter& rewriter) const final {
        auto loc = op->getLoc();
        set_insertion_point(rewriter);

        auto index = map_indices(rewriter, loc, operands[1]);
        auto input = get_operand<LoadOp>(rewriter, loc, operands[0], index);
        rewriter.create<StoreOp>(loc, input, operands[1], index);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ConstantScalarOpLowering
        : public OpRewritePattern<dialect::ConstantScalarOp>,
          public GpuLoweringHelper {
    ConstantScalarOpLowering(MLIRContext* ctx, scf::ForOp* for_op, Value index,
                             const Layout& dest)
            : OpRewritePattern<dialect::ConstantScalarOp>(ctx),
              GpuLoweringHelper(for_op, index, dest) {}

    LogicalResult matchAndRewrite(dialect::ConstantScalarOp op,
                                  PatternRewriter& rewriter) const final {
        set_insertion_point(rewriter);
        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, op.value());
        return success();
    }
};

struct DimshuffleLowering : public ConversionPattern, public GpuLoweringHelper {
    DimshuffleLowering(MLIRContext* ctx, scf::ForOp* for_op, Value index,
                       const Layout& dest)
            : ConversionPattern(dialect::Dimshuffle::getOperationName(), 1,
                                ctx),
              GpuLoweringHelper(for_op, index, dest) {}

    static std::vector<mlir::Value> get_index_from_pattern(
            const std::vector<int32_t>& pattern,
            const std::vector<mlir::Value>& index) {
        size_t ndim = *std::max_element(pattern.begin(), pattern.end()) + 1;
        std::vector<mlir::Value> res(ndim);
        for (size_t i = 0; i < pattern.size(); i++) {
            int32_t j = pattern[i];
            if (j >= 0) {
                res[j] = index[i];
            }
        }
        return res;
    }

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                  Rewriter& rewriter) const final {
        auto loc = op->getLoc();
        set_insertion_point(rewriter);

        auto pattern = llvm::dyn_cast<dialect::Dimshuffle>(op).pattern();
        auto index = map_indices(rewriter, loc, operands[0]);
        auto shuffled_index = get_index_from_pattern(pattern, index);

        rewriter.replaceOp(op, get_operand<LoadOp>(rewriter, loc, operands[0],
                                                   shuffled_index));
        return success();
    }
};

struct ElemwiseLowering : public ConversionPattern, public GpuLoweringHelper {
    ElemwiseLowering(MLIRContext* ctx, scf::ForOp* for_op, Value index,
                     const Layout& dest)
            : ConversionPattern(dialect::Elemwise::getOperationName(), 1, ctx),
              GpuLoweringHelper(for_op, index, dest) {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                  Rewriter& rewriter) const final {
        auto loc = op->getLoc();
        set_insertion_point(rewriter);

        // currently Elemwise handles at most three operands
        auto inputs = llvm::to_vector<4>(
                llvm::map_range(operands, [&](mlir::Value val) {
                    auto index = map_indices(rewriter, loc, val);
                    return get_operand<LoadOp>(rewriter, loc, val, index);
                }));

        rewriter.replaceOp(op,
                           lower_elemwise_to_std(op, rewriter, loc, inputs));
        return success();
    }
};

struct ReturnOpLowering : public ConversionPattern {
    ReturnOpLowering(MLIRContext* ctx, scf::ForOp*, Value, const Layout&)
            : ConversionPattern(dialect::ReturnOp::getOperationName(), 1, ctx) {
    }

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value>,
                                  Rewriter& rewriter) const final {
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op);
        return success();
    }
};

struct TypeCvtLowering : public ConversionPattern, public GpuLoweringHelper {
    TypeCvtLowering(MLIRContext* ctx, scf::ForOp* for_op, Value index,
                    const Layout& dest)
            : ConversionPattern(dialect::TypeCvt::getOperationName(), 1, ctx),
              GpuLoweringHelper(for_op, index, dest) {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                  Rewriter& rewriter) const final {
        auto loc = op->getLoc();
        set_insertion_point(rewriter);

        auto index = map_indices(rewriter, loc, operands[0]);
        auto input = get_operand<LoadOp>(rewriter, loc, operands[0], index);

        rewriter.replaceOp(op, lower_typecvt_to_std(op, rewriter, loc, input));
        return success();
    }
};

/* ===================== MgbToGpuLoweringPass ===================== */

class MgbToGpuLoweringPass
        : public PassWrapper<MgbToGpuLoweringPass, FunctionPass> {
public:
    void getDependentDialects(DialectRegistry& registry) const override;
    void runOnFunction() final;

private:
    Value get_idx(OpBuilder& builder, Location loc);
    Layout get_dest_layout(FuncOp func_op);
};

void MgbToGpuLoweringPass::getDependentDialects(
        DialectRegistry& registry) const {
    registry.insert<gpu::GPUDialect, scf::SCFDialect, StandardOpsDialect>();
}

void MgbToGpuLoweringPass::runOnFunction() {
    FuncOp func_op = getFunction();
    Location loc = func_op.getLoc();
    OpBuilder builder(func_op.getBody());

    // create gpu::LaunchOp
    Value one = builder.create<ConstantIndexOp>(loc, 1);
    gpu::LaunchOp launch_op =
            builder.create<gpu::LaunchOp>(loc, one, one, one, one, one, one);
    builder.setInsertionPointToEnd(&(launch_op.body().front()));

    // create scf::ForOp
    auto it = func_op.getArguments().end();
    Value nr_threads = *(--it);
    Value nr_elements = *(--it);
    Value idx = get_idx(builder, loc);
    auto for_op = builder.create<scf::ForOp>(loc, idx, nr_elements, nr_threads);

    builder.create<gpu::TerminatorOp>(loc);

    Layout dest = get_dest_layout(func_op);
    Value for_idx = for_op.getLoopBody().getArgument(0);

    OwningRewritePatternList patterns;
    patterns.insert<AssignOpLowering, ConstantScalarOpLowering,
                    DimshuffleLowering, ElemwiseLowering, ReturnOpLowering,
                    TypeCvtLowering>(&getContext(), &for_op, for_idx, dest);

    ConversionTarget target(getContext());
    target.addLegalDialect<gpu::GPUDialect, scf::SCFDialect,
                           StandardOpsDialect>();
    target.addIllegalDialect<MgbDialect>();

    if (failed(applyPartialConversion(func_op, target, std::move(patterns)))) {
        signalPassFailure();
    }
}

//! block_dim * block_idx + thread_idx
Value MgbToGpuLoweringPass::get_idx(OpBuilder& builder, Location loc) {
    IndexType idx_type = builder.getIndexType();
    StringAttr x = builder.getStringAttr("x");

    Value block_dim = builder.create<gpu::BlockDimOp>(loc, idx_type, x);
    Value block_idx = builder.create<gpu::BlockIdOp>(loc, idx_type, x);
    Value thread_idx = builder.create<gpu::ThreadIdOp>(loc, idx_type, x);

    Value prod = builder.create<MulIOp>(loc, block_dim, block_idx);
    return builder.create<AddIOp>(loc, prod, thread_idx);
}

//! traverse the body of func_op and get dest_layout from AssignOp
Layout MgbToGpuLoweringPass::get_dest_layout(FuncOp func_op) {
    Layout dest_layout;
    bool found = false;
    func_op.walk([&](dialect::AssignOp assign_op) {
        dest_layout = mlir_type_to_layout(assign_op.lhs().getType());
        found = true;
        return WalkResult::interrupt();
    });
    mgb_assert(found, "AssignOp not found in the body of FuncOp");
    return dest_layout;
}

}  // namespace

/* ===================== create_lower_to_gpu_pass ===================== */

std::unique_ptr<mlir::Pass> mgb::jit::create_lower_to_gpu_pass() {
    return std::make_unique<MgbToGpuLoweringPass>();
}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
