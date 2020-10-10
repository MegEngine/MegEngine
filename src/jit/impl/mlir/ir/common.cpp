/**
 * \file src/jit/impl/mlir/ir/common.cpp
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
#include "megbrain/jit/mlir/ir/utils.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>

using namespace mgb;
using namespace jit;

#define cb(name, op)                                                         \
    mlir::Value ValueBuilderHelper::name(mlir::Value lhs, mlir::Value rhs) { \
        return m_builder.create<mlir::op>(m_location, lhs, rhs);             \
    }
cb(add, AddFOp);
cb(sub, SubFOp);
cb(mul, MulFOp);
cb(div, DivFOp);
cb(divI, SignedDivIOp);
cb(mod, RemFOp);
cb(bit_and, AndOp);
cb(bit_or, OrOp);
cb(modI, SignedRemIOp);
#undef cb

#define cb(name, mode)                                                       \
    mlir::Value ValueBuilderHelper::name(mlir::Value lhs, mlir::Value rhs) { \
        return m_builder.create<mlir::CmpFOp>(                               \
                m_location, mlir::CmpFPredicate::mode, lhs, rhs);            \
    }
cb(gt, OGT);
cb(ge, OGE);
cb(lt, OLT);
cb(le, OLE);
cb(eq, OEQ);
#undef cb

mlir::Value ValueBuilderHelper::min(mlir::Value lhs, mlir::Value rhs) {
    mlir::Value cmp = m_builder.create<mlir::CmpFOp>(
            m_location, mlir::CmpFPredicate::OLT, lhs, rhs);
    return m_builder.create<mlir::SelectOp>(m_location, cmp, lhs, rhs);
}

mlir::Value ValueBuilderHelper::max(mlir::Value lhs, mlir::Value rhs) {
    mlir::Value cmp = m_builder.create<mlir::CmpFOp>(
            m_location, mlir::CmpFPredicate::OGT, lhs, rhs);
    return m_builder.create<mlir::SelectOp>(m_location, cmp, lhs, rhs);
}

mlir::Value ValueBuilderHelper::const_val(float val) {
    return m_builder.create<mlir::ConstantOp>(m_location,
                                              m_builder.getF32FloatAttr(val));
}

mlir::Value ValueBuilderHelper::constI(int32_t val) {
    return m_builder.create<mlir::ConstantOp>(m_location,
                                              m_builder.getIndexAttr(val));
}

#define cb(name, op)                                        \
    mlir::Value ValueBuilderHelper::name(mlir::Value lhs) { \
        return m_builder.create<mlir::op>(m_location, lhs); \
    }

cb(neg, NegFOp);
cb(ceil, CeilFOp);
cb(cos, CosOp);
cb(exp, ExpOp);
cb(exp2, Exp2Op);
cb(log10, Log10Op);
cb(log2, Log2Op);
cb(log, LogOp);
cb(rsqrt, RsqrtOp);
cb(sin, SinOp);
cb(sqrt, SqrtOp);
cb(tanh, TanhOp);
#undef cb

mlir::Value ValueBuilderHelper::abs(mlir::Value lhs) {
    auto zero = const_val(0.f);
    return select(ge(lhs, zero), lhs, sub(zero, lhs));
}

mlir::Value ValueBuilderHelper::floor(mlir::Value lhs) {
    //! FIXME use standard floor when upgrade llvm
    return neg(ceil(neg(lhs)));
}

mlir::Value ValueBuilderHelper::select(mlir::Value cond, mlir::Value true_val,
                                       mlir::Value false_val) {
    return m_builder.create<mlir::SelectOp>(m_location, cond, true_val,
                                            false_val);
}

mlir::AffineMap jit::get_affinemap(mlir::OpBuilder& builder,
                                   const mlir::Value& val,
                                   const megdnn::TensorLayout& layout) {
    auto type = val.getType().cast<mlir::MemRefType>();
    mgb_assert(type, "currently only support MemRefType");
    std::vector<mlir::AffineExpr> exprs;
    for (int i = 0; i < type.getRank(); ++i) {
        if (layout[i] == 1) {
            exprs.push_back(builder.getAffineConstantExpr(0));
        } else {
            exprs.push_back(builder.getAffineDimExpr(i));
        }
    }
    auto map = mlir::AffineMap::get(type.getRank(), 0, exprs,
                                    builder.getContext());
    return map;
}

mlir::Value jit::get_affine_load_op(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& val,
                               const mlir::ValueRange& index,
                               const megdnn::TensorLayout& dst) {
    if (val.getType().isa<mlir::MemRefType>()) {
        auto type = val.getType().cast<mlir::MemRefType>();
        megdnn::TensorLayout src_layout = mlir_type_to_layout(type);
        src_layout.init_contiguous_stride();
        if (src_layout.eq_shape(dst)) {
            return builder.create<mlir::AffineLoadOp>(loc, val, index);
        } else {
            auto lhs_map = get_affinemap(builder, val, src_layout);
            return builder.create<mlir::AffineLoadOp>(loc, val, lhs_map, index);
        }
    } else {
        return val;
    }
}

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
