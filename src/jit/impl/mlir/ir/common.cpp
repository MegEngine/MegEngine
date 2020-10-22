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

#include "mlir/Dialect/StandardOps/IR/Ops.h"

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
cb(mod, RemFOp);
cb(bit_and, AndOp);
cb(bit_or, OrOp);
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

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
