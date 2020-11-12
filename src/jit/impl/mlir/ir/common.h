/**
 * \file src/jit/impl/mlir/ir/common.h
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

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megbrain/tensor.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>

namespace mgb {
namespace jit {

/**
 * \brief Helper function for common value builder
 */
class ValueBuilderHelper {
public:
    ValueBuilderHelper(mlir::OpBuilder& b, mlir::Location location)
            : m_builder{b}, m_location{location} {};

#define cb(name)                                                              \
    mlir::Value name(mlir::ValueRange operands) { return name(operands[0]); } \
    mlir::Value name(mlir::Value lhs)

    // unary functions
    cb(abs);
    cb(ceil);
    cb(cos);
    cb(exp);
    cb(exp2);
    cb(floor);
    cb(log);
    cb(log10);
    cb(log2);
    cb(neg);
    cb(rsqrt);
    cb(sin);
    cb(sqrt);
    cb(tanh);

#undef cb

#define cb(name)                                  \
    mlir::Value name(mlir::ValueRange operands) { \
        return name(operands[0], operands[1]);    \
    }                                             \
    mlir::Value name(mlir::Value lhs, mlir::Value rhs)

    // binary functions
    cb(add);
    cb(bit_and);
    cb(bit_or);
    cb(div);
    cb(divI);
    cb(eq);
    cb(ge);
    cb(gt);
    cb(le);
    cb(lt);
    cb(max);
    cb(min);
    cb(mod);
    cb(modI);
    cb(mul);
    cb(sub);

#undef cb

    // constant functions
    mlir::Value const_f32(float val);
    mlir::Value const_i32(int32_t val);

    // select function
    mlir::Value select(mlir::Value cond, mlir::Value true_val,
                       mlir::Value false_val);

private:
    mlir::OpBuilder& m_builder;
    mlir::Location m_location;
};

template <typename Op>
mlir::Value get_operand(mlir::OpBuilder& builder, const mlir::Location& loc,
                        const mlir::Value& val, const mlir::ValueRange& index) {
    if (val.getType().isa<mlir::MemRefType>()) {
        return builder.create<Op>(loc, val, index);
    } else {
        return val;
    }
}

mlir::AffineMap get_affinemap(mlir::OpBuilder& builder, const mlir::Value& val,
                              const TensorLayout& layout);

mlir::Value get_affine_load_op(mlir::OpBuilder& builder,
                               const mlir::Location& loc,
                               const mlir::Value& val,
                               const mlir::ValueRange& index,
                               const TensorLayout& dst);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
