/**
 * \file src/jit/impl/mlir/ir/utils.cpp
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
#include "megbrain/exception.h"
#include "megbrain/jit/mlir/ir/utils.h"
#include "megdnn/oprs/general.h"
#include "megdnn/basic_types.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

using namespace mgb;
using namespace jit;

mlir::Value jit::insert_alloc_and_dealloc(mlir::MemRefType type,
                                          mlir::Location loc,
                                          mlir::PatternRewriter& rewriter) {
    auto alloc = rewriter.create<mlir::AllocOp>(loc, type);

    // Make sure to allocate at the beginning of the block.
    auto* parent_block = alloc.getOperation()->getBlock();
    alloc.getOperation()->moveBefore(&parent_block->front());

    // Make sure to deallocate this alloc at the end of the block. This is fine
    // as toy functions have no control flow.
    auto dealloc = rewriter.create<mlir::DeallocOp>(loc, alloc);
    dealloc.getOperation()->moveBefore(&parent_block->back());
    return alloc;
}

mlir::Type jit::deduce_result_type(mlir::ValueRange operands) {
    megdnn::TensorShapeArray srcs;
    megdnn::TensorShape dst;
    megdnn::DType dst_type;
    for (auto operand : operands) {
        if (operand.getType().isa<mlir::FloatType>()) {
            continue;
        }
        auto type = operand.getType().dyn_cast_or_null<mlir::MemRefType>();
        mgb_assert(type, "currently only support MemRefType");

        srcs.push_back(mlir_type_to_layout(type));
    }
    megdnn::Elemwise::deduce_shape(srcs, dst);
    mlir::Builder builder(operands[0].getContext());
    return layout_to_mlir_type({dst, mlir_type_to_dtype(operands[0].getType())},
                               builder);
}

megdnn::TensorLayout jit::mlir_type_to_layout(mlir::Type type) {
    megdnn::TensorLayout ret;
    if (type.isa<mlir::MemRefType>()) {
        auto real_type = type.dyn_cast_or_null<mlir::MemRefType>();
        mgb_assert(real_type);
        ret.ndim = real_type.getRank();
        for (size_t i = 0; i < ret.ndim; i++) {
            ret.shape[i] = real_type.getDimSize(i);
        }
        ret.dtype = mlir_type_to_dtype(real_type.getElementType());
    }
    return ret;
}

megdnn::DType jit::mlir_type_to_dtype(mlir::Type type) {
    mlir::Type element_type = type;
    if (auto cast = type.dyn_cast_or_null<mlir::MemRefType>()) {
        element_type = cast.getElementType();
    }
    switch (element_type.getKind()) {
        case mlir::StandardTypes::F32:
            return megdnn::dtype::Float32{};
        default:
            mgb_throw(InternalError,
                      "Unsupport mlir type for MemRefType, got: %s\n",
                      mlir_type_to_string(type).c_str());
    }
    return {};
}

mlir::MemRefType jit::layout_to_mlir_type(const megdnn::TensorLayout& layout,
                                          mlir::Builder& builder) {
    std::vector<int64_t> shape;
    for (size_t i = 0; i < layout.ndim; i++) {
        shape.push_back(layout[i]);
    }

    switch (layout.dtype.enumv()) {
        case megdnn::DTypeEnum::Float32:
            return mlir::MemRefType::get(shape, builder.getF32Type());
        default:
            mgb_throw(InternalError, "No supported dtype: %s",
                      layout.dtype.name());
    }
}

#endif  // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
