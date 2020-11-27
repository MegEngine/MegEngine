/**
 * \file src/jit/impl/mlir/ir/types.cpp
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

#include "./types.h"

#include "megbrain/common.h"
#include "megbrain/exception.h"
#include "megbrain/jit/mlir/ir/utils.h"

namespace mgb {
namespace jit {

mlir::Type megdnn_dtype_to_mlir_type(megdnn::DType type,
                                     mlir::MLIRContext* ctx) {
    switch (type.enumv()) {
        case megdnn::DTypeEnum::Float32:
            return mlir::FloatType::getF32(ctx);
        case megdnn::DTypeEnum::Uint8:
            return mlir::IntegerType::get(8, mlir::IntegerType::Unsigned, ctx);
        case megdnn::DTypeEnum::Int8:
            return mlir::IntegerType::get(8, mlir::IntegerType::Signed, ctx);
        case megdnn::DTypeEnum::Int16:
            return mlir::IntegerType::get(16, mlir::IntegerType::Signed, ctx);
        case megdnn::DTypeEnum::Int32:
            return mlir::IntegerType::get(32, mlir::IntegerType::Signed, ctx);
        case megdnn::DTypeEnum::IntB1:
            return mlir::IntegerType::get(1, ctx);
        case megdnn::DTypeEnum::IntB2:
            return mlir::IntegerType::get(2, ctx);
        case megdnn::DTypeEnum::IntB4:
            return mlir::IntegerType::get(4, ctx);
        case megdnn::DTypeEnum::Byte:
            return mlir::IntegerType::get(8, ctx);
        case megdnn::DTypeEnum::Float16:
            return mlir::FloatType::getF16(ctx);
        case megdnn::DTypeEnum::UintB4:
            return mlir::IntegerType::get(4, ctx);
        case megdnn::DTypeEnum::BFloat16:
            return mlir::FloatType::getBF16(ctx);
        case megdnn::DTypeEnum::Bool:
            return mlir::IntegerType::get(1, ctx);
        default:
            mgb_throw(InternalError, "Unsupported MegDNN dtype: %s",
                      type.name());
    }
}

mlir::Type signless(mlir::Type type) {
    if (auto intty = type.dyn_cast<mlir::IntegerType>()) {
        return mlir::IntegerType::get(intty.getWidth(), type.getContext());
    }
    return type;
}

megdnn::DType mlir_type_to_megdnn_dtype(mlir::Type type) {
    mlir::Type element_type = type;
    if (auto cast = type.dyn_cast_or_null<mlir::MemRefType>()) {
        element_type = cast.getElementType();
    }

    megdnn::DTypeEnum enumv;
    if (element_type.isF32()) {
        enumv = megdnn::DTypeEnum::Float32;
    } else if (element_type.isSignlessInteger(1)) {
        enumv = megdnn::DTypeEnum::IntB1;
    } else if (element_type.isSignlessInteger(2)) {
        enumv = megdnn::DTypeEnum::IntB2;
    } else if (element_type.isSignlessInteger(4)) {
        enumv = megdnn::DTypeEnum::IntB4;
    } else if (element_type.isSignlessInteger(8)) {
        enumv = megdnn::DTypeEnum::Int8;
    } else if (element_type.isSignlessInteger(16)) {
        enumv = megdnn::DTypeEnum::Int16;
    } else if (element_type.isSignlessInteger(32)) {
        enumv = megdnn::DTypeEnum::Int32;
    } else if (element_type.isF16()) {
        enumv = megdnn::DTypeEnum::Float16;
    } else if (element_type.isBF16()) {
        enumv = megdnn::DTypeEnum::BFloat16;
    } else if (element_type.isSignlessInteger(1)) {
        enumv = megdnn::DTypeEnum::Bool;
    } else {
        mgb_throw(InternalError, "Unsupported MLIR Type: %s",
                  mlir_type_to_string(element_type).c_str());
    }
    return megdnn::DType::from_enum(enumv);
}

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
