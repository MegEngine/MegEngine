/**
 * \file src/jit/impl/mlir/ir/types.h
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

#include "megdnn/dtype.h"

#include <mlir/IR/StandardTypes.h>

namespace mgb {
namespace jit {

#define FOR_EACH_DNN_DTYPE(cb)                     \
    cb(Float32, dt_float32);                       \
    cb(Uint8, dt_uint8);                           \
    cb(Int8, dt_int8);                             \
    cb(Int16, dt_int16);                           \
    cb(Int32, dt_int32);                           \
    cb(Byte, dt_byte);                             \
    MEGDNN_INC_FLOAT16(cb(Float16, dt_float16));   \
    MEGDNN_INC_FLOAT16(cb(BFloat16, dt_bfloat16)); \
    cb(Bool, dt_bool);

mlir::Type megdnn_dtype_to_mlir_type(megdnn::DType type,
                                     mlir::MLIRContext* ctx);
mlir::Type signless(mlir::Type type);

megdnn::DType mlir_type_to_megdnn_dtype(mlir::Type type);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
