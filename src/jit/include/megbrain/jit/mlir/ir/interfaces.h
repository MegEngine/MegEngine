/**
 * \file src/jit/include/mlir/ir/interfaces.h
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
#if MGB_JIT_MLIR

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

namespace mlir {
/// Include the auto-generated declarations.
#include "megbrain/jit/mlir/ir/interfaces.h.inc"
}

#endif // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
