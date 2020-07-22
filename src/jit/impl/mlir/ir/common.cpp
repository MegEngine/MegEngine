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

#include "common.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>

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

#endif  // MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
