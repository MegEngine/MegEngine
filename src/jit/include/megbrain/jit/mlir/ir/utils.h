#pragma once

#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megdnn/basic_types.h"
#include "megdnn/dtype.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Value.h>

namespace mgb {
namespace jit {

template <typename T>
std::string mlir_type_to_string(T&& t) {
    std::string ret;
    llvm::raw_string_ostream stream(ret);
    t.print(stream);
    return ret;
}

mlir::Value insert_alloc_and_dealloc(
        mlir::MemRefType type, mlir::Location loc, mlir::PatternRewriter& rewriter);

mlir::Type deduce_elemwise_res_type(mlir::ValueRange operands);

/**
 * \brief convert MLIR Type to TensorLayout
 */
megdnn::TensorLayout mlir_type_to_layout(mlir::Type type);

/**
 * \brief convert TensorLayout to MLIR Type
 */
mlir::MemRefType layout_to_mlir_type(
        const megdnn::TensorLayout& layout, mlir::Builder& builder);

}  // namespace jit
}  // namespace mgb

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
