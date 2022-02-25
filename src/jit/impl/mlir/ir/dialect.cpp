#include "megbrain_build_config.h"
#if MGB_JIT && MGB_JIT_MLIR

#include "megbrain/jit/mlir/ir/dialect.h"

#include "./types.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Support/LogicalResult.h>

using namespace mgb;
using namespace jit;

MgbDialect::MgbDialect(mlir::MLIRContext* ctx)
        : mlir::Dialect("mgb", ctx, mlir::TypeID::get<MgbDialect>()) {
    addOperations<
#define GET_OP_LIST
#include "megbrain/jit/mlir/ir/mgb_dialect.cpp.inc"
            >();
}

#define GET_OP_CLASSES
#include "megbrain/jit/mlir/ir/mgb_dialect.cpp.inc"

#endif  // MGB_JIT && MGB_JIT_MLIR

// vim: syntax=cpp.doxygen
