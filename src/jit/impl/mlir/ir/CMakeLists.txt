set(MGB_MLIR_TABLEGEN_INC_BASE ${CMAKE_CURRENT_BINARY_DIR}/include/)
file(MAKE_DIRECTORY ${MGB_MLIR_TABLEGEN_INC_BASE}/megbrain/jit/mlir/ir/)
list(APPEND MGB_MLIR_TABLEGEN_INC ${MGB_MLIR_TABLEGEN_INC_BASE})

external_tablegen_library(
    NAME
        mlir_shape_inference
    TBLGEN
        MLIR
    SRCS
        "interfaces.td"
    INCLUDES
        ${MGB_MLIR_TABLEGEN_INC} ${MLIR_LLVM_INCLUDE_DIR}
    OUTS
        -gen-op-interface-decls include/megbrain/jit/mlir/ir/interfaces.h.inc
        -gen-op-interface-defs include/megbrain/jit/mlir/ir/interfaces.cpp.inc
)

external_tablegen_library(
    NAME
        mlir_op_def
    TBLGEN
        MLIR
    SRCS
        "ops.td"
    INCLUDES
        ${MGB_MLIR_TABLEGEN_INC} ${MLIR_LLVM_INCLUDE_DIR}
    OUTS
        -gen-op-decls  include/megbrain/jit/mlir/ir/ops.h.inc
        -gen-op-defs   include/megbrain/jit/mlir/ir/ops.cpp.inc
)

# mgb_dialect
set(MGB_DIALECT_TD ${PROJECT_SOURCE_DIR}/src/jit/include/megbrain/jit/mlir/ir/mgb_dialect.td)
set(LLVM_TARGET_DEFINITIONS ${MGB_DIALECT_TD})
tablegen(MLIR mgb_dialect.h.inc ${MGE_IR_INCLUDE_DIRS} "--gen-op-decls")
tablegen(MLIR mgb_dialect.cpp.inc ${MGE_IR_INCLUDE_DIRS} "--gen-op-defs")
add_custom_target(mgb_dialect DEPENDS mgb_dialect.h.inc mgb_dialect.cpp.inc ${MGB_DIALECT_TD} ${MGE_IR_TDS})
add_dependencies(mgb_dialect param_defs_tblgen)
