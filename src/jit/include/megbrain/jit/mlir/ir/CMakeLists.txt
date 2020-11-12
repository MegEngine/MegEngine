# mgb_dialect
set(LLVM_TARGET_DEFINITIONS mgb_dialect.td)
tablegen(MLIR mgb_dialect.h.inc ${MGE_IR_INCLUDE_DIRS} "--gen-op-decls")
tablegen(MLIR mgb_dialect.cpp.inc ${MGE_IR_INCLUDE_DIRS} "--gen-op-defs")
add_custom_target(mgb_dialect DEPENDS mgb_dialect.h.inc mgb_dialect.cpp.inc)
add_dependencies(mgb_dialect param_defs_tblgen)
