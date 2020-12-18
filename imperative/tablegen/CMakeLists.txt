# mgb tablegen executable
set(TABLE_TARGET mgb-mlir-autogen)
add_executable(${TABLE_TARGET} autogen.cpp)
target_include_directories(${TABLE_TARGET} PRIVATE ${MLIR_LLVM_INCLUDE_DIR})
target_link_libraries(${TABLE_TARGET} PRIVATE LLVMTableGen MLIRTableGen LLVMSupport)
set(MGB_TABLEGEN_EXE ${TABLE_TARGET})

# generate megbrain opdef c header and python bindings
set(LLVM_TARGET_DEFINITIONS ${MGE_IR_DIR}/ops.td)
tablegen(MGB opdef.h.inl ${MGE_IR_INCLUDE_DIRS} "--gen-cpp-header")
tablegen(MGB opdef.cpp.inl ${MGE_IR_INCLUDE_DIRS} "--gen-cpp-body")
tablegen(MGB opdef.py.inl ${MGE_IR_INCLUDE_DIRS} "--gen-python-binding")
add_custom_target(mgb_opdef ALL DEPENDS opdef.h.inl opdef.cpp.inl opdef.py.inl param_defs_tblgen)
set(MGB_OPDEF_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
