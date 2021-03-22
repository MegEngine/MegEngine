get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        LLVMSupport
        MLIROptLib
        MLIRIR
        MLIRPass
        MLIRSupport
        )
add_executable(mgb-opt mgb-opt.cpp)

target_include_directories(
  mgb-opt
  PRIVATE ${MLIR_LLVM_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src/jit/include
          ${PROJECT_BINARY_DIR}/src/jit/include)

add_dependencies(mgb-opt mgb_dialect)

target_link_libraries(mgb-opt PRIVATE ${LIBS} megbrain megdnn ${MGE_CUDA_LIBS})

llvm_update_compile_flags(mgb-opt)
