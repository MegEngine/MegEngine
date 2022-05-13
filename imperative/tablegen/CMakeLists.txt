set(MGB_OPDEF_OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
set(MGB_OPDEF_OPS_SRC ${CMAKE_SOURCE_DIR}/src/core/include/megbrain/ir/ops.td)
set(MGB_OPDEF_PARAMS_SRC ${CMAKE_SOURCE_DIR}/dnn/scripts/opr_param_defs.py)

# we set CMAKE_CONFIGURE_DEPENDS so that when source files or hash.txt was modified,
# cmake configure would be triggered to update ${MD5_MISMATCH}
execute_process(
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/checkhash.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ERROR_QUIET
  RESULT_VARIABLE MD5_MISMATCH)

if(${MD5_MISMATCH})
  # mgb tablegen executable
  set(TABLE_TARGET mgb-mlir-autogen)
  file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.h
       ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
  add_executable(${TABLE_TARGET} ${SRCS})
  target_include_directories(${TABLE_TARGET} PRIVATE ${MLIR_LLVM_INCLUDE_DIR})
  target_link_libraries(${TABLE_TARGET} PRIVATE LLVMTableGen MLIRTableGen LLVMSupport)
  set(MGB_TABLEGEN_EXE ${TABLE_TARGET})

  # generate megbrain opdef c header and python bindings basically same as
  # third_party/llvm-project/llvm/cmake/modules/TableGen.cmake but change output folder
  # and add extra dependency
  set(LLVM_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llvm-project/llvm)
  set(LLVM_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm)
  set(MLIR_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/llvm-project/mlir)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR}/third_party/llvm-project/mlir)
  set(MGB_TABLEGEN_INCLUDES
      -I${LLVM_SOURCE_DIR}/include
      -I${LLVM_BINARY_DIR}/include
      -I${MLIR_SOURCE_DIR}/include
      -I${MLIR_BINARY_DIR}/include
      -I${CMAKE_SOURCE_DIR}/src/core/include/megbrain/ir
      -I${CMAKE_BINARY_DIR}/src/core/include/megbrain/ir)
  set(MGB_TABLEGEN_FLAGS --write-if-changed)
  set(MGB_TABLEGEN_TARGETS)

  function(tablegen_opdef target output)
    add_custom_target(
      mgb_opdef_${target}
      COMMAND
        ${MGB_TABLEGEN_EXE} ${MGB_TABLEGEN_INCLUDES} --gen-${target}
        ${MGB_OPDEF_OPS_SRC} ${MGB_TABLEGEN_FLAGS} -o ${MGB_OPDEF_OUT_DIR}/${output}
      DEPENDS param_defs_tblgen)
    set(MGB_TABLEGEN_TARGETS
        ${MGB_TABLEGEN_TARGETS} mgb_opdef_${target}
        PARENT_SCOPE)
  endfunction()

  tablegen_opdef(cpp-header opdef.h.inl)
  tablegen_opdef(cpp-body opdef.cpp.inl)
  tablegen_opdef(python-binding opdef.py.inl)
  tablegen_opdef(python-c-extension opdef.cpy.inl)
  tablegen_opdef(enum-list-macro enum_macro.h)

  add_custom_target(
    mgb_opdef_genhash
    ${CMAKE_COMMAND} -P genhash.cmake
    DEPENDS ${MGB_TABLEGEN_TARGETS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(mgb_opdef DEPENDS ${MGB_TABLEGEN_TARGETS} mgb_opdef_genhash)
else()
  # add extra dependencies for auto reconfiguration
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
             ${MGB_OPDEF_OPS_SRC}
             ${MGB_OPDEF_PARAMS_SRC}
             generated/opdef.h.inl
             generated/opdef.cpp.inl
             generated/opdef.py.inl
             generated/opdef.cpy.inl
             generated/enum_macro
             generated/hash.txt)
  # additional check for safety
  add_custom_target(
    mgb_opdef_checkhash
    ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/checkhash.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_custom_target(mgb_opdef DEPENDS mgb_opdef_checkhash)
endif()

add_library(mgb_opdef_inc INTERFACE)
target_include_directories(mgb_opdef_inc INTERFACE ${MGB_OPDEF_OUT_DIR})
add_dependencies(mgb_opdef_inc mgb_opdef)
