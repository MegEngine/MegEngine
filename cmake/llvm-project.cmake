# - Find the llvm/mlir libraries
# This module finds if llvm/mlir is installed, or build llvm/mlir from source.
# This module sets the following variables.
#
#  MLIR_LLVM_INCLUDE_DIR     - path to the LLVM/MLIR include files
#  MLIR_LLVM_LIBS            - path to the LLVM/MLIR libraries
#
# This module define the following functions.
#
# external_tablegen_library  - created interface library which depends on tablegen outputs

include(CMakeParseArguments)

function(external_tablegen_library)
    cmake_parse_arguments(
        _RULE
        "TESTONLY"
        "NAME;TBLGEN"
        "SRCS;INCLUDES;OUTS"
        ${ARGN}
        )

    if(_RULE_TESTONLY AND NOT MGE_WITH_TEST)
        return()
    endif()

    set(_NAME ${_RULE_NAME})

    set(LLVM_TARGET_DEFINITIONS ${_RULE_SRCS})
    set(_INCLUDE_DIRS ${_RULE_INCLUDES})
    list(TRANSFORM _INCLUDE_DIRS PREPEND "-I")
    set(_OUTPUTS)
    while(_RULE_OUTS)
        list(GET _RULE_OUTS 0 _COMMAND)
        list(REMOVE_AT _RULE_OUTS 0)
        list(GET _RULE_OUTS 0 _FILE)
        list(REMOVE_AT _RULE_OUTS 0)
        tablegen(${_RULE_TBLGEN} ${_FILE} ${_COMMAND} ${_INCLUDE_DIRS})
        list(APPEND _OUTPUTS ${CMAKE_CURRENT_BINARY_DIR}/${_FILE})
    endwhile()
    add_custom_target(${_NAME}_target DEPENDS ${_OUTPUTS})

    add_library(${_NAME} INTERFACE)
    add_dependencies(${_NAME} ${_NAME}_target)

    target_include_directories(${_NAME} INTERFACE
        "$<BUILD_INTERFACE:${_RULE_INCLUDES}>")

    install(TARGETS ${_NAME} EXPORT ${MGE_EXPORT_TARGETS})
endfunction()

set(LLVM_LIBS LLVMCore LLVMSupport LLVMX86CodeGen LLVMOrcJIT LLVMNVPTXCodeGen LLVMNVPTXDesc LLVMNVPTXInfo)
set(MLIR_CORE_LIBS MLIRAnalysis MLIRExecutionEngine MLIRIR MLIRParser MLIRPass MLIRSideEffectInterfaces MLIRTransforms)
set(MLIR_DIALECT_LIBS MLIRAsync MLIRAVX512 MLIRGPU MLIRLLVMAVX512 MLIRNVVMIR MLIROpenACC MLIRPDL MLIRPDLInterp MLIRQuant MLIRROCDLIR MLIRSDBM MLIRShape MLIRSPIRV MLIRStandardOpsTransforms MLIRTosa)
set(MLIR_CONVERSION_LIBS MLIRAffineToStandard MLIRAVX512ToLLVM MLIRGPUToGPURuntimeTransforms MLIRGPUToNVVMTransforms MLIRSCFToStandard)
set(MLIR_TRANSLATION_LIBS MLIRTargetLLVMIR MLIRTargetNVVMIR)
set(MLIR_LIBS ${MLIR_CORE_LIBS} ${MLIR_DIALECT_LIBS} ${MLIR_CONVERSION_LIBS} ${MLIR_TRANSLATION_LIBS})
set(MLIR_LLVM_LIBS ${LLVM_LIBS} ${MLIR_LIBS})

function(add_mge_mlir_src_dep llvm_monorepo_path)
    set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
    if(NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL)$")
        set(CMAKE_BUILD_TYPE "Debug")
    endif()
    set(_CMAKE_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    add_subdirectory("${llvm_monorepo_path}/llvm" ${LLVM_BUILD_DIR} EXCLUDE_FROM_ALL)

    # Reset CMAKE_BUILD_TYPE to its previous setting
    set(CMAKE_BUILD_TYPE "${_CMAKE_BUILD_TYPE}" CACHE STRING "Build type" FORCE)
    # Reset BUILD_SHARED_LIBS to its previous setting
    set(BUILD_SHARED_LIBS ${_CMAKE_BUILD_SHARED_LIBS} CACHE BOOL "Build shared libraries" FORCE)
endfunction()

# llvm build options
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(LLVM_ENABLE_RTTI ${MGE_ENABLE_RTTI} CACHE BOOL "" FORCE)
set(LLVM_TARGETS_TO_BUILD "X86;NVPTX;AArch64;ARM" CACHE STRING "" FORCE)
set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_BUILD_DIR ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm)

add_mge_mlir_src_dep("third_party/llvm-project")

set(MLIR_LLVM_INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${PROJECT_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${PROJECT_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include
    )
set(MLIR_TABLEGEN_EXE mlir-tblgen)
