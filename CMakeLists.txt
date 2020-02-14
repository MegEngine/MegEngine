cmake_minimum_required(VERSION 3.9.0)
project(MegEngine)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

if(NOT MSVC)
    set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> Dqc <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> Dq  <TARGET> <LINK_FLAGS> <OBJECTS>")
    set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> -D <TARGET>")
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-Wclass-memaccess CXX_SUPPORT_WCLASS_MEMACCESS)

set(MGE_ARCH AUTO CACHE STRING "Architecture on which MegEngine to be built.")
set_property(CACHE MGE_ARCH PROPERTY STRINGS AUTO
    x86_64 i386
    naive fallback
)


if(${MGE_ARCH} STREQUAL "AUTO")
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(MGE_ARCH "x86_64")
    elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i386" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i686")
        set(MGE_ARCH "i386")
    else()
        message(FATAL "Unknown machine architecture for MegEngine.")
    endif()
endif()

CHECK_CXX_COMPILER_FLAG(-fuse-ld=gold CXX_SUPPORT_GOLD)
if(CXX_SUPPORT_GOLD)
    message("-- Using GNU gold linker.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold")
endif()

option(MGE_WITH_JIT "Build MegEngine with JIT." ON)
option(MGE_WITH_HALIDE "Build MegEngine with Halide JIT" ON)
option(MGE_DISABLE_FLOAT16 "Disable MegEngine float16 support." OFF)
option(MGE_WITH_CUDA "Enable MegEngine CUDA support." ON)
option(MGE_CUDA_USE_STATIC "Enable MegEngine CUDA static linking." ON)
option(MGE_WITH_TRT "Build MegEngine with TensorRT." ON)
option(MGE_USE_SYSTEM_LIB "Build MegEngine with system libraries." OFF)
option(MGB_WITH_FLATBUFFERS "Build MegBrain with FlatBuffers serialization support." ON)

if(MGE_WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "CUDA compiler not found in PATH")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if(NOT MGE_WITH_CUDA)
    message("-- Disable JIT support, as CUDA is not enabled.")
    set(MGE_WITH_JIT OFF)
    set(MGE_WITH_HALIDE OFF)
    message("-- Disable TensorRT support, as CUDA is not enabled.")
    set(MGE_WITH_TRT OFF)
endif()

find_package(PythonInterp 3 REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)
if(${CMAKE_THREAD_LIBS_INIT} STREQUAL "-pthread" AND MGE_WITH_CUDA)
    set_property(TARGET Threads::Threads
        PROPERTY INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-pthread>"
        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-pthread>")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    add_definitions(-DMGB_HAVE_THREAD=1)
endif()


set(MGE_BLAS MKL CACHE STRING "BLAS implementaion used by MegEngine.")
set_property(CACHE MGE_BLAS PROPERTY STRINGS MKL OpenBLAS)
set(MGE_CUDA_GENCODE "" CACHE STRING "Overwrite -gencode specifications for CUDA")
if(NOT CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER $(CMAKE_CXX_COMPILER))
endif()

option(MGE_ENABLE_RTTI "Build with RTTI" ON)
option(MGE_ENABLE_LOGGING "Build with logging" ON)
option(MGE_DEBUG_UTIL "Enable debug utility" ON)

if(MGE_DEBUG_UTIL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMGB_ENABLE_DEBUG_UTIL=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMGB_ENABLE_DEBUG_UTIL=0")
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

if(NOT MGE_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

option(MGE_ENABLE_EXCEPTIONS "Build with exceptions" ON)
if(NOT MGE_ENABLE_EXCEPTIONS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exception")
endif()

# RTTI
if(MGE_ENABLE_RTTI)
    add_definitions(-DMEGDNN_ENABLE_MANGLING=0 -DMEGDNN_ENABLE_RTTI=1)
else()
    add_definitions(-DMEGDNN_ENABLE_MANGLING=1 -DMEGDNN_ENABLE_RTTI=0)
endif()

# Logging
if(MGE_ENABLE_LOGGING)
    add_definitions(-DMEGDNN_ENABLE_LOGGING=1 -DMGB_ENABLE_LOGGING=1 -DMGB_ENABLE_JSON=1)
else()
    add_definitions(-DMEGDNN_ENABLE_LOGGING=0 -DMGB_ENABLE_LOGGING=0 -DMGB_ENABLE_JSON=0)
endif()

# Exception
if(MGE_ENABLE_EXCEPTIONS)
    add_definitions(-DMEGDNN_ENABLE_EXCEPTIONS=1)
else()
    message(STATUS "Exceptions disabled; MegEngine would kill itself when it is supposed to throw an exception.")
    add_definitions(-DMEGDNN_ENABLE_EXCEPTIONS=0)
endif()

if(MGE_WITH_JIT AND MGE_WITH_HALIDE)
    set(HALIDE_SHARED_LIBRARY OFF CACHE BOOL "Build as a shared library")
    include(cmake/Halide.cmake)
    add_definitions(-DMGB_JIT_HALIDE=1)
endif()

option(MGE_WITH_TEST "Enable test for MegEngine." OFF)
if(MGE_WITH_TEST)
    include(cmake/gtest.cmake)
endif()

option(MGE_WITH_DISTRIBUTED "Build with distributed support" ON)

if(NOT MGE_WITH_CUDA)
    message("-- Disable distributed support, as CUDA is not enabled.")
    set(MGE_WITH_DISTRIBUTED OFF)
endif()

option(MGE_INFERENCE_ONLY "Build inference only library." OFF)
option(MGE_WITH_PYTHON_MODULE "Build MegEngine Python Module." ON)
if(MGE_INFERENCE_ONLY)
    message("-- Disable distributed support for inference only build.")
    set(MGE_WITH_DISTRIBUTED OFF)
    message("-- Disable python module for inference only build.")
    set(MGE_WITH_PYTHON_MODULE OFF)
    message("-- Disable tests for inference only build.")
    set(MGE_WITH_TEST OFF)
endif()

if(MGE_WITH_DISTRIBUTED)
    include(cmake/protobuf.cmake)
    include(cmake/zmq.cmake)
endif()

if(MGB_WITH_FLATBUFFERS)
    include(cmake/flatbuffers.cmake)
endif()

if(MSVC)
    add_compile_definitions(NOMINMAX=1 _USE_MATH_DEFINES=1 WIN32=1)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

if(MGE_WITH_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    foreach(path ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
        get_filename_component(_NAME ${path} NAME)
        if(NOT ${_NAME} STREQUAL "stubs")
            list(APPEND CUDA_LINK_DIRECTORIES ${path})
        endif()
    endforeach()
    link_directories(${CUDA_LINK_DIRECTORIES})

    set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O3 -g")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "-Os")
    set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall,-Wextra -Xfatbin -compress-all")

    if(NOT MGE_ENABLE_RTTI)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-rtti")
    endif()
    if(NOT MGE_ENABLE_EXCEPTIONS)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-exception")
    endif()

    if(NOT MGE_CUDA_GENCODE)
        if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMEGDNN_THREADS_512=0")
            if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=sm_75")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_75,code=compute_75")
            elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "9.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "9.0.0")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=sm_70")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_70,code=compute_70")
            else()
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_35,code=sm_35")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_52,code=sm_52")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_60,code=sm_60")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=sm_61")
                set(MGE_CUDA_GENCODE "${MGE_CUDA_GENCODE} -gencode arch=compute_61,code=compute_61")
            endif()
        else()
            message(FATAL_ERROR "Unsupported CUDA host arch.")
        endif()
    else()
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DMEGDNN_THREADS_512=1")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${MGE_CUDA_GENCODE}")
    include(cmake/cudnn.cmake)
    if(MGE_WITH_TRT)
        include(cmake/tensorrt.cmake)
    endif()
    if(MGE_CUDA_USE_STATIC)
        if(MGE_WITH_TRT)
            list(APPEND MGE_CUDA_LIBS -Wl,--whole-archive libnvinfer libcudnn -Wl,--no-whole-archive)
        else()
            list(APPEND MGE_CUDA_LIBS -Wl,--whole-archive libcudnn -Wl,--no-whole-archive)
        endif()
        list(APPEND MGE_CUDA_LIBS cusolver_static cublas_static curand_static culibos cudart_static cusparse_static)
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.1.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.1.0")
            list(APPEND MGE_CUDA_LIBS cublasLt_static)
        endif()
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.0.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.0.0")
            # mark all symbols from liblapack_static.a as weak to avoid
            # duplicated definition with mkl
            find_library(
                LAPACK_STATIC_PATH lapack_static
                HINTS ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
            if(NOT LAPACK_STATIC_PATH)
                message(FATAL_ERROR "liblapack_static.a not found")
            endif()
            set(LAPACK_STATIC_COPY_PATH ${CMAKE_CURRENT_BINARY_DIR}/liblapack_static_copy.a)

            # add a target that run objcopy
            add_custom_command(
                OUTPUT ${LAPACK_STATIC_COPY_PATH}
                COMMAND ${CMAKE_OBJCOPY} -w -W* ${LAPACK_STATIC_PATH} ${LAPACK_STATIC_COPY_PATH}
                VERBATIM)
            add_custom_target(lapack_static_weak_target DEPENDS ${LAPACK_STATIC_COPY_PATH})

            # create a library named "lapack_static_weak"
            add_library(lapack_static_weak STATIC IMPORTED GLOBAL)
            add_dependencies(lapack_static_weak lapack_static_weak_target)
            set_target_properties(
                lapack_static_weak PROPERTIES
                IMPORTED_LOCATION ${LAPACK_STATIC_COPY_PATH})
            list(APPEND MGE_CUDA_LIBS lapack_static_weak ${LAPACK_STATIC_COPY_PATH})
        endif()
    else()
        if(MGE_WITH_TRT)
            list(APPEND MGE_CUDA_LIBS libnvinfer)
        endif()
        list(APPEND MGE_CUDA_LIBS libcudnn)
        if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER "10.1.0" OR ${CMAKE_CUDA_COMPILER_VERSION} VERSION_EQUAL "10.1.0")
            list(APPEND MGE_CUDA_LIBS cublasLt cusolver cublas curand)
        endif()
    endif()

    add_subdirectory(dnn/cuda-stub)
    list(APPEND MGE_CUDA_LIBS nvrtc cuda-stub nvToolsExt)
    set(MGE_CUDA_LIBS "${MGE_CUDA_LIBS}")
endif()

find_program(CCACHE_BIN ccache)
if(CCACHE_BIN)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_BIN})
    if(MGE_WITH_CUDA AND NOT ${CMAKE_VERSION} VERSION_LESS "3.10.0")
        message("-- Using ccache as CMAKE_CUDA_COMPILER_LAUNCHER")
        set(CMAKE_CUDA_COMPILER_LAUNCHER ${CCACHE_BIN})
    endif()
endif()

if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
    if(${MGE_BLAS} STREQUAL "MKL")
        include(cmake/mkl.cmake)
        set(MGE_BLAS_LIBS libmkl)
    elseif(${MGE_BLAS} STREQUAL "OpenBLAS")
        include(cmake/OpenBLAS.cmake)
        set(MGE_BLAS_LIBS libopenblas)
    else()
        message(FATAL_ERROR "Unknown BLAS implementation ${MGE_BLAS}")
    endif()
endif()

option(MGE_WITH_MKLDNN "Enable Intel MKL_DNN support," ON)

# MKLDNN build
if(MGE_WITH_MKLDNN AND ${MGE_ARCH} STREQUAL "x86_64")
    add_definitions(-DMEGDNN_X86_WITH_MKL_DNN)
    include(cmake/MKL_DNN.cmake)
endif()


add_subdirectory(dnn)


set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DMGB_ASSERT_LOC=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DMGB_ASSERT_LOC=0")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DMGB_ASSERT_LOC=1")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DMGB_ASSERT_LOC=0")

if(MGE_ENABLE_RTTI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMGB_VERBOSE_TYPEINFO_NAME=1")
endif()

if(MGE_ENABLE_EXCEPTIONS)
    add_definitions(-DMGB_ENABLE_EXCEPTION=1)
else()
    add_definitions(-DMGB_ENABLE_EXCEPTION=0)
endif()

list(APPEND MGB_OPR_PARAM_DEFS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py)
set(MGB_OPR_PARAM_DEFS_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/dnn/scripts/gen_param_defs.py)

set(MGB_OPR_PARAM_DEFS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/opr/include/)
file(MAKE_DIRECTORY ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr)
add_custom_command(
    OUTPUT
        ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
    COMMAND ${PYTHON_EXECUTABLE} ${MGB_OPR_PARAM_DEFS_SCRIPT} ${MGB_OPR_PARAM_DEFS_SRCS}
        ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
    DEPENDS ${MGB_OPR_PARAM_DEFS_SRCS} ${MGB_OPR_PARAM_DEFS_SCRIPT}
    VERBATIM
)

list(APPEND MGB_OPR_PARAM_DEFS_OUTS
    ${MGB_OPR_PARAM_DEFS_OUT_DIR}/megbrain/opr/param_defs.h
)

install(FILES ${MGB_OPR_PARAM_DEFS_OUTS} DESTINATION include/megbrain/opr/)

list(APPEND MGB_OPR_PARAM_DEFS_INC ${MGB_OPR_PARAM_DEFS_OUT_DIR})
add_custom_target(_mgb_opr_param_defs DEPENDS ${MGB_OPR_PARAM_DEFS_OUTS})
add_library(mgb_opr_param_defs INTERFACE)
target_include_directories(mgb_opr_param_defs INTERFACE ${MGB_OPR_PARAM_DEFS_INC})
add_dependencies(mgb_opr_param_defs _mgb_opr_param_defs)

if(MGE_WITH_DISTRIBUTED)
    add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/MegRay)
endif()

add_subdirectory(src)
add_subdirectory(sdk/load-and-run)

if(MGE_WITH_PYTHON_MODULE)
    add_subdirectory(python_module)
endif()

if(MGE_WITH_TEST AND MGE_ENABLE_RTTI)
    add_subdirectory(test)
endif()

if(TARGET _mgb)
    add_custom_target(
        develop
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/python_module/megengine/_internal/$<TARGET_FILE_NAME:_mgb>
          ${CMAKE_CURRENT_SOURCE_DIR}/python_module/megengine/_internal/$<TARGET_FILE_NAME:_mgb>
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/python_module/megengine/_internal/mgb.py
          ${CMAKE_CURRENT_SOURCE_DIR}/python_module/megengine/_internal/mgb.py
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/python_module/megengine/_internal/opr.py
          ${CMAKE_CURRENT_SOURCE_DIR}/python_module/megengine/_internal/opr.py
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/python_module/megengine/_internal/opr_param_defs.py
          ${CMAKE_CURRENT_SOURCE_DIR}/python_module/megengine/_internal/opr_param_defs.py
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/python_module/megengine/_internal/include
          ${CMAKE_CURRENT_SOURCE_DIR}/python_module/megengine/_internal/include
        DEPENDS _mgb
        VERBATIM
    )
endif()

set(MGB_CUDA ${MGE_WITH_CUDA})
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
    set(MGB_ASSERT_LOC 1)
else()
    set(MGB_ASSERT_LOC 0)
endif()
set(MGB_ENABLE_DEBUG_UTIL ${MGE_DEBUG_UTIL})
set(MGB_ENABLE_LOGGING ${MGE_ENABLE_LOGGING})
set(MGB_VERBOSE_TYPEINFO_NAME ${MGE_ENABLE_RTTI})
set(MGB_ENABLE_EXCEPTION ${MGE_ENABLE_EXCEPTIONS})
set(MGB_JIT ${MGE_WITH_JIT})
set(MGB_JIT_HALIDE ${MGE_WITH_HALIDE})
set(MGB_ENABLE_TENSOR_RT ${MGE_WITH_TRT})
set(MGB_ENABLE_JSON ${MGE_ENABLE_LOGGING})
set(MGB_ENABLE_GRAD NOT ${MGE_INFERENCE_ONLY})
set(MGB_BUILD_SLIM_SERVING ${MGE_INFERENCE_ONLY})
configure_file(src/core/include/megbrain_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h)
file(READ src/core/include/megbrain_build_config.h _CONTENT)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h ${_CONTENT})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/genfiles/megbrain_build_config.h DESTINATION include)

