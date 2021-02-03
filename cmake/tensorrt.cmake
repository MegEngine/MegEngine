if(NOT "$ENV{LIBRARY_PATH}" STREQUAL "")
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
endif()

if("${TRT_ROOT_DIR}" STREQUAL "" AND NOT "$ENV{TRT_ROOT_DIR}"  STREQUAL "")
    set(TRT_ROOT_DIR $ENV{TRT_ROOT_DIR})
endif()

if(MGE_CUDA_USE_STATIC)
    find_library(TRT_LIBRARY 
        NAMES libnvinfer_static.a nvinfer.lib
        PATHS $ENV{LD_LIBRARY_PATH} ${TRT_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        HINTS ${SYSTEM_LIBRARY_PATHS}
        PATH_SUFFIXES lib lib64
        DOC "TRT library." )
else()
    find_library(TRT_LIBRARY 
        NAMES libnvinfer.so libnvinfer.dylib nvinfer.dll
        PATHS $ENV{LD_LIBRARY_PATH} ${TRT_ROOT_DIR} ${CMAKE_INSTALL_PREFIX}
        HINTS ${SYSTEM_LIBRARY_PATHS}
        PATH_SUFFIXES lib lib64
        DOC "TRT library." )
endif()

if(TRT_LIBRARY STREQUAL "TRT_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find TensorRT Library")
endif()

get_filename_component(__found_trt_root ${TRT_LIBRARY}/../.. REALPATH)
find_path(TRT_INCLUDE_DIR 
    NAMES NvInfer.h
    HINTS ${TRT_ROOT_DIR} ${CUDA_TOOLKIT_INCLUDE} ${__found_trt_root}
    PATH_SUFFIXES include 
    DOC "Path to TRT include directory." )

if(TRT_INCLUDE_DIR STREQUAL "TRT_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find TensorRT Library")
endif()

file(STRINGS "${TRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
file(STRINGS "${TRT_INCLUDE_DIR}/NvInfer.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
file(STRINGS "${TRT_INCLUDE_DIR}/NvInfer.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

if (TensorRT_MAJOR STREQUAL "")
    file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
    file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
    file(STRINGS "${TRT_INCLUDE_DIR}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")
endif()

string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
set(TRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")

if(MGE_CUDA_USE_STATIC)
    add_library(libnvinfer STATIC IMPORTED)
else()
    add_library(libnvinfer SHARED IMPORTED)
endif()

set_target_properties(libnvinfer PROPERTIES
    IMPORTED_LOCATION ${TRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TRT_INCLUDE_DIR}
)

message(STATUS "Found TensorRT: ${__found_trt_root} (found version: ${TRT_VERSION_STRING})")

if(TensorRT_VERSION_MAJOR GREATER_EQUAL 7)
    if(MGE_CUDA_USE_STATIC)
        find_library(LIBMYELIN_COMPILER
            NAMES libmyelin_compiler_static.a myelin_compiler_static.lib
            PATHS ${__found_trt_root}/lib
            )
        if(LIBMYELIN_COMPILER STREQUAL "LIBMYELIN_COMPILER-NOTFOUND")
            message(FATAL_ERROR "Can not find LIBMYELIN_COMPILER Library")
        else()
            message(STATUS "Found TensorRT myelin_compiler: ${LIBMYELIN_COMPILER}")
        endif()
        add_library(libmyelin_compiler STATIC IMPORTED)
        set_target_properties(libmyelin_compiler PROPERTIES
            IMPORTED_LOCATION ${LIBMYELIN_COMPILER}
            )

        find_library(LIBMYELIN_EXECUTOR
            NAMES libmyelin_executor_static.a myelin_executor_static.lib
            PATHS ${__found_trt_root}/lib
            )
        if(LIBMYELIN_EXECUTOR STREQUAL "LIBMYELIN_EXECUTOR-NOTFOUND")
            message(FATAL_ERROR "Can not find LIBMYELIN_EXECUTOR Library")
        else()
            message(STATUS "Found TensorRT libmyelin_executor: ${LIBMYELIN_EXECUTOR}")
        endif()
        add_library(libmyelin_executor STATIC IMPORTED)
        set_target_properties(libmyelin_executor PROPERTIES
            IMPORTED_LOCATION ${LIBMYELIN_EXECUTOR}
            )

        find_library(LIBMYELIN_PATTERN_RUNTIME
            NAMES libmyelin_pattern_runtime_static.a myelin_pattern_runtime_static.lib
            PATHS ${__found_trt_root}/lib
            )
        if(LIBMYELIN_PATTERN_RUNTIME STREQUAL "LIBMYELIN_PATTERN_RUNTIME-NOTFOUND")
            message(FATAL_ERROR "Can not find LIBMYELIN_PATTERN_RUNTIME Library")
        else()
            message(STATUS "Found TensorRT libmyelin_pattern_runtime: ${LIBMYELIN_PATTERN_RUNTIME}")
        endif()
        add_library(libmyelin_pattern_runtime STATIC IMPORTED)
        set_target_properties(libmyelin_pattern_runtime PROPERTIES
            IMPORTED_LOCATION ${LIBMYELIN_PATTERN_RUNTIME}
            )

        find_library(LIBMYELIN_PATTERN_LIBRARY
            NAMES libmyelin_pattern_library_static.a myelin_pattern_library_static.lib
            PATHS ${__found_trt_root}/lib
            )
        if(LIBMYELIN_PATTERN_LIBRARY STREQUAL "LIBMYELIN_PATTERN_LIBRARY-NOTFOUND")
            message(FATAL_ERROR "Can not find LIBMYELIN_PATTERN_LIBRARY Library")
        else()
            message(STATUS "Found TensorRT libmyelin_pattern_library: ${LIBMYELIN_PATTERN_LIBRARY}")
        endif()
        add_library(libmyelin_pattern_library STATIC IMPORTED)
        set_target_properties(libmyelin_pattern_library PROPERTIES
            IMPORTED_LOCATION ${LIBMYELIN_PATTERN_LIBRARY}
            )
    else()
        find_library(LIBMYELIN_SHARED
            NAMES libmyelin.so myelin.dll
            PATHS ${__found_trt_root}/lib
            )

        if(LIBMYELIN_SHARED STREQUAL "LIBMYELIN_SHARED-NOTFOUND")
            message(FATAL_ERROR "Can not find LIBMYELIN_SHARED Library")
        else()
            message(STATUS "Found TensorRT libmyelin_shared: ${LIBMYELIN_SHARED}")
        endif()
        add_library(libmyelin SHARED IMPORTED)
        set_target_properties(libmyelin PROPERTIES
            IMPORTED_LOCATION ${LIBMYELIN_SHARED}
            )
    endif()
endif()
