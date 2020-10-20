if(NOT "$ENV{LIBRARY_PATH}" STREQUAL "")
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
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

