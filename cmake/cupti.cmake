if("${CUDA_ROOT_DIR}" STREQUAL "" AND NOT "$ENV{CUDA_ROOT_DIR}" STREQUAL "")
  set(CUDA_ROOT_DIR $ENV{CUDA_ROOT_DIR})
endif()
if("${CUDA_ROOT_DIR}" STREQUAL "" AND NOT "$ENV{CUDA_PATH}" STREQUAL "")
  set(CUDA_ROOT_DIR $ENV{CUDA_PATH})
endif()
if("${CUDA_ROOT_DIR}" STREQUAL "" AND NOT "$ENV{CUDA_BIN_PATH}" STREQUAL "")
  set(CUDA_ROOT_DIR $ENV{CUDA_BIN_PATH})
endif()
# ${CUDA_ROOT_DIR} check removed here because users may not always keep env variable

# TODO: find_library(CUDA_ROOT_DIR) in cmake/cuda.cmake

set(MGE_CUPTI_USE_STATIC ${MGE_CUDA_USE_STATIC})

# relates https://stackoverflow.com/questions/67485114
if(${MGE_CUDA_USE_STATIC} AND ${CXX_SUPPORT_GOLD})
  message(WARNING "static linking CuPTI with gold may break exception handling,\
                  use shared one instead")
  set(MGE_CUPTI_USE_STATIC OFF)
endif()

if(MGE_CUPTI_USE_STATIC)
  find_library(
    CUPTI_LIBRARY
    NAMES libcupti_static.a
    HINTS ${CUDA_ROOT_DIR} ${CUDA_ROOT_DIR}/extras/CUPTI
    PATH_SUFFIXES lib lib64
    DOC "CuPTI library.")

  if("${CUPTI_LIBRARY}" STREQUAL "CUPTI_LIBRARY-NOTFOUND")
    message(WARNING "Can not find static CuPTI Library, use shared one instead")
    set(MGE_CUPTI_USE_STATIC OFF)
  endif()
endif()

if(NOT ${MGE_CUPTI_USE_STATIC})
  find_library(
    CUPTI_LIBRARY
    NAMES libcupti.so
    HINTS ${CUDA_ROOT_DIR} ${CUDA_ROOT_DIR}/extras/CUPTI
    PATH_SUFFIXES lib lib64
    DOC "CuPTI library.")
  set(CUPTI_LIBRARY_TYPE SHARED)
else()
  set(CUPTI_LIBRARY_TYPE STATIC)
endif()

if("${CUPTI_LIBRARY}" STREQUAL "CUPTI_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find CuPTI Library")
endif()

find_path(
  CUPTI_INCLUDE_DIR
  NAMES cupti.h
  HINTS ${CUDA_ROOT_DIR} ${CUDA_ROOT_DIR}/extras/CUPTI
  PATH_SUFFIXES include
  DOC "Path to CuPTI include directory.")

if(CUPTI_INCLUDE_DIR STREQUAL "CUPTI_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find CuPTI INCLUDE")
endif()

if(EXISTS ${CUPTI_INCLUDE_DIR}/cupti_version.h)
  file(READ ${CUPTI_INCLUDE_DIR}/cupti_version.h CUPTI_VERSION_FILE_CONTENTS)
else()
  file(READ ${CUPTI_INCLUDE_DIR}/cupti.h CUPTI_VERSION_FILE_CONTENTS)
endif()

string(REGEX MATCH "define CUPTI_API_VERSION * +([0-9]+)" CUPTI_API_VERSION
             "${CUPTI_VERSION_FILE_CONTENTS}")
string(REGEX REPLACE "define CUPTI_API_VERSION * +([0-9]+)" "\\1" CUPTI_API_VERSION
                     "${CUPTI_API_VERSION}")

add_library(libcupti ${CUPTI_LIBRARY_TYPE} IMPORTED)

set_target_properties(
  libcupti PROPERTIES IMPORTED_LOCATION ${CUPTI_LIBRARY} INTERFACE_INCLUDE_DIRECTORIES
                                                         ${CUPTI_INCLUDE_DIR})

message(STATUS "Found CuPTI: ${CUPTI_LIBRARY} (found version: ${CUPTI_API_VERSION})")
