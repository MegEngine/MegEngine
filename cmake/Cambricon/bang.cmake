find_path(
  NEUWARE_MODULE_DIR
  NAMES FindBANG.cmake
  PATHS "$ENV{NEUWARE_HOME}" "/usr/local/neuware"
  PATH_SUFFIXES cmake/modules
  DOC "Path to FindBang directory.")

if(NOT NEUWARE_MODULE_DIR)
  message(
    FATAL_ERROR
      "Can not find Cambrcion neuware cmake modules. Please set NEUWARE_HOME to specify the search path."
  )
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${NEUWARE_MODULE_DIR})

find_package(BANG)
if(NOT BANG_FOUND)
  message(FATAL_ERROR "Can not find BANG")
endif()

set(BANG_CNCC_FLAGS
    "${BANG_CNCC_FLAGS} --bang-mlu-arch=mtp_372 --bang-mlu-arch=mtp_592")
set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -std=c++11 -Werror -fPIC")
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g -O0")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -O3")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -g -O3")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "MinSizeRel")
  set(BANG_CNCC_FLAGS "${BANG_CNCC_FLAGS} -Os")
endif()

macro(BANG_INCLUDE_DIRECTORIES_V1)
  foreach(dir ${ARGN})
    list(APPEND BANG_CNCC_INCLUDE_ARGS -I${dir})
  endforeach()
endmacro()
