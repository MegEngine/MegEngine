find_library(
  CNNL_LIBRARY
  NAMES libcnnl.so
  PATHS ${ALTER_LD_LIBRARY_PATHS} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
  HINTS ${ALTER_LIBRARY_PATHS}
  PATH_SUFFIXES lib lib64
  DOC "CNNL library.")

if(CNNL_LIBRARY STREQUAL "CNNL_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find CNNL Library")
endif()

get_filename_component(__found_cnnl_root "${CNNL_LIBRARY}/../.." REALPATH)
find_path(
  CNNL_INCLUDE_DIR
  NAMES cnnl.h
  HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cnnl_root}
  PATH_SUFFIXES include
  DOC "Path to CNNL include directory.")

if(CNNL_INCLUDE_DIR STREQUAL "CNNL_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find CNNL Library")
endif()

file(STRINGS "${CNNL_INCLUDE_DIR}/cnnl.h" CNNL_MAJOR
     REGEX "^#define CNNL_MAJOR [0-9]+.*$")
file(STRINGS "${CNNL_INCLUDE_DIR}/cnnl.h" CNNL_MINOR
     REGEX "^#define CNNL_MINOR [0-9]+.*$")
file(STRINGS "${CNNL_INCLUDE_DIR}/cnnl.h" CNNL_PATCH
     REGEX "^#define CNNL_PATCHLEVEL [0-9]+.*$")

string(REGEX REPLACE "^#define CNNL_MAJOR ([0-9]+).*$" "\\1" CNNL_VERSION_MAJOR
                     "${CNNL_MAJOR}")
string(REGEX REPLACE "^#define CNNL_MINOR ([0-9]+).*$" "\\1" CNNL_VERSION_MINOR
                     "${CNNL_MINOR}")
string(REGEX REPLACE "^#define CNNL_PATCHLEVEL ([0-9]+).*$" "\\1" CNNL_VERSION_PATCH
                     "${CNNL_PATCH}")
set(CNNL_VERSION_STRING
    "${CNNL_VERSION_MAJOR}.${CNNL_VERSION_MINOR}.${CNNL_VERSION_PATCH}")

add_library(libcnnl SHARED IMPORTED)

set_target_properties(
  libcnnl PROPERTIES IMPORTED_LOCATION ${CNNL_LIBRARY} INTERFACE_INCLUDE_DIRECTORIES
                                                       ${CNNL_INCLUDE_DIR})

message(
  STATUS "Found CNNL: ${__found_cnnl_root} (found version: ${CNNL_VERSION_STRING})")

find_library(
  CNNL_EXTRA_LIBRARY
  NAMES libcnnl_extra.so
  PATHS ${ALTER_LD_LIBRARY_PATHS} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
  HINTS ${ALTER_LIBRARY_PATHS}
  PATH_SUFFIXES lib lib64
  DOC "CNNL_EXTRA library.")

if(CNNL_EXTRA_LIBRARY STREQUAL "CNNL_EXTRA_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find CNNL_EXTRA Library")
endif()

get_filename_component(__found_cnnl_extra_root "${CNNL_EXTRA_LIBRARY}/../.." REALPATH)
find_path(
  CNNL_EXTRA_INCLUDE_DIR
  NAMES cnnl_extra.h
  HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cnnl_extra_root}
  PATH_SUFFIXES include
  DOC "Path to CNNL_EXTRA include directory.")

if(CNNL_EXTRA_INCLUDE_DIR STREQUAL "CNNL_EXTRA_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find CNNL_EXTRA Library")
endif()

file(STRINGS "${CNNL_EXTRA_INCLUDE_DIR}/cnnl_extra.h" CNNL_EXTRA_MAJOR
     REGEX "^#define CNNL_EXTRA_MAJOR [0-9]+.*$")
file(STRINGS "${CNNL_EXTRA_INCLUDE_DIR}/cnnl_extra.h" CNNL_EXTRA_MINOR
     REGEX "^#define CNNL_EXTRA_MINOR [0-9]+.*$")
file(STRINGS "${CNNL_EXTRA_INCLUDE_DIR}/cnnl_extra.h" CNNL_EXTRA_PATCH
     REGEX "^#define CNNL_EXTRA_PATCHLEVEL [0-9]+.*$")

string(REGEX REPLACE "^#define CNNL_EXTRA_MAJOR ([0-9]+).*$" "\\1"
                     CNNL_EXTRA_VERSION_MAJOR "${CNNL_EXTRA_MAJOR}")
string(REGEX REPLACE "^#define CNNL_EXTRA_MINOR ([0-9]+).*$" "\\1"
                     CNNL_EXTRA_VERSION_MINOR "${CNNL_EXTRA_MINOR}")
string(REGEX REPLACE "^#define CNNL_EXTRA_PATCHLEVEL ([0-9]+).*$" "\\1"
                     CNNL_EXTRA_VERSION_PATCH "${CNNL_EXTRA_PATCH}")
set(CNNL_EXTRA_VERSION_STRING
    "${CNNL_EXTRA_VERSION_MAJOR}.${CNNL_EXTRA_VERSION_MINOR}.${CNNL_EXTRA_VERSION_PATCH}"
)

add_library(libcnnl_extra SHARED IMPORTED)

set_target_properties(
  libcnnl_extra PROPERTIES IMPORTED_LOCATION ${CNNL_EXTRA_LIBRARY}
                           INTERFACE_INCLUDE_DIRECTORIES ${CNNL_EXTRA_INCLUDE_DIR})

message(
  STATUS
    "Found CNNL_EXTRA: ${__found_cnnl_extra_root} (found version: ${CNNL_EXTRA_VERSION_STRING})"
)
