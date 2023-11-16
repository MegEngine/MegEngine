find_library(
  CNDRV_LIBRARY
  NAMES libcndrv.so
  PATHS ${ALTER_LD_LIBRARY_PATHS} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
  HINTS ${ALTER_LIBRARY_PATHS}
  PATH_SUFFIXES lib lib64
  DOC "CNDRV library.")

if(CNDRV_LIBRARY STREQUAL "CNDRV_LIBRARY-NOTFOUND")
  message(FATAL_ERROR "Can not find CNDRV Library")
endif()

get_filename_component(__found_cndrv_root "${CNDRV_LIBRARY}/../.." REALPATH)
find_path(
  CNDRV_INCLUDE_DIR
  NAMES cn_api.h
  HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cndrv_root}
  PATH_SUFFIXES include
  DOC "Path to CNDRV include directory.")

if(CNDRV_INCLUDE_DIR STREQUAL "CNDRV_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Can not find CNDRV Library")
endif()

file(STRINGS "${CNDRV_INCLUDE_DIR}/cn_api.h" CNDRV_MAJOR
     REGEX "^#define CN_MAJOR_VERSION [0-9]+.*$")
file(STRINGS "${CNDRV_INCLUDE_DIR}/cn_api.h" CNDRV_MINOR
     REGEX "^#define CN_MINOR_VERSION [0-9]+.*$")
file(STRINGS "${CNDRV_INCLUDE_DIR}/cn_api.h" CNDRV_PATCH
     REGEX "^#define CN_PATCH_VERSION [0-9]+.*$")

string(REGEX REPLACE "^#define CN_MAJOR_VERSION ([0-9]+).*$" "\\1" CNDRV_VERSION_MAJOR
                     "${CNDRV_MAJOR}")
string(REGEX REPLACE "^#define CN_MINOR_VERSION ([0-9]+).*$" "\\1" CNDRV_VERSION_MINOR
                     "${CNDRV_MINOR}")
string(REGEX REPLACE "^#define CN_PATCH_VERSION ([0-9]+).*$" "\\1" CNDRV_VERSION_PATCH
                     "${CNDRV_PATCH}")
set(CNDRV_VERSION_STRING
    "${CNDRV_VERSION_MAJOR}.${CNDRV_VERSION_MINOR}.${CNDRV_VERSION_PATCH}")

add_library(libcndrv SHARED IMPORTED)

set_target_properties(
  libcndrv PROPERTIES IMPORTED_LOCATION ${CNDRV_LIBRARY} INTERFACE_INCLUDE_DIRECTORIES
                                                         ${CNDRV_INCLUDE_DIR})

message(
  STATUS "Found CNDRV: ${__found_cndrv_root} (found version: ${CNDRV_VERSION_STRING})")
