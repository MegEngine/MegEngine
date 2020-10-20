if($ENV{LIBRARY_PATH})
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
endif()

find_library(CNDEV_LIBRARY 
    NAMES libcndev.so
    PATHS $ENV{LD_LIBRARY_PATH} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
    HINTS ${SYSTEM_LIBRARY_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "CNDEV library." )

if(CNDEV_LIBRARY STREQUAL "CNDEV_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find CNDEV Library")
endif()

get_filename_component(__found_cndev_root "${CNDEV_LIBRARY}/../include" REALPATH)
find_path(CNDEV_INCLUDE_DIR 
    NAMES cndev.h
    HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cndev_root}
    PATH_SUFFIXES include 
    DOC "Path to CNDEV include directory." )

if(CNDEV_INCLUDE_DIR STREQUAL "CNDEV_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find CNDEV Library")
endif()

file(STRINGS "${CNDEV_INCLUDE_DIR}/cndev.h" CNDEV_1 REGEX "^#define CNDEV_VERSION_1 [0-9]+.*$")
file(STRINGS "${CNDEV_INCLUDE_DIR}/cndev.h" CNDEV_2 REGEX "^#define CNDEV_VERSION_2 [0-9]+.*$")
file(STRINGS "${CNDEV_INCLUDE_DIR}/cndev.h" CNDEV_3 REGEX "^#define CNDEV_VERSION_3 [0-9]+.*$")
file(STRINGS "${CNDEV_INCLUDE_DIR}/cndev.h" CNDEV_4 REGEX "^#define CNDEV_VERSION_4 [0-9]+.*$")
file(STRINGS "${CNDEV_INCLUDE_DIR}/cndev.h" CNDEV_5 REGEX "^#define CNDEV_VERSION_5 [0-9]+.*$")

string(REGEX REPLACE "^#define CNDEV_VERSION_1 ([0-9]+).*$" "\\1" CNDEV_VERSION_1 "${CNDEV_1}")
string(REGEX REPLACE "^#define CNDEV_VERSION_2 ([0-9]+).*$" "\\1" CNDEV_VERSION_2 "${CNDEV_2}")
string(REGEX REPLACE "^#define CNDEV_VERSION_3 ([0-9]+).*$" "\\1" CNDEV_VERSION_3 "${CNDEV_3}")
string(REGEX REPLACE "^#define CNDEV_VERSION_4 ([0-9]+).*$" "\\1" CNDEV_VERSION_4 "${CNDEV_4}")
string(REGEX REPLACE "^#define CNDEV_VERSION_5 ([0-9]+).*$" "\\1" CNDEV_VERSION_5 "${CNDEV_5}")
set(CNDEV_VERSION_STRING "${CNDEV_VERSION_1}.${CNDEV_VERSION_2}.${CNDEV_VERSION_3}.${CNDEV_VERSION_4}.${CNDEV_VERSION_5}")

add_library(libcndev SHARED IMPORTED)

set_target_properties(libcndev PROPERTIES
    IMPORTED_LOCATION ${CNDEV_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CNDEV_INCLUDE_DIR}
)

message(STATUS "Found CNDEV: ${__found_cndev_root} (found version: ${CNDEV_VERSION_STRING})")

