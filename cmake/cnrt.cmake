if($ENV{LIBRARY_PATH})
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
endif()

find_library(CNRT_LIBRARY 
    NAMES libcnrt.so
    PATHS $ENV{LD_LIBRARY_PATH} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
    HINTS ${SYSTEM_LIBRARY_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "CNRT library." )

if(CNRT_LIBRARY STREQUAL "CNRT_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find CNRT Library")
endif()

get_filename_component(__found_cnrt_root "${CNRT_LIBRARY}/../include" REALPATH)
find_path(CNRT_INCLUDE_DIR 
    NAMES cnrt.h
    HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cnrt_root}
    PATH_SUFFIXES include 
    DOC "Path to CNRT include directory." )

if(CNRT_INCLUDE_DIR STREQUAL "CNRT_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find CNRT Library")
endif()

file(STRINGS "${CNRT_INCLUDE_DIR}/cnrt.h" CNRT_MAJOR REGEX "^#define CNRT_MAJOR_VERSION [0-9]+.*$")
file(STRINGS "${CNRT_INCLUDE_DIR}/cnrt.h" CNRT_MINOR REGEX "^#define CNRT_MINOR_VERSION [0-9]+.*$")
file(STRINGS "${CNRT_INCLUDE_DIR}/cnrt.h" CNRT_PATCH REGEX "^#define CNRT_PATCH_VERSION [0-9]+.*$")

string(REGEX REPLACE "^#define CNRT_MAJOR_VERSION ([0-9]+).*$" "\\1" CNRT_VERSION_MAJOR "${CNRT_MAJOR}")
string(REGEX REPLACE "^#define CNRT_MINOR_VERSION ([0-9]+).*$" "\\1" CNRT_VERSION_MINOR "${CNRT_MINOR}")
string(REGEX REPLACE "^#define CNRT_PATCH_VERSION ([0-9]+).*$" "\\1" CNRT_VERSION_PATCH "${CNRT_PATCH}")
set(CNRT_VERSION_STRING "${CNRT_VERSION_MAJOR}.${CNRT_VERSION_MINOR}.${CNRT_VERSION_PATCH}")

add_library(libcnrt SHARED IMPORTED)

set_target_properties(libcnrt PROPERTIES
    IMPORTED_LOCATION ${CNRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CNRT_INCLUDE_DIR}
)

message(STATUS "Found CNRT: ${__found_cnrt_root} (found version: ${CNRT_VERSION_STRING})")

