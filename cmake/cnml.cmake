if($ENV{LIBRARY_PATH})
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
endif()

find_library(CNML_LIBRARY 
    NAMES libcnml.so
    PATHS $ENV{LD_LIBRARY_PATH} "$ENV{NEUWARE_HOME}/lib64" ${CMAKE_INSTALL_PREFIX}
    HINTS ${SYSTEM_LIBRARY_PATHS}
    PATH_SUFFIXES lib lib64
    DOC "CNML library." )

if(CNML_LIBRARY STREQUAL "CNML_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find CNML Library")
endif()

get_filename_component(__found_cnml_root "${CNML_LIBRARY}/../include" REALPATH)
find_path(CNML_INCLUDE_DIR 
    NAMES cnml.h
    HINTS "$ENV{NEUWARE_HOME}/include" ${__found_cnml_root}
    PATH_SUFFIXES include 
    DOC "Path to CNML include directory." )

if(CNML_INCLUDE_DIR STREQUAL "CNML_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find CNML Library")
endif()

file(STRINGS "${CNML_INCLUDE_DIR}/cnml.h" CNML_MAJOR REGEX "^#define CNML_MAJOR_VERSION [0-9]+.*$")
file(STRINGS "${CNML_INCLUDE_DIR}/cnml.h" CNML_MINOR REGEX "^#define CNML_MINOR_VERSION [0-9]+.*$")
file(STRINGS "${CNML_INCLUDE_DIR}/cnml.h" CNML_PATCH REGEX "^#define CNML_PATCH_VERSION [0-9]+.*$")

string(REGEX REPLACE "^#define CNML_MAJOR_VERSION ([0-9]+).*$" "\\1" CNML_VERSION_MAJOR "${CNML_MAJOR}")
string(REGEX REPLACE "^#define CNML_MINOR_VERSION ([0-9]+).*$" "\\1" CNML_VERSION_MINOR "${CNML_MINOR}")
string(REGEX REPLACE "^#define CNML_PATCH_VERSION ([0-9]+).*$" "\\1" CNML_VERSION_PATCH "${CNML_PATCH}")
set(CNML_VERSION_STRING "${CNML_VERSION_MAJOR}.${CNML_VERSION_MINOR}.${CNML_VERSION_PATCH}")

add_library(libcnml SHARED IMPORTED)

set_target_properties(libcnml PROPERTIES
    IMPORTED_LOCATION ${CNML_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CNML_INCLUDE_DIR}
)

message(STATUS "Found CNML: ${__found_cnml_root} (found version: ${CNML_VERSION_STRING})")

