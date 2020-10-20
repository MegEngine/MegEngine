if($ENV{LIBRARY_PATH})
    string(REPLACE ":" ";" SYSTEM_LIBRARY_PATHS $ENV{LIBRARY_PATH})
endif()

find_library(ACLRT_LIBRARY
    NAMES libascendcl.so
    PATHS $ENV{LD_LIBRARY_PATH} "$ENV{ACLRT_HOME}/lib64/stub" ${CMAKE_INSTALL_PREFIX}
    HINTS ${SYSTEM_LIBRARY_PATHS}
    PATH_SUFFIXES stub
    DOC "ACL library." )

if(ACLRT_LIBRARY STREQUAL "ACLRT_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find ACLRT Library")
endif()

get_filename_component(__found_aclrt_root "${ACLRT_LIBRARY}/../../../" REALPATH)
find_path(ACLRT_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS "$ENV{ACLRT_HOME}/include" ${__found_aclrt_root}
    PATH_SUFFIXES include
    DOC "Path to ACLRT include directory." )

if(ACLRT_INCLUDE_DIR STREQUAL "ACLRT_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find ACLRT Library")
endif()

add_library(libascendcl SHARED IMPORTED)

set_target_properties(libascendcl PROPERTIES
    IMPORTED_LOCATION ${ACLRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${ACLRT_INCLUDE_DIR}
)

message(STATUS "Found ACLRT: ${__found_aclrt_root}")

