find_path(MC20_ROOT_DIR
    include/ax_interpreter_external_api.h
    PATHS
    ${PROJECT_SOURCE_DIR}/third_party/mc20/
    $ENV{MC20DIR}
)

if(${MC20_ROOT_DIR} STREQUAL "MC20_ROOT_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find MC20")
endif()
message(STATUS "Build with MC20 in ${MC20_ROOT_DIR}")

find_path(MC20_INCLUDE_DIR
    ax_interpreter_external_api.h
    PATHS
    ${MC20_ROOT_DIR}/include
    ${INCLUDE_INSTALL_DIR}
)

add_library(libmc20 INTERFACE IMPORTED)
find_library(MC20_LIBRARY
    NAMES libax_interpreter_external.x86.a
    PATHS ${MC20_ROOT_DIR}/lib/)

if(${MC20_LIBRARY} STREQUAL "MC20_LIBRARY-NOTFOUND")
    message(FATAL_ERROR "Can not find MC20 library")
endif()
target_link_libraries(libmc20 INTERFACE ${MC20_LIBRARY})
target_include_directories(libmc20 INTERFACE ${MC20_INCLUDE_DIR})
