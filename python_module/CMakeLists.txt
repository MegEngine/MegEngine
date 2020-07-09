cmake_policy(SET CMP0086 NEW)

find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT REQUIRED)

find_package(Git)
if(GIT_FOUND)
  message("git found: ${GIT_EXECUTABLE}")
endif()

find_package(NumPy REQUIRED)

find_package(SWIG REQUIRED)
set(SWIG_SRC src/swig/mgb.i)
if (APPLE)
    set(CMAKE_SWIG_FLAGS -Wall -threads -py3 -modern)
else()
    set(CMAKE_SWIG_FLAGS -Wall -threads -py3 -modern -DSWIGWORDSIZE64)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

file(GLOB_RECURSE OPR_DECL_SRCS "${PROJECT_SOURCE_DIR}/src/**/*.oprdecl")
file(GLOB_RECURSE PYTHON_SRCS setup.py
    src/python/*.py
    test/*.py
    megengine/*.py)
list(REMOVE_ITEM PYTHON_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/megengine/_internal/mgb.py
    ${CMAKE_CURRENT_SOURCE_DIR}/megengine/_internal/opr.py
    ${CMAKE_CURRENT_SOURCE_DIR}/megengine/_internal/opr_param_defs.py
)
list(APPEND PYTHON_SRCS ${MGB_SRCS})

file(GLOB_RECURSE ALL_HEADERS src/cpp/megbrain_pubapi.h
    ${PROJECT_SOURCE_DIR}/src/core/include/*
    ${PROJECT_SOURCE_DIR}/src/opr/include/*
    ${PROJECT_SOURCE_DIR}/src/serialization/include/*
    ${PROJECT_SOURCE_DIR}/src/plugin/include/*
    ${PROJECT_SOURCE_DIR}/dnn/include/*)

file(COPY ${PROJECT_SOURCE_DIR}/dnn/scripts/opr_param_defs.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(READ ${PROJECT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py CONTENTS)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/opr_param_defs.py ${CONTENTS})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/opr.py ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/opr_param_defs.py
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/python ${CMAKE_CURRENT_BINARY_DIR}/src/python
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/src/python/genopr.py ${OPR_DECL_SRCS}
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/dnn/scripts/gen_param_defs.py -t py ${CMAKE_CURRENT_BINARY_DIR}/opr_param_defs.py ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/opr_param_defs.py
    DEPENDS ${OPR_DECL_SRCS}
    VERBATIM
)

add_custom_target(mgb_opr_py DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/opr.py)

set(SRCS src/cpp/craniotome.cpp src/cpp/function_replace.cpp src/cpp/intbx.cpp src/cpp/bfloat16.cpp src/cpp/megbrain_config.cpp src/cpp/megbrain_pubapi.cpp src/cpp/megbrain_serialize.cpp src/cpp/megbrain_wrap.cpp src/cpp/opr_defs.cpp src/cpp/opr_helper.cpp src/cpp/plugin.cpp src/cpp/python_helper.cpp)

include(UseSWIG)
set_property(SOURCE ${SWIG_SRC} PROPERTY CPLUSPLUS ON)

# cmake < 3.12 do not honor INCLUDE_DIRECTORIES property, just add include directory into SWIG_FLAGS
# Add -I${PROJECT_BINARY_DIR}/genfiles in order to include megbrain_build_config.h so that we don't need to pass cmake flags by -D.
set_property(SOURCE ${SWIG_SRC} PROPERTY SWIG_FLAGS -I${PROJECT_SOURCE_DIR}/src/serialization/include -I${PROJECT_BINARY_DIR}/genfiles)

set(SWIG_OUTFILE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_SWIG_OUTDIR ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal)
swig_add_library(mgb LANGUAGE python SOURCES ${SWIG_SRC} ${SRCS})

set(VERSION_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/version.ld)
add_custom_target(version_ld SOURCES ${VERSION_SCRIPT})

set_target_properties(mgb PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal)
if (APPLE)
    target_link_libraries(mgb megbrain megdnn)
else()
    target_link_libraries(mgb megbrain megdnn -Wl,--version-script=${VERSION_SCRIPT})
endif()
target_include_directories(mgb PRIVATE ${PYTHON_INCLUDE_DIRS} src/cpp ${CMAKE_CURRENT_BINARY_DIR} ${NUMPY_INCLUDE_DIR})
target_link_libraries(mgb ${PYTHON_LIBRARIES})

if (MGE_WITH_DISTRIBUTED)
    target_link_libraries(mgb megray)
endif()

add_dependencies(mgb mgb_opr_py version_ld)

add_custom_command(
    TARGET mgb POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/ACKNOWLEDGMENTS ${PROJECT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/megengine ${CMAKE_CURRENT_BINARY_DIR}/megengine
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_BINARY_DIR}/test
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requires.txt ${CMAKE_CURRENT_BINARY_DIR}/requires.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requires-style.txt ${CMAKE_CURRENT_BINARY_DIR}/requires-style.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/requires-test.txt ${CMAKE_CURRENT_BINARY_DIR}/requires-test.txt
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/megbrain_pubapi.h ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include/megbrain_pubapi.h
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/src/core/include ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/src/opr/include ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/src/serialization/include ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/src/plugin/include ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/dnn/include ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/genfiles/megbrain_build_config.h ${CMAKE_CURRENT_BINARY_DIR}/megengine/_internal/include/megbrain_build_config.h
)

