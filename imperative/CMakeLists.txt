find_package(NumPy REQUIRED)

set(PACKAGE_NAME megengine)
set(PACKAGE_NAME ${PACKAGE_NAME} PARENT_SCOPE)
set(MODULE_NAME _imperative_rt)
set(MODULE_NAME ${MODULE_NAME} PARENT_SCOPE)
file(GLOB_RECURSE SRCS src/impl/*.cpp src/include/*.h python/src/*.cpp python/src/*.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMGB_WITH_IMPERATIVE=1")

file(GLOB_RECURSE OPR_DECL_SRCS "${PROJECT_SOURCE_DIR}/src/**/*.oprdecl")
file(GLOB_RECURSE PYTHON_SRCS python/${PACKAGE_NAME}/*.py)
list(REMOVE_ITEM PYTHON_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/ops/_internal/generated_ops.py ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/ops/_internal/param_defs.py)
file(GLOB_RECURSE ALL_HEADERS src/cpp/megbrain_pubapi.h
    ${PROJECT_SOURCE_DIR}/src/core/include/*
    ${PROJECT_SOURCE_DIR}/src/opr/include/*
    ${PROJECT_SOURCE_DIR}/src/serialization/include/*
    ${PROJECT_SOURCE_DIR}/src/plugin/include/*
    ${PROJECT_SOURCE_DIR}/dnn/include/*)

set(MEGENGINE_DIR ${CMAKE_CURRENT_BINARY_DIR}/python/)
set(GEN_OPS_DIR ${MEGENGINE_DIR}/${PACKAGE_NAME}/core/ops/_internal)
file(MAKE_DIRECTORY ${GEN_OPS_DIR})
set(GEN_OPS_FILE ${GEN_OPS_DIR}/generated_ops.py)
set(GEN_OP_PARAMS_FILE ${MEGENGINE_DIR}/${PACKAGE_NAME}/core/ops/_internal/param_defs.py)
set(GEN_OP_PARAMS_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/python/tools/ops.tpl.py)

##################### generate python opr_param_defs.py   ##############

file(COPY ${PROJECT_SOURCE_DIR}/dnn/scripts/opr_param_defs.py DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(READ ${PROJECT_SOURCE_DIR}/tools/param_defs/mgb_opr_param_defs.py CONTENTS)
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/opr_param_defs.py ${CONTENTS})

add_custom_command(
    OUTPUT ${GEN_OPS_FILE}
    COMMAND ${CMAKE_COMMAND} -E touch ${MEGENGINE_DIR}/${PACKAGE_NAME}/core/${MODULE_NAME}.so ${GEN_OPS_FILE} ${GEN_OP_PARAMS_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/${PACKAGE_NAME} ${MEGENGINE_DIR}/${PACKAGE_NAME}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${MEGENGINE_DIR}/${PACKAGE_NAME}/core/${MODULE_NAME}.so ${GEN_OPS_FILE} ${GEN_OP_PARAMS_FILE}
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/tools/gen_ops.py ${OPR_DECL_SRCS} -o ${GEN_OPS_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/test ${MEGENGINE_DIR}/${PACKAGE_NAME}/test
    COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/dnn/scripts/gen_param_defs.py -t py --imperative ${CMAKE_CURRENT_BINARY_DIR}/opr_param_defs.py ${GEN_OP_PARAMS_FILE}
    DEPENDS ${OPR_DECL_SRCS} ${PYTHON_SRCS} ${ALL_HEADERS} ${GEN_OP_PARAMS_TEMPLATE}
    VERBATIM
)

add_custom_target(gen_opr_py DEPENDS ${GEN_OPS_FILE})

##################### end of opdef generation #########################

set(VERSION_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/src/version.ld)
add_custom_target(_version_ld SOURCES ${VERSION_SCRIPT})

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 ${PROJECT_BINARY_DIR}/third_party/pybind11)
pybind11_add_module(${MODULE_NAME} NO_EXTRAS ${SRCS})
if (APPLE OR MSVC OR WIN32)
    target_link_libraries(${MODULE_NAME} PRIVATE megbrain megdnn)
else()
    target_link_libraries(${MODULE_NAME} PRIVATE megbrain megdnn -Wl,--version-script=${VERSION_SCRIPT})
endif()
if (MGE_WITH_DISTRIBUTED)
    message("Imperative configured to link megray")
    target_link_libraries(${MODULE_NAME} PRIVATE megray)
endif()
target_include_directories(${MODULE_NAME} PUBLIC src/include PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_compile_definitions(${MODULE_NAME} PRIVATE MODULE_NAME=${MODULE_NAME})
target_compile_options(${MODULE_NAME} PRIVATE -Wno-unused-parameter)
if(CXX_SUPPORT_WCLASS_MEMACCESS)
    target_compile_options(${MODULE_NAME} PRIVATE "-Wno-class-memaccess")
endif()
set_target_properties(${MODULE_NAME} PROPERTIES
    SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
    LIBRARY_OUTPUT_DIRECTORY ${MEGENGINE_DIR}/${PACKAGE_NAME}/core
)
if (APPLE OR MSVC OR WIN32)
    message("-- overwriting SUFFIX at macos and windows before config by set_target_properties")
    pybind11_extension(${MODULE_NAME})
endif()
add_dependencies(${MODULE_NAME} gen_opr_py _version_ld)

if(MGE_WITH_TEST AND MGE_ENABLE_RTTI)
    add_subdirectory(test)
endif()

add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/ACKNOWLEDGMENTS ${PROJECT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine ${CMAKE_CURRENT_BINARY_DIR}/python/megengine
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/test ${CMAKE_CURRENT_BINARY_DIR}/python/test
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py ${CMAKE_CURRENT_BINARY_DIR}/python/setup.py
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires.txt ${CMAKE_CURRENT_BINARY_DIR}/python/requires.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires-style.txt ${CMAKE_CURRENT_BINARY_DIR}/python/requires-style.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires-test.txt ${CMAKE_CURRENT_BINARY_DIR}/python/requires-test.txt
)

