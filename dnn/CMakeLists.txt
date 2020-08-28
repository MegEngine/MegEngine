list(APPEND OPR_PARAM_DEFS_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/opr_param_defs.py)
set(OPR_PARAM_DEFS_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_param_defs.py)

set(OPR_PARAM_DEFS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/include/)
file(MAKE_DIRECTORY ${OPR_PARAM_DEFS_OUT_DIR}/megdnn)
add_custom_command(
    OUTPUT
        ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_defs.h
        ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_json.h
    COMMAND ${PYTHON_EXECUTABLE} ${OPR_PARAM_DEFS_SCRIPT} ${OPR_PARAM_DEFS_SRCS}
        ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_defs.h
    COMMAND ${PYTHON_EXECUTABLE} ${OPR_PARAM_DEFS_SCRIPT} ${OPR_PARAM_DEFS_SRCS}
       tmp_unuse.log --write-cppjson ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_json.h
    DEPENDS ${OPR_PARAM_DEFS_SRCS} ${OPR_PARAM_DEFS_SCRIPT}
    VERBATIM
)

list(APPEND OPR_PARAM_DEFS_OUTS
    ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_defs.h
    ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_json.h
)
list(APPEND OPR_PARAM_DEFS_INC ${OPR_PARAM_DEFS_OUT_DIR})

set(OPR_PARAM_DEFS_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${OPR_PARAM_DEFS_OUT_DIR}/src/common)
add_custom_command(
    OUTPUT
        ${OPR_PARAM_DEFS_OUT_DIR}/src/common/opr_param_defs_enumv.cuh
    COMMAND ${PYTHON_EXECUTABLE} ${OPR_PARAM_DEFS_SCRIPT}
        --enumv ${OPR_PARAM_DEFS_SRCS}
        ${OPR_PARAM_DEFS_OUT_DIR}/src/common/opr_param_defs_enumv.cuh
    DEPENDS ${OPR_PARAM_DEFS_SRCS} ${OPR_PARAM_DEFS_SCRIPT}
    VERBATIM
)

list(APPEND OPR_PARAM_DEFS_OUTS
    ${OPR_PARAM_DEFS_OUT_DIR}/src/common/opr_param_defs_enumv.cuh
)
list(APPEND OPR_PARAM_DEFS_INC ${OPR_PARAM_DEFS_OUT_DIR})


install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/megdnn DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")

add_custom_target(_opr_param_defs DEPENDS ${OPR_PARAM_DEFS_OUTS})
add_library(opr_param_defs INTERFACE)
target_include_directories(opr_param_defs
    INTERFACE
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
foreach (INCPATH IN LISTS OPR_PARAM_DEFS_INC)
    target_include_directories(opr_param_defs
        INTERFACE $<BUILD_INTERFACE:${INCPATH}>
    )
endforeach()

add_dependencies(opr_param_defs _opr_param_defs)
install(TARGETS opr_param_defs EXPORT ${MGE_EXPORT_TARGETS})


if(MGE_WITH_CUDA)
    add_library(cutlass INTERFACE)
    target_include_directories(cutlass 
        INTERFACE 
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/cutlass/include>)
endif()

if(MGE_WITH_TEST)
    add_subdirectory(test)
endif()

add_subdirectory(src)

# install(TARGETS megdnn opr_param_defs libmkl EXPORT ${MGE_EXPORT_TARGETS})
