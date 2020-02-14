if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
    if(${MGE_BLAS} STREQUAL "MKL")
        add_definitions(-DMEGDNN_X86_WITH_MKL)
    elseif(${MGE_BLAS} STREQUAL "OpenBLAS")
        add_definitions(-DMEGDNN_X86_WITH_OPENBLAS)
    endif()
endif()

# Enable Naive
if(${MGE_ARCH} STREQUAL "naive")
    add_definitions(-DMEGDNN_NAIVE=1)
    message(WARNING "MEGDNN_NAIVE is enabled; MegDNN performance is degraded.")
else()
    add_definitions(-DMEGDNN_NAIVE=0)
endif()


if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
    add_definitions(-DMEGDNN_X86=1)
    if(${MGE_ARCH} STREQUAL "x86_64")
        add_definitions(-DMEGDNN_X86_64 -DMEGDNN_64_BIT)
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
        endif()
    else()
        add_definitions(-DMEGDNN_X86_32)
        if(NOT MSVC)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
        endif()
    endif()
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mfpmath=sse")
    endif()
endif()


set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MARCH}")

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
        /dev/null --write-cppjson ${OPR_PARAM_DEFS_OUT_DIR}/megdnn/opr_param_json.h
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


install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/megdnn DESTINATION include FILES_MATCHING PATTERN "*.h")

add_custom_target(_opr_param_defs DEPENDS ${OPR_PARAM_DEFS_OUTS})
add_library(opr_param_defs INTERFACE)
target_include_directories(opr_param_defs INTERFACE ${OPR_PARAM_DEFS_INC})
add_dependencies(opr_param_defs _opr_param_defs)





if(MGE_WITH_TEST)
    # use multi threads
    add_definitions (-DMEGDNN_ENABLE_MULTI_THREADS=1)
    add_subdirectory(test)
endif()

add_subdirectory(src)
