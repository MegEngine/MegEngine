option(DNNL_BUILD_TESTS "" OFF)
option(DNNL_BUILD_EXAMPLES "" OFF)
set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "config dnnl to STATIC")
# we do not want to use OMP now, so config to CPU mode
# if set to OMP, some dnnl algo will be more fast
set(DNNL_CPU_RUNTIME DNNL_RUNTIME_SEQ CACHE STRING "config dnnl to DNNL_RUNTIME_SEQ")
if(MGE_BLAS STREQUAL "MKL")
    option(_DNNL_USE_MKL "" ON)
    set(MKLROOT ${MKL_ROOT_DIR} CACHE STRING "MKL ROOT FOR DNNL")
    if (WIN32)
        set(MKLLIB "mkl_core" CACHE STRING "MKLLIB NAME FOR DNNL")
    else()
        set(MKLLIB "libmkl_core.a" CACHE STRING "MKLLIB NAME FOR DNNL")
    endif()
    # workround for DNNL link failed, we do not want to modify
    # dnnl MKL.cmake of intel-mkl-dnn when include intel-mkl-dnn
    # via add_subdirectory api
    link_directories(${MKL_ROOT_DIR}/lib)
else()
    option(_DNNL_USE_MKL "" OFF)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-parameter -Wno-extra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter -Wno-extra")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/intel-mkl-dnn)
