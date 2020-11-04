if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP QUIET)
if (HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

string(REPLACE "." ";" HIP_VERSION_LIST ${HIP_VERSION})
list(GET HIP_VERSION_LIST 0 HIP_VERSION_MAJOR)
list(GET HIP_VERSION_LIST 1 HIP_VERSION_MINOR)
if (NOT ${HIP_VERSION_MAJOR} STREQUAL "3")
    message(FATAL_ERROR "ROCM version needed 3.x, Please update ROCM.")
else()
    if (${HIP_VERSION_MINOR} LESS "7")
        message(WARNING "ROCM version 3.x which x(got ${HIP_VERSION_MINOR}) greater equal 7 is prefered.")
    endif()
endif()

set(MGE_ROCM_LIBS OpenCL amdhip64 MIOpen rocblas rocrand)

set(HIP_INCLUDE_DIR ${HIP_ROOT_DIR}/../include)
set(HIP_LIBRARY_DIR ${HIP_ROOT_DIR}/../lib)

#miopen
get_filename_component(__found_miopen_library ${HIP_ROOT_DIR}/../miopen/lib REALPATH)
find_path(MIOPEN_LIBRARY_DIR
    NAMES libMIOpen.so
    HINTS ${PC_MIOPEN_INCLUDE_DIRS} ${MIOPEN_ROOT_DIR} ${ROCM_TOOLKIT_INCLUDE} ${__found_miopen_library}
    PATH_SUFFIXES lib
    DOC "Path to MIOPEN library directory." )

if(MIOPEN_LIBRARY_DIR STREQUAL "MIOPEN_LIBRARY_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find MIOPEN Library")
endif()

get_filename_component(__found_miopen_include ${HIP_ROOT_DIR}/../miopen/include REALPATH)
find_path(MIOPEN_INCLUDE_DIR
    NAMES miopen
    HINTS ${PC_MIOPEN_INCLUDE_DIRS} ${MIOPEN_ROOT_DIR} ${ROCM_TOOLKIT_INCLUDE} ${__found_miopen_include}
    PATH_SUFFIXES include
    DOC "Path to MIOPEN include directory." )

if(MIOPEN_INCLUDE_DIR STREQUAL "MIOPEN_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find MIOEPN INCLUDE")
endif()

#rocblas
get_filename_component(__found_rocblas_library ${HIP_ROOT_DIR}/../rocblas/lib REALPATH)
find_path(ROCBLAS_LIBRARY_DIR
    NAMES librocblas.so
    HINTS ${PC_ROCBLAS_INCLUDE_DIRS} ${ROCBLAS_ROOT_DIR} ${ROCM_TOOLKIT_INCLUDE} ${__found_rocblas_library}
    PATH_SUFFIXES lib
    DOC "Path to ROCBLAS library directory." )

if(ROCBLAS_LIBRARY_DIR STREQUAL "ROCBLAS_LIBRARY_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find ROCBLAS Library")
endif()

get_filename_component(__found_rocblas_include ${HIP_ROOT_DIR}/../rocblas/include REALPATH)
find_path(ROCBLAS_INCLUDE_DIR
    NAMES rocblas.h
    HINTS ${PC_ROCBLAS_INCLUDE_DIRS} ${ROCBLAS_ROOT_DIR} ${ROCM_TOOLKIT_INCLUDE} ${__found_rocblas_include}
    PATH_SUFFIXES include
    DOC "Path to ROCBLAS include directory." )

if(ROCBLAS_INCLUDE_DIR STREQUAL "ROCBLAS_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find ROCBLAS INCLUDE")
endif()

#rocrand
get_filename_component(__found_rocrand_library ${HIP_ROOT_DIR}/../rocrand/lib REALPATH)
find_path(ROCRAND_LIBRARY_DIR
    NAMES librocrand.so
    HINTS ${PC_ROCRAND_INCLUDE_DIRS} ${ROCRAND_ROOT_DIR} ${ROCM_TOOLKIT_INCLUDE} ${__found_rocrand_library}
    PATH_SUFFIXES lib
    DOC "Path to ROCRAND library directory." )

if(ROCRAND_LIBRARY_DIR STREQUAL "ROCRAND_LIBRARY_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find ROCRAND Library")
endif()

get_filename_component(__found_rocrand_include ${HIP_ROOT_DIR}/../rocrand/include REALPATH)
find_path(ROCRAND_INCLUDE_DIR
    NAMES rocrand.h
    HINTS ${PC_ROCRAND_INCLUDE_DIRS} ${ROCRAND_ROOT_DIR} ${ROCM_TOOLKIT_INCLUDE} ${__found_rocrand_include}
    PATH_SUFFIXES include
    DOC "Path to ROCRAND include directory." )

if(ROCRAND_INCLUDE_DIR STREQUAL "ROCRAND_INCLUDE_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find ROCRAND INCLUDE")
endif()


