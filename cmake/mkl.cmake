find_path(MKL_ROOT_DIR
    include/mkl_cblas.h
    PATHS
    ${PROJECT_SOURCE_DIR}/third_party/mkl/${MGE_ARCH}
    ${PROJECT_SOURCE_DIR}/third_party/mkl/${MGE_ARCH}/Library
    ${PROJECT_SOURCE_DIR}/third_party/mkl/x86_32/Library
    $ENV{MKLDIR}
    /opt/intel/mkl/*/
    /opt/intel/cmkl/*/
    /Library/Frameworks/Intel_MKL.framework/Versions/Current/lib/universal
)

if(${MKL_ROOT_DIR} STREQUAL "MKL_ROOT_DIR-NOTFOUND")
    message(FATAL_ERROR "Can not find MKL")
endif()
message(STATUS "Build with MKL in ${MKL_ROOT_DIR}")

find_path(MKL_INCLUDE_DIR
    mkl_cblas.h
    PATHS
    ${MKL_ROOT_DIR}/include
    ${INCLUDE_INSTALL_DIR}
)

option(MGE_MKL_USE_STATIC "Build MegEngine with static MKL" ON)
if(MGE_MKL_USE_STATIC)
    find_library(MKL_CORE_LIBRARY
        NAMES libmkl_core.a mkl_core.lib
        PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)

    find_library(MKL_SEQUENTIAL_LIBRARY
        NAMES libmkl_sequential.a mkl_sequential.lib
        PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)

    if(${MGE_ARCH} STREQUAL "x86_64")
        find_library(MKL_IPL_LIBRARY
            NAMES libmkl_intel_ilp64.a mkl_intel_ilp64.lib
            PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)
    elseif(${MGE_ARCH} STREQUAL "i386")
        find_library(MKL_IPL_LIBRARY
            NAMES libmkl_intel_32.a mkl_intel_32.lib mkl_intel_c.lib
            PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)
    endif()

    add_library(libmkl INTERFACE IMPORTED)
    if(UNIX AND NOT APPLE)
        target_link_libraries(libmkl INTERFACE -Wl,--start-group ${MKL_CORE_LIBRARY} ${MKL_SEQUENTIAL_LIBRARY} ${MKL_IPL_LIBRARY} -Wl,--end-group)
    else()
        target_link_libraries(libmkl INTERFACE ${MKL_CORE_LIBRARY} ${MKL_SEQUENTIAL_LIBRARY} ${MKL_IPL_LIBRARY})
    endif()
    target_include_directories(libmkl INTERFACE ${MKL_INCLUDE_DIR})
else()
    find_library(MKL_CORE_LIBRARY
        NAMES libmkl_core.so libmkl_core.dylib
        PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)

    find_library(MKL_SEQUENTIAL_LIBRARY
        NAMES libmkl_sequential.so libmkl_sequential.dylib
        PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)

    if(${MGE_ARCH} STREQUAL "x86_64")
        find_library(MKL_IPL_LIBRARY
            NAMES libmkl_intel_ilp64.so libmkl_intel_ilp64.dylib
            PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)
    elseif(${MGE_ARCH} STREQUAL "x86_32")
        find_library(MKL_IPL_LIBRARY
            NAMES libmkl_intel_32.so libmkl_intel_32.dylib
            PATHS ${MKL_ROOT_DIR}/lib/${MKL_ARCH_DIR} ${MKL_ROOT_DIR}/lib/)
    endif()
    target_link_libraries(libmkl INTERFACE ${MKL_CORE_LIBRARY} ${MKL_SEQUENTIAL_LIBRARY} ${MKL_IPL_LIBRARY})
    target_include_directories(libmkl INTERFACE ${MKL_INCLUDE_DIR})
endif()

if(${MGE_ARCH} STREQUAL "x86_64")
    target_compile_definitions(libmkl INTERFACE -DMKL_ILP64)
endif()
