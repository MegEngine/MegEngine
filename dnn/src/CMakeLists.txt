set(LIBMEGDNN_DEF)
file(GLOB_RECURSE SOURCES common/*.cpp naive/*.cpp)

# Build configure
list(APPEND SOURCES ${PROJECT_BINARY_DIR}/genfiles/megbrain_build_config.h)

if(NOT ${MGE_ARCH} STREQUAL "naive")
    file(GLOB_RECURSE SOURCES_ fallback/*.cpp)
    list(APPEND SOURCES ${SOURCES_})
    if(${MGE_ARCH} STREQUAL "fallback")
        message(WARNING "build only with fallback")
    elseif(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
        file(GLOB_RECURSE SOURCES_ x86/*.cpp)
        list(APPEND SOURCES ${SOURCES_})
        if(NOT MSVC)
            file(GLOB_RECURSE SOURCES_ x86/*.S)
            set_source_files_properties(${SOURCES_} PROPERTIES LANGUAGE C)
            list(APPEND SOURCES ${SOURCES_})
        endif()
    elseif(${MGE_ARCH} STREQUAL "armv7")
        file(GLOB_RECURSE SOURCES_ armv7/*.cpp)
        list(APPEND SOURCES ${SOURCES_})
        file(GLOB_RECURSE SOURCES_ arm_common/*.cpp)
        list(APPEND SOURCES ${SOURCES_})
        file(GLOB_RECURSE SOURCES_ armv7/*.S)
        set_source_files_properties(${SOURCES_} PROPERTIES LANGUAGE C)
        list(APPEND SOURCES ${SOURCES_})
    elseif(${MGE_ARCH} STREQUAL "aarch64")
        file(GLOB_RECURSE SOURCES_ aarch64/*.cpp)
        list(APPEND SOURCES ${SOURCES_})
        file(GLOB_RECURSE SOURCES_ arm_common/*.cpp)
        list(APPEND SOURCES ${SOURCES_})
        file(GLOB_RECURSE SOURCES_ aarch64/*.S)
        set_source_files_properties(${SOURCES_} PROPERTIES LANGUAGE C)
        list(APPEND SOURCES ${SOURCES_})
    endif()
endif()

if(MGE_WITH_MIDOUT_PROFILE)
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/third_party/midout/src/midout.cpp)
endif()

###############################################################################
# HIP_COMPILE
###############################################################################
macro (HIP_COMPILE _hip_target _hip_objs)
    # Separate the sources from the options
    HIP_GET_SOURCES_AND_OPTIONS(_sources
        _cmake_options
        _hipcc_options
        _hcc_options
        _nvcc_options
        ${ARGN})
    HIP_PREPARE_TARGET_COMMANDS(${_hip_target}
        OBJ _generated_files _source_files ${_sources} ${_cmake_options}
        HIPCC_OPTIONS ${_hipcc_options}
        HCC_OPTIONS ${_hcc_options}
        NVCC_OPTIONS ${_nvcc_options})
    if(_source_files)
        list(REMOVE_ITEM _sources ${_source_files})
    endif()

    add_custom_target(${_hip_target})

    # set return value
    set (${_hip_objs} ${_generated_files})
endmacro()

if (MGE_WITH_ROCM)
    file (GLOB_RECURSE SOURCES_ rocm/*.cpp)
    list (APPEND SOURCES ${SOURCES_})

    # FIXME rocm may lost the first hip file, so currently we just create an
    # empty file to bypass this error.
    file(GLOB start.cpp.hip "" )
    list(APPEND HIP_SOURCES start.cpp.hip)

    file (GLOB_RECURSE HIPSOURCES rocm/*.cpp.hip)
    set(HIP_TARGET_NAME hip_kernel)
    set(_HIPCC_OPTIONS "-fPIC")
    set(_HCC_OPTIONS "-fPIC")
    set(_NVCC_OPTIONS "-fPIC")

    list(APPEND HIP_SOURCES ${HIPSOURCES})
    set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    HIP_INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/dnn
        ${PROJECT_SOURCE_DIR}/dnn/include
        ${PROJECT_BINARY_DIR}/dnn
        ${PROJECT_BINARY_DIR}/genfiles
        ${PROJECT_BINARY_DIR}/dnn/include
        ${HIP_INCLUDE_DIR}
        ${MIOPEN_INCLUDE_DIR}
        ${ROCBLAS_INCLUDE_DIR}
        ${ROCRAND_INCLUDE_DIR})
    hip_compile(
       ${HIP_TARGET_NAME} HIPOBJS ${HIP_SOURCES}
       HIPCC_OPTIONS ${_HIPCC_OPTIONS}
       HCC_OPTIONS ${_HCC_OPTIONS}
       NVCC_OPTIONS ${_NVCC_OPTIONS})
    list (APPEND SOURCES ${HIPOBJS})
endif ()

if(MGE_WITH_CUDA)
    file(GLOB_RECURSE SOURCES_ cuda/*.cpp)
    list(APPEND SOURCES ${SOURCES_})

    file(GLOB_RECURSE CUSOURCES cuda/*.cu)
    list(APPEND SOURCES ${CUSOURCES})
endif()

if(MGE_WITH_CAMBRICON)
    file(GLOB_RECURSE SOURCES_ cambricon/*.cpp)
    list(APPEND SOURCES ${SOURCES_})

    file(GLOB_RECURSE BANG_SOURCES cambricon/*.mlu)
    list(APPEND MEGDNN_INCLUDES "${PROJECT_SOURCE_DIR}/dnn/include")
    list(APPEND MEGDNN_INCLUDES "${PROJECT_SOURCE_DIR}/dnn")
    list(APPEND MEGDNN_INCLUDES "${PROJECT_BINARY_DIR}/genfiles")
    bang_compile(BANG_OBJS "${BANG_SOURCES}" "${MEGDNN_INCLUDES}")
    list(APPEND SOURCES ${BANG_OBJS})

endif()

if(MGE_WITH_ATLAS)
    file(GLOB_RECURSE SOURCES_ atlas/*.cpp)
    list(APPEND SOURCES ${SOURCES_})
    list(APPEND LIBMEGDNN_DEF -DMEGDNN_WITH_ATLAS=1)
endif()


add_definitions(${LIBMEGDNN_DEF})
add_library(megdnn EXCLUDE_FROM_ALL OBJECT ${SOURCES})

target_link_libraries(megdnn PUBLIC opr_param_defs)
if(MGE_WITH_CUDA)
    target_link_libraries(megdnn PRIVATE $<BUILD_INTERFACE:cutlass>)
endif()

if(MGE_WITH_ROCM)
        target_include_directories(megdnn PUBLIC
        ${HIP_INCLUDE_DIR}
        ${MIOPEN_INCLUDE_DIR}
        ${ROCBLAS_INCLUDE_DIR}
        ${ROCRAND_INCLUDE_DIR})
        target_link_directories(megdnn PUBLIC
        ${HIP_LIBRARY_DIR}
        ${MIOPEN_LIBRARY_DIR}
        ${ROCBLAS_LIBRARY_DIR}
        ${ROCRAND_LIBRARY_DIR})
endif()

if(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386" OR ${MGE_ARCH} STREQUAL "armv7" OR ${MGE_ARCH} STREQUAL "aarch64")
    if(MGE_ENABLE_CPUINFO)
        target_link_libraries(megdnn PRIVATE $<BUILD_INTERFACE:cpuinfo>)
    endif()
endif()

target_include_directories(megdnn
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/genfiles>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/dnn/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/dnn
        ${PROJECT_SOURCE_DIR}/third_party/midout/src
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/dnn/include DESTINATION . FILES_MATCHING PATTERN "*.h*")

if(CXX_SUPPORT_WCLASS_MEMACCESS)
    if(MGE_WITH_CUDA)
        target_compile_options(megdnn PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-class-memaccess>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-class-memaccess>")
    else()
        target_compile_options(megdnn PRIVATE "-Wno-class-memaccess")
    endif()
endif()
target_compile_definitions(megdnn INTERFACE ${LIBMEGDNN_DEF})

if(MGE_WITH_MKLDNN AND ${MGE_ARCH} STREQUAL "x86_64")
    if (BUILD_SHARED_LIBS)
        target_link_libraries(megdnn PRIVATE $<BUILD_INTERFACE:dnnl>)
    else()
        target_link_libraries(megdnn PRIVATE dnnl)
    endif()
endif()
target_link_libraries(megdnn PRIVATE ${MGE_CUDA_LIBS})
if (BUILD_SHARED_LIBS)
    target_link_libraries(megdnn PRIVATE $<BUILD_INTERFACE:${MGE_BLAS_LIBS}>)
else()
    target_link_libraries(megdnn PRIVATE ${MGE_BLAS_LIBS})
endif()

if (MGE_WITH_ROCM)
    target_link_libraries(megdnn PRIVATE ${HIPOBJS} ${MGE_ROCM_LIBS})
endif ()

if(MGE_WITH_ATLAS)
    if (BUILD_SHARED_LIBS)
        target_link_libraries(megdnn PRIVATE $<BUILD_INTERFACE:${MGE_ATLAS_LIBS}>)
    else()
        target_link_libraries(megdnn PRIVATE ${MGE_ATLAS_LIBS})
    endif()

endif()

if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(megdnn PRIVATE Threads::Threads)
endif()

if(MGE_WITH_CAMBRICON)
    target_link_libraries(megdnn PRIVATE ${BANG_OBJS} ${MGE_CAMBRICON_LIBS})
endif()

install(TARGETS megdnn EXPORT ${MGE_EXPORT_TARGETS})
