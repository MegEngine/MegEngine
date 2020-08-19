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


if(MGE_WITH_CUDA)
    file(GLOB_RECURSE SOURCES_ cuda/*.cpp)
    list(APPEND SOURCES ${SOURCES_})

    file(GLOB_RECURSE CUSOURCES cuda/*.cu)
    list(APPEND SOURCES ${CUSOURCES})
endif()



add_definitions(${LIBMEGDNN_DEF})
add_library(megdnn EXCLUDE_FROM_ALL OBJECT ${SOURCES})

target_link_libraries(megdnn PUBLIC opr_param_defs)
if(MGE_WITH_CUDA)
    target_link_libraries(megdnn PUBLIC cutlass)
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

if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(megdnn PRIVATE Threads::Threads)
endif()

install(TARGETS megdnn EXPORT ${MGE_EXPORT_TARGETS})
