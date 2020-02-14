
set(LIBMEGDNN_DEF)
file(GLOB_RECURSE SOURCES common/*.cpp naive/*.cpp)

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
    endif()
endif()

if(MGE_WITH_CUDA)
    file(GLOB_RECURSE SOURCES_ cuda/*.cpp)
    list(APPEND SOURCES ${SOURCES_})

    file(GLOB_RECURSE CUSOURCES cuda/*.cu)
    list(APPEND SOURCES ${CUSOURCES})
    list(APPEND LIBMEGDNN_DEF -DMEGDNN_WITH_CUDA=1)
endif()



add_definitions(${LIBMEGDNN_DEF})
add_library(megdnn EXCLUDE_FROM_ALL STATIC ${SOURCES})

target_link_libraries(megdnn opr_param_defs)
target_include_directories(megdnn PUBLIC ${PROJECT_SOURCE_DIR}/dnn/include)
target_include_directories(megdnn PRIVATE ${PROJECT_SOURCE_DIR}/dnn ${PROJECT_SOURCE_DIR}/third_party/midout/src)

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
    target_link_libraries(megdnn libmkl_dnn)
endif()
target_link_libraries(megdnn ${MGE_CUDA_LIBS})
target_link_libraries(megdnn ${MGE_BLAS_LIBS})
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(megdnn Threads::Threads)
endif()

