include_directories("..")

file(GLOB_RECURSE SOURCES common/*.cpp naive/*.cpp dispatcher/*.cpp)
file(GLOB SOURCES_ *.cpp)
list(APPEND SOURCES ${SOURCES_})

if(NOT ${MGE_ARCH} STREQUAL "naive")
    file(GLOB_RECURSE SOURCES_ fallback/*.cpp)
    list(APPEND SOURCES ${SOURCES_})
    file(GLOB_RECURSE SOURCES_ cpu/*.cpp)
    list(APPEND SOURCES ${SOURCES_})
    if(${MGE_ARCH} STREQUAL "fallback")
        message(WARNING "build only with fallback")
    elseif(${MGE_ARCH} STREQUAL "x86_64" OR ${MGE_ARCH} STREQUAL "i386")
        file(GLOB_RECURSE SOURCES_ x86/*.cpp)
        list(APPEND SOURCES ${SOURCES_})
    endif()
endif()

if(MGE_WITH_CUDA)
    file(GLOB_RECURSE SOURCES_ cuda/*.cpp)
    list(APPEND SOURCES ${SOURCES_})

    file(GLOB_RECURSE CUSOURCES cuda/*.cu)
    list(APPEND SOURCES ${CUSOURCES})
endif()

if(MGE_WITH_MIDOUT_PROFILE)
    list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/third_party/midout/src/midout.cpp)
endif()

if(MGE_WITH_CAMBRICON)
    file(GLOB_RECURSE SOURCES_ cambricon/*.cpp)
    list(APPEND SOURCES ${SOURCES_})
endif()

if(MGE_WITH_ATLAS)
    file(GLOB_RECURSE SOURCES_ atlas/*.cpp)
    list(APPEND SOURCES ${SOURCES_})
endif()

if (MGE_WITH_ROCM)
    file (GLOB_RECURSE SOURCES_ rocm/*.cpp)
    list (APPEND SOURCES ${SOURCES_})
endif()


add_executable(megdnn_test ${SOURCES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
target_link_libraries(megdnn_test gtest)
target_link_libraries(megdnn_test megdnn ${MGE_BLAS_LIBS})

if (MGE_WITH_CUDA)
    target_link_libraries(megdnn_test cutlass)
    target_include_directories(megdnn_test PRIVATE ${CUDNN_INCLUDE_DIR})
endif()

target_include_directories(megdnn_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/midout/src
)

if(APPLE OR ANDROID)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
else()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

if(MGE_ENABLE_COVERAGE)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

if (MEG_WITH_ROCM)
    target_link_libraries (megdnn_test ${MGE_ROCM_LIBS})
endif ()

if(UNIX)
    if(APPLE OR ANDROID)
        target_link_libraries(megdnn_test dl)
    else()
        target_link_libraries(megdnn_test dl rt)
    endif()
endif()
