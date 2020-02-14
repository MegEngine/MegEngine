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
    elseif(${MGE_ARCH} STREQUAL "x86")
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


add_executable(megdnn_test ${SOURCES})
target_link_libraries(megdnn_test gtest)
target_link_libraries(megdnn_test megdnn)

if(UNIX)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
endif()

if(MGE_ENABLE_COVERAGE)
    set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

if(UNIX)
    target_link_libraries(megdnn_test dl rt)
endif()

install(TARGETS megdnn_test RUNTIME DESTINATION test)
