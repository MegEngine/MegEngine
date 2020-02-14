if (MGE_USE_SYSTEM_LIB)
    find_package(FlatBuffers REQUIRED)
    return()
endif()

option(FLATBUFFERS_BUILD_TESTS "" OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/flatbuffers
                 ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers
                 EXCLUDE_FROM_ALL)