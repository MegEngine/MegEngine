include_directories("${CMAKE_CURRENT_BINARY_DIR}/gflags/include")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/gflags
                 ${CMAKE_CURRENT_BINARY_DIR}/gflags)
