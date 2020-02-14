file (GLOB_RECURSE SOURCES src/*.cpp)

add_library (cuda-stub SHARED ${SOURCES})
set_target_properties(cuda-stub PROPERTIES OUTPUT_NAME cuda)
target_compile_definitions(cuda-stub PRIVATE __CUDA_API_VERSION_INTERNAL)
target_link_libraries(cuda-stub PRIVATE dl -Wl,--no-undefined)
