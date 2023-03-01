file(GLOB_RECURSE CUDA_STUB src/libcuda.cpp)
file(GLOB_RECURSE NVRTC_STUB src/libnvrtc.cpp)

if(MGE_WITH_CUDA_STUB)
  list(APPEND STUB_SRC ${CUDA_STUB})
endif()

if(MGE_WITH_NVRTC_STUB)
  list(APPEND STUB_SRC ${NVRTC_STUB})
endif()

add_library(cuda-stub OBJECT ${STUB_SRC})

set_target_properties(cuda-stub PROPERTIES OUTPUT_NAME cuda_stub)
target_compile_definitions(cuda-stub PRIVATE __CUDA_API_VERSION_INTERNAL)
if(MSVC OR WIN32)
  target_link_libraries(cuda-stub PRIVATE -Wl,--no-undefined)
else()
  target_link_libraries(cuda-stub PRIVATE dl -Wl,--no-undefined)
endif()
target_include_directories(cuda-stub
                           PRIVATE $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/genfiles>)
