find_package(NumPy REQUIRED)

set(PACKAGE_NAME megengine)
set(PACKAGE_NAME
    ${PACKAGE_NAME}
    PARENT_SCOPE)
set(MODULE_NAME _imperative_rt)
set(MODULE_NAME
    ${MODULE_NAME}
    PARENT_SCOPE)
file(GLOB_RECURSE SRCS src/impl/*.cpp src/include/*.h python/src/*.cpp python/src/*.h)
set(SRCS ${SRCS} ${CPP_REDIS_SRCS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMGB_WITH_IMPERATIVE=1")

file(GLOB_RECURSE PYTHON_SRCS python/${PACKAGE_NAME}/*.py)
file(
  GLOB_RECURSE
  ALL_HEADERS
  src/cpp/megbrain_pubapi.h
  ${PROJECT_SOURCE_DIR}/src/core/include/*
  ${PROJECT_SOURCE_DIR}/src/opr/include/*
  ${PROJECT_SOURCE_DIR}/src/serialization/include/*
  ${PROJECT_SOURCE_DIR}/src/plugin/include/*
  ${PROJECT_SOURCE_DIR}/dnn/include/*)

set(MEGENGINE_DIR ${CMAKE_CURRENT_BINARY_DIR}/python/)

add_subdirectory(tablegen)

add_custom_target(_version_ld SOURCES ${MGE_VERSION_SCRIPT})

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11
                 ${PROJECT_BINARY_DIR}/third_party/pybind11)
pybind11_add_module(${MODULE_NAME} NO_EXTRAS ${SRCS})
if(APPLE)
  target_link_libraries(${MODULE_NAME} PRIVATE megengine_shared)
elseif(MSVC OR WIN32)
  target_link_libraries(${MODULE_NAME} PRIVATE megengine_shared)
  message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
  set_target_properties(${MODULE_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY
                                                  "${CMAKE_MSVC_RUNTIME_LIBRARY}")
else()
  # use to fix runtime crash when build both mgb(MGE_WITH_PYTHON_MODULE) and
  # imperative(MGE_BUILD_IMPERATIVE_RT)
  target_link_libraries(
    ${MODULE_NAME} PRIVATE megengine_shared -Wl,--version-script=${MGE_VERSION_SCRIPT})
endif()

if(ANDROID)
  target_link_libraries(${MODULE_NAME} PRIVATE ${PYTHON_LIBRARIES})
endif()

if(MGE_WITH_CUPTI)
  target_link_libraries(${MODULE_NAME} PRIVATE libcupti)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/range-v3
                 ${PROJECT_BINARY_DIR}/third_party/range-v3)
target_link_libraries(${MODULE_NAME} PRIVATE range-v3)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/Json
                 ${PROJECT_BINARY_DIR}/third_party/Json)
target_link_libraries(${MODULE_NAME} PRIVATE nlohmann_json::nlohmann_json)

target_include_directories(
  ${MODULE_NAME}
  PUBLIC src/include
  PRIVATE ${PROJECT_SOURCE_DIR}/third_party/boost_subset/boost ${PYTHON_INCLUDE_DIRS}
          ${NUMPY_INCLUDE_DIR} ${CPP_REDIS_INCLUDES})
target_link_libraries(${MODULE_NAME} PRIVATE mgb_opdef_inc)
target_compile_definitions(${MODULE_NAME} PRIVATE MODULE_NAME=${MODULE_NAME})
target_compile_options(${MODULE_NAME} PRIVATE -Wno-unused-parameter)
if(CXX_SUPPORT_WCLASS_MEMACCESS)
  target_compile_options(${MODULE_NAME} PRIVATE "-Wno-class-memaccess")
endif()
set_target_properties(
  ${MODULE_NAME}
  PROPERTIES SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
             LIBRARY_OUTPUT_DIRECTORY ${MEGENGINE_DIR}/${PACKAGE_NAME}/core)
if(APPLE
   OR MSVC
   OR WIN32)
  message(
    VERBOSE
    "overwriting SUFFIX at macos and windows before config by set_target_properties")
  pybind11_extension(${MODULE_NAME})
endif()
add_dependencies(${MODULE_NAME} mgb_opdef _version_ld)

if(MGE_WITH_TEST AND MGE_ENABLE_RTTI)
  add_subdirectory(test)
endif()

add_custom_command(
  TARGET ${MODULE_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/LICENSE
          ${PROJECT_SOURCE_DIR}/ACKNOWLEDGMENTS ${PROJECT_BINARY_DIR}
  COMMAND
    ${CMAKE_COMMAND} -E remove -f
    ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/$<TARGET_FILE_NAME:${MODULE_NAME}> # clean
                                                                                         # develop
  COMMAND ${CMAKE_COMMAND} -E remove -f
          ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/version.py # clean develop
  COMMAND ${CMAKE_COMMAND} -E remove -f
          ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/include # clean develop
  COMMAND ${CMAKE_COMMAND} -E remove -f
          ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/lib # clean develop
  COMMAND ${CMAKE_COMMAND} -E remove -f
          ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/${MODULE_NAME} # clean
                                                                           # develop
  COMMAND
    ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine
    ${CMAKE_CURRENT_BINARY_DIR}/python/megengine
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/test
          ${CMAKE_CURRENT_BINARY_DIR}/python/test
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/src/custom/include
          ${CMAKE_CURRENT_BINARY_DIR}/python/megengine/core/include
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py
          ${CMAKE_CURRENT_BINARY_DIR}/python/setup.py
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires.txt
          ${CMAKE_CURRENT_BINARY_DIR}/python/requires.txt
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires-style.txt
          ${CMAKE_CURRENT_BINARY_DIR}/python/requires-style.txt
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires-test.txt
          ${CMAKE_CURRENT_BINARY_DIR}/python/requires-test.txt)

if(DEFINED MGB_VER_MAJOR)
  set(IS_INTERNAL "--internal")
else()
  set(IS_INTERNAL "")
endif(DEFINED MGB_VER_MAJOR)

if(DEFINED MGE_EXTRA_NAME)
  set(RC_NAME "--rc=${MGE_EXTRA_NAME}")
else()
  set(RC_NAME "")
endif(DEFINED MGE_EXTRA_NAME)

add_custom_command(
  TARGET ${MODULE_NAME}
  POST_BUILD
  COMMAND
    "${PYTHON_EXECUTABLE}" ${CMAKE_CURRENT_SOURCE_DIR}/python/gen_version.py --output
    ${CMAKE_CURRENT_BINARY_DIR}/python/megengine/version.py --major ${MGE_VER_MAJOR}
    --minor ${MGE_VER_MINOR} --patch ${MGE_VER_PATCH} ${RC_NAME} ${IS_INTERNAL})
