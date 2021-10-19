option(LITE_BUILD_WITH_MGE "Build lite with MegEngine." ON)

# config lite_build_config.h.in
set(LITE_WITH_OPENCL ${MGE_WITH_OPENCL})
set(LITE_WITH_CUDA ${MGE_WITH_CUDA})
set(LITE_ENABLE_LOGGING ${MGE_ENABLE_LOGGING})
set(LITE_ENABLE_EXCEPTION ${MGE_ENABLE_EXCEPTIONS})
set(LITE_ASSERT_LOC ${MGB_ASSERT_LOC})

if(NOT MGB_WITH_FLATBUFFERS)
    include(../cmake/flatbuffers.cmake)
endif()

file(GLOB_RECURSE SRC_FBS src/**/*.fbs)
build_flatbuffers(
    "${SRC_FBS}"
    ""
    lite_fbs_generate
    ""
    "${CMAKE_CURRENT_BINARY_DIR}"
    ""
    ""
    )

file(GLOB_RECURSE SOURCES_LITE src/*.cpp src/*.cc lite-c/*.cpp)

if(MGE_WITH_MINIMUM_SIZE)
    set(LITE_ENABLE_LOGGING OFF)
    set(LITE_ENABLE_EXCEPTION OFF)
endif()

# Write out lite_build_config.h
# It defines macros needed by lite
configure_file(src/lite_build_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/genfiles/lite_build_config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/genfiles/lite_build_config.h DESTINATION ${CMAKE_INSTALL_PREFIX}/lite/include)

# begin config lite
add_library(lite_static STATIC ${SOURCES_LITE})
add_dependencies(lite_static lite_fbs_generate)
include_directories($<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/genfiles>)

if(LITE_BUILD_WITH_MGE)
    target_link_libraries(lite_static PRIVATE megbrain megdnn ${MGE_CUDA_LIBS})
    add_compile_definitions(LITE_BUILD_WITH_MGE=1)
    if(WIN32 OR MSVC)
        target_compile_definitions(lite_static PRIVATE MGE_DLL_IMPORT_DATA)
    endif()
    message(STATUS "build lite with MegEngine.")
else()
    target_link_libraries(lite_static PUBLIC flatbuffers)
endif()

include_directories(
    PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/lite/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lite/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lite/include/lite>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lite/lite-c/include>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lite/src>
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/Json/include>
    )
# end config lite

# define a shared lib
add_library(lite_shared SHARED $<TARGET_OBJECTS:lite_static>)
if(LITE_BUILD_WITH_MGE)
    target_link_libraries(lite_shared PRIVATE megbrain megdnn ${MGE_CUDA_LIBS})
endif()
if(ANDROID)
    link_libraries(log)
    target_link_libraries(lite_static PRIVATE log)
    target_link_libraries(lite_shared PRIVATE log)
endif()

# define a shared lib for whl
add_library(lite_shared_whl SHARED $<TARGET_OBJECTS:lite_static>)
if(LITE_BUILD_WITH_MGE)
    if (IOS)
        target_link_libraries(lite_shared_whl PRIVATE megbrain megdnn ${MGE_CUDA_LIBS})
    else()
        target_link_libraries(lite_shared_whl PRIVATE megengine_shared)
    endif()
    if(WIN32 OR MSVC)
        target_compile_definitions(lite_shared_whl PRIVATE MGE_DLL_IMPORT_DATA)
    endif()
endif()
if(ANDROID)
    target_link_libraries(lite_shared_whl PRIVATE log)
endif()

# add lite_static_all_in_one same name build by BUILD
# please do not change flatbuffers/cpuinfo/clog/lite_static order, if change!, cmake
# can not gen flatbuffers/cpuinfo/clog OBJs to lite_static_all_in_one, this may cmake issue
# NOTICE: this target always use to separate build with lite, if build lite via include
# MegEngine/megbrain ROOT_DIR/CMakeLists.txt, just depends lite_static or lite_shared
#TODO: need refine lite_static_all_in_one depend objects, but now cmake do not support
# define a add_library which OBJECTS args is a set or list or string
if (MGE_ENABLE_CPUINFO AND MGE_WITH_OPENCL)
    add_library(lite_static_all_in_one STATIC $<TARGET_OBJECTS:flatbuffers> $<TARGET_OBJECTS:cpuinfo> $<TARGET_OBJECTS:clog> $<TARGET_OBJECTS:OpenCL> $<TARGET_OBJECTS:lite_static>)
elseif (MGE_ENABLE_CPUINFO AND NOT MGE_WITH_OPENCL)
    add_library(lite_static_all_in_one STATIC $<TARGET_OBJECTS:flatbuffers> $<TARGET_OBJECTS:cpuinfo> $<TARGET_OBJECTS:clog> $<TARGET_OBJECTS:lite_static>)
elseif (NOT MGE_ENABLE_CPUINFO AND MGE_WITH_OPENCL)
    add_library(lite_static_all_in_one STATIC $<TARGET_OBJECTS:flatbuffers> $<TARGET_OBJECTS:OpenCL> $<TARGET_OBJECTS:lite_static>)
else()
    add_library(lite_static_all_in_one STATIC $<TARGET_OBJECTS:flatbuffers> $<TARGET_OBJECTS:lite_static>)
endif()
if(LITE_BUILD_WITH_MGE)
    target_link_libraries(lite_static_all_in_one PRIVATE megbrain megdnn ${MGE_CUDA_LIBS})
endif()
set(LITE_VERSION_SCRIPT ${PROJECT_SOURCE_DIR}/lite/src/version_lite.ld CACHE INTERNAL "Path to linker version script")
add_custom_target(_lite_version_ld SOURCES ${LITE_VERSION_SCRIPT})
if(NOT MSVC AND NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
endif()
#TODO: implemente version script for other OS
if (UNIX AND NOT APPLE)
    target_link_options(lite_shared PRIVATE -Wl,--version-script=${LITE_VERSION_SCRIPT})
    set_target_properties(lite_shared PROPERTIES LINK_DEPENDS ${LITE_VERSION_SCRIPT})
    target_link_options(lite_shared_whl PRIVATE -Wl,--version-script=${LITE_VERSION_SCRIPT})
    set_target_properties(lite_shared_whl PROPERTIES LINK_DEPENDS ${LITE_VERSION_SCRIPT})
endif()

# config install
install(TARGETS lite_static
    LIBRARY DESTINATION lite/lib/${MGE_ARCH}
    FRAMEWORK DESTINATION lite/lib/${MGE_ARCH}
    ARCHIVE DESTINATION lite/lib/${MGE_ARCH})

install(TARGETS lite_shared
    LIBRARY DESTINATION lite/lib/${MGE_ARCH}
    FRAMEWORK DESTINATION lite/lib/${MGE_ARCH}
    ARCHIVE DESTINATION lite/lib/${MGE_ARCH}
    )

install(TARGETS lite_static_all_in_one
    LIBRARY DESTINATION lite/lib/${MGE_ARCH}
    FRAMEWORK DESTINATION lite/lib/${MGE_ARCH}
    ARCHIVE DESTINATION lite/lib/${MGE_ARCH})
install(FILES ${PROJECT_SOURCE_DIR}/lite/include/lite/common_enum_c.h
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lite/include/lite-c)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/lite/include
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lite FILES_MATCHING PATTERN "*.h")

install(DIRECTORY ${PROJECT_SOURCE_DIR}/lite/lite-c/include
    DESTINATION ${CMAKE_INSTALL_PREFIX}/lite FILES_MATCHING PATTERN "*.h")

add_subdirectory(example)
if(MGE_WITH_TEST)
    add_subdirectory(test)
endif()

# tools and example
add_executable(rc4_encryptor tools/rc4_encrypt.cpp)

target_link_libraries(rc4_encryptor lite_static)
if(LITE_BUILD_WITH_MGE AND MGE_WITH_ROCM)
    # FIXME: hip obj can not find cpp obj only through lite_static
    target_link_libraries(rc4_encryptor megdnn)
endif()
target_include_directories(rc4_encryptor PRIVATE
    {PROJECT_SOURCE_DIR}/lite/src/decryption)
install (TARGETS rc4_encryptor
    EXPORT ${LITE_EXPORT_TARGETS}
    RUNTIME DESTINATION lite/tools)
