if (MGE_USE_SYSTEM_LIB)
    find_package(Cpuinfo)
    message(STATUS "Using system provided cpuinfo ${cpuinfo_VERSION}")
    add_library(libcpuinfo IMPORTED GLOBAL)
    set_target_properties(
        libcpuinfo PROPERTIES
        IMPORTED_LOCATION ${cpuinfo_LIBRARIES}
        INTERFACE_INCLUDE_DIRECTORIES ${cpuinfo_INCLUDE_DIRS}
        )
    return()
endif()

SET(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "Type of cpuinfo library (shared, static, or default) to build")
OPTION(CPUINFO_BUILD_TOOLS "Build command-line tools" OFF)
OPTION(CPUINFO_BUILD_UNIT_TESTS "Build cpuinfo unit tests" OFF)
OPTION(CPUINFO_BUILD_MOCK_TESTS "Build cpuinfo mock tests" OFF)
OPTION(CPUINFO_BUILD_BENCHMARKS "Build cpuinfo micro-benchmarks" OFF)
include_directories("${PROJECT_SOURCE_DIR}/third_party/cpuinfo/include")
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/cpuinfo ${CMAKE_CURRENT_BINARY_DIR}/cpuinfo EXCLUDE_FROM_ALL)

