set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
set(MGB_TEST_DIR ${PROJECT_SOURCE_DIR}/test/src)

file(GLOB_RECURSE SOURCES ../src/test/*.cpp ../src/impl/*.cpp ${MGB_TEST_DIR}/*.cpp)

# disable distributed tests
if(NOT MGE_WITH_DISTRIBUTED)
    list(FILTER SOURCES EXCLUDE REGEX ".*test/collective_comm.cpp")
    list(FILTER SOURCES EXCLUDE REGEX ".*test/io_remote.cpp")
endif()

# TODO: turn python binding into a static/object library
add_executable(imperative_test ${SOURCES} ${SRCS})
target_include_directories(imperative_test PRIVATE ${MGB_TEST_DIR}/include ../src/include ${MGB_OPDEF_OUT_DIR})

# Python binding
target_include_directories(imperative_test PRIVATE ${MODULE_SRC_INCLUDE} ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
target_compile_definitions(imperative_test PRIVATE MODULE_NAME=C)
target_compile_options(imperative_test PRIVATE -Wno-unused-parameter)

set(LINK_LIBS megbrain megdnn gtest gmock pybind11::embed range-v3)
if(MGE_WITH_CUDA)
    list(APPEND LINK_LIBS cudart)
endif()

if(MGE_WITH_DISTRIBUTED)
    list(APPEND LINK_LIBS megray)
endif()

target_link_libraries(imperative_test ${LINK_LIBS})
if(CXX_SUPPORT_WCLASS_MEMACCESS)
    if(MGE_WITH_CUDA)
        target_compile_options(imperative_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-class-memaccess>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-class-memaccess>")
    else()
        target_compile_options(imperative_test PRIVATE "-Wno-class-memaccess")
    endif()
endif()

if(UNIX)
    if(APPLE OR ANDROID)
        target_link_libraries(imperative_test dl)
    else()
        target_link_libraries(imperative_test dl rt)
    endif()
endif()

install(TARGETS imperative_test RUNTIME DESTINATION test)
