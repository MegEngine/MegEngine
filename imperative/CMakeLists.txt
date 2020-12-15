find_package(NumPy REQUIRED)

set(PACKAGE_NAME megengine)
set(PACKAGE_NAME ${PACKAGE_NAME} PARENT_SCOPE)
set(MODULE_NAME _imperative_rt)
set(MODULE_NAME ${MODULE_NAME} PARENT_SCOPE)
file(GLOB_RECURSE SRCS src/impl/*.cpp src/include/*.h python/src/*.cpp python/src/*.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMGB_WITH_IMPERATIVE=1")

file(GLOB_RECURSE PYTHON_SRCS python/${PACKAGE_NAME}/*.py)
file(GLOB_RECURSE ALL_HEADERS src/cpp/megbrain_pubapi.h
    ${PROJECT_SOURCE_DIR}/src/core/include/*
    ${PROJECT_SOURCE_DIR}/src/opr/include/*
    ${PROJECT_SOURCE_DIR}/src/serialization/include/*
    ${PROJECT_SOURCE_DIR}/src/plugin/include/*
    ${PROJECT_SOURCE_DIR}/dnn/include/*)

set(MEGENGINE_DIR ${CMAKE_CURRENT_BINARY_DIR}/python/)

add_subdirectory(tablegen)

add_custom_target(_version_ld SOURCES ${MGE_VERSION_SCRIPT})

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11 ${PROJECT_BINARY_DIR}/third_party/pybind11)
pybind11_add_module(${MODULE_NAME} NO_EXTRAS ${SRCS})
if (APPLE)
    target_link_libraries(${MODULE_NAME} PRIVATE megengine_export)
elseif (MSVC OR WIN32)
    # Windows does not support implicitly importing data members from DLL.
    target_link_libraries(${MODULE_NAME} PRIVATE megbrain megdnn)
    message(STATUS "CMAKE_MSVC_RUNTIME_LIBRARY: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
    set_target_properties(${MODULE_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY "${CMAKE_MSVC_RUNTIME_LIBRARY}")
else()
    if (MGE_WITH_PYTHON_MODULE)
        # use to fix runtime crash when build both mgb(MGE_WITH_PYTHON_MODULE) and imperative(MGE_BUILD_IMPERATIVE_RT)
        target_link_libraries(${MODULE_NAME} PRIVATE megengine_export -Wl,--version-script=${MGE_VERSION_SCRIPT})
    else()
        # use to reduce whl size by depend on megbrain/dnn directly, caused by cmake create two cuda fatbin
        # elf section on both megengine_export and target which depend on megengine_export
        target_link_libraries(${MODULE_NAME} PRIVATE megbrain megdnn -Wl,--version-script=${MGE_VERSION_SCRIPT})
        if (MGE_WITH_DISTRIBUTED)
            message(VERBOSE "Imperative configured to link megray")
            target_link_libraries(${MODULE_NAME} PRIVATE megray)
        endif()
    endif()
endif()

target_include_directories(${MODULE_NAME} PUBLIC src/include PRIVATE ${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR} ${MGB_OPDEF_OUT_DIR})
target_compile_definitions(${MODULE_NAME} PRIVATE MODULE_NAME=${MODULE_NAME})
target_compile_options(${MODULE_NAME} PRIVATE -Wno-unused-parameter)
if(CXX_SUPPORT_WCLASS_MEMACCESS)
    target_compile_options(${MODULE_NAME} PRIVATE "-Wno-class-memaccess")
endif()
set_target_properties(${MODULE_NAME} PROPERTIES
    SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX}
    LIBRARY_OUTPUT_DIRECTORY ${MEGENGINE_DIR}/${PACKAGE_NAME}/core
)
if (APPLE OR MSVC OR WIN32)
    message(VERBOSE "overwriting SUFFIX at macos and windows before config by set_target_properties")
    pybind11_extension(${MODULE_NAME})
endif()
add_dependencies(${MODULE_NAME} mgb_opdef _version_ld)

if(MGE_WITH_TEST AND MGE_ENABLE_RTTI)
    add_subdirectory(test)
endif()

add_custom_command(
    TARGET ${MODULE_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/LICENSE ${PROJECT_SOURCE_DIR}/ACKNOWLEDGMENTS ${PROJECT_BINARY_DIR}
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine/core/$<TARGET_FILE_NAME:${MODULE_NAME}> # clean develop
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/megengine ${CMAKE_CURRENT_BINARY_DIR}/python/megengine
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/python/test ${CMAKE_CURRENT_BINARY_DIR}/python/test
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py ${CMAKE_CURRENT_BINARY_DIR}/python/setup.py
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires.txt ${CMAKE_CURRENT_BINARY_DIR}/python/requires.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires-style.txt ${CMAKE_CURRENT_BINARY_DIR}/python/requires-style.txt
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/python/requires-test.txt ${CMAKE_CURRENT_BINARY_DIR}/python/requires-test.txt
)

