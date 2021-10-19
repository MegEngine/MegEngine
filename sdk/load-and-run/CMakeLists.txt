include_directories(src)
file(GLOB_RECURSE SOURCES src/*.cpp main.cpp)

add_executable(load_and_run ${SOURCES})
target_link_libraries(load_and_run megbrain megdnn ${MGE_CUDA_LIBS})

# load_and_run_depends_shared always for CI check, please do not delete
if(BUILD_SHARED_LIBS)
    add_executable(load_and_run_depends_shared ${SOURCES})
    target_link_libraries(load_and_run_depends_shared megengine)
    if(WIN32 OR MSVC)
        target_compile_definitions(load_and_run_depends_shared PRIVATE MGE_DLL_IMPORT_DATA)
    endif()
endif()

install(TARGETS load_and_run EXPORT ${MGE_EXPORT_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(BUILD_SHARED_LIBS)
    install(TARGETS load_and_run_depends_shared EXPORT ${MGE_EXPORT_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(MGE_WITH_TEST)
    add_executable(json_loader_test test/json_loader_test.cpp src/json_loader.h src/json_loader.cpp)
    target_link_libraries(json_loader_test megbrain megdnn ${MGE_CUDA_LIBS})
endif()
