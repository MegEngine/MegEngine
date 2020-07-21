include_directories(src)
file (GLOB_RECURSE SOURCES src/*.cpp main.cpp)
add_executable (load_and_run  ${SOURCES})

if (WIN32)
    # Windows does not support implicitly importing data members from DLL.
    target_link_libraries(load_and_run megbrain megdnn)
else()
    target_link_libraries (load_and_run megengine)
endif()
install (TARGETS load_and_run EXPORT ${MGE_EXPORT_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(MGE_WITH_TEST)
    add_executable(json_loader_test test/json_loader_test.cpp src/json_loader.h src/json_loader.cpp)
    target_link_libraries (json_loader_test megengine)
endif()
