include_directories(src)
file (GLOB_RECURSE SOURCES src/*.cpp main.cpp)
add_executable (load_and_run  ${SOURCES})

target_link_libraries (load_and_run megbrain)
install (TARGETS load_and_run RUNTIME DESTINATION bin)

if(MGE_WITH_TEST)
    add_executable(json_loader_test test/json_loader_test.cpp src/json_loader.h src/json_loader.cpp)
    target_link_libraries (json_loader_test megbrain)
endif()
