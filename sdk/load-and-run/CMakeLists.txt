include_directories(src)
file (GLOB_RECURSE SOURCES src/*.cpp main.cpp)
add_executable (load_and_run  ${SOURCES})

target_link_libraries (load_and_run megbrain)
install (TARGETS load_and_run RUNTIME DESTINATION bin)
