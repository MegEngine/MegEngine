if (MGE_WITH_TEST)
    file (GLOB_RECURSE SOURCES ./*.cpp main.cpp)
    add_executable (lite_test  ${SOURCES})

    target_link_libraries(lite_test gtest)
    target_link_libraries(lite_test lite_static)
    if(LITE_BUILD_WITH_MGE)
        # lite_test will depends megbrain interface
        target_link_libraries(lite_test megbrain)
    endif()

    if(UNIX)
        if(APPLE OR ANDROID)
            target_link_libraries(lite_test dl)
        else()
            target_link_libraries(lite_test dl rt)
        endif()
    endif()

    install (TARGETS lite_test
        EXPORT ${LITE_EXPORT_TARGETS}
        RUNTIME DESTINATION lite/bin)
endif()
