if(MGE_WITH_TEST)
  include_directories(PUBLIC
                      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lite/load_and_run/src>)
  file(GLOB_RECURSE SOURCES ./*.cpp main.cpp)
  add_executable(lite_test ${SOURCES})
  target_link_libraries(lite_test lar_object)
  target_link_libraries(lite_test gtest)
  if(LITE_BUILD_WITH_MGE)
    if(MGE_WITH_ROCM)
      # FIXME: hip obj can not find cpp obj only through lite_static
      message(WARNING "MGE_WITH_ROCM is valid link to megdnn")
      target_link_libraries(lite_test megdnn)
    endif()
  endif()

  if(UNIX)
    if(APPLE
       OR ANDROID
       OR OHOS)
      target_link_libraries(lite_test dl)
    else()
      target_link_libraries(lite_test dl rt)
    endif()
  endif()

  install(
    TARGETS lite_test
    EXPORT ${LITE_EXPORT_TARGETS}
    RUNTIME DESTINATION lite/bin)
endif()
