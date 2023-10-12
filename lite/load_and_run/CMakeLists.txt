# BUILD the load and run for lite
include_directories(PUBLIC
                    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/lite/load_and_run/src>)
file(GLOB_RECURSE SOURCES src/**/*.cpp ${PROJECT_SOURCE_DIR}/lite/src/pack_model/*.cpp)

add_library(lar_object OBJECT ${SOURCES})

target_link_libraries(lar_object lite_static)
target_link_libraries(lar_object megbrain)
if(APPLE)
  target_link_libraries(lar_object gflags)
else()
  target_link_libraries(lar_object gflags -Wl,--version-script=${MGE_VERSION_SCRIPT})
endif()

if(LITE_BUILD_WITH_MGE
   AND NOT WIN32
   AND NOT APPLE)
  # FXIME third_party cpp redis do not support build with clang-cl
  target_include_directories(lar_object PRIVATE ${CPP_REDIS_INCLUDES})
endif()

add_executable(load_and_run src/main.cpp)
target_link_libraries(load_and_run lar_object)

if(LITE_BUILD_WITH_RKNPU)
  # rknn sdk1.0.0 depend on libc++_shared, use gold to remove NEEDED so symbol check
  target_link_options(load_and_run PRIVATE "-fuse-ld=gold")
endif()

if(LITE_BUILD_WITH_MGE AND MGE_WITH_ROCM)
  message(WARNING "MGE_WITH_ROCM is valid link to megdnn")
  # FIXME: hip obj can not find cpp obj only through lite_static
  target_link_libraries(load_and_run megdnn)
endif()

if(UNIX)
  if(APPLE
     OR ANDROID
     OR OHOS)
    target_link_libraries(load_and_run dl)
  else()
    target_link_libraries(load_and_run dl rt)
  endif()
endif()

install(
  TARGETS load_and_run
  EXPORT ${LITE_EXPORT_TARGETS}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_SHARED_LIBS)
  if(LITE_BUILD_WITH_MGE
     AND NOT WIN32
     AND NOT APPLE)
    # FXIME third_party cpp redis do not support build with clang-cl
    list(APPEND SOURCES ${CPP_REDIS_SRCS})
  endif()
  add_executable(load_and_run_depends_shared ${SOURCES} src/main.cpp)
  target_link_libraries(load_and_run_depends_shared lite_shared)
  target_link_libraries(load_and_run_depends_shared gflags)
  target_link_libraries(load_and_run_depends_shared megengine)

  if(LITE_BUILD_WITH_RKNPU)
    # rknn sdk1.0.0 depend on libc++_shared, use gold to remove NEEDED so symbol check
    target_link_options(load_and_run_depends_shared PRIVATE "-fuse-ld=gold")
  endif()

  if(LITE_BUILD_WITH_MGE AND MGE_WITH_ROCM)
    message(WARNING "MGE_WITH_ROCM is valid link to megdnn")
    # FIXME: hip obj can not find cpp obj only through lite_static
    target_link_libraries(load_and_run_depends_shared megdnn)
  endif()

  if(UNIX)
    if(APPLE
       OR ANDROID
       OR OHOS)
      target_link_libraries(load_and_run_depends_shared dl)
    else()
      target_link_libraries(load_and_run_depends_shared dl rt)
    endif()
  endif()

  if(LITE_BUILD_WITH_MGE
     AND NOT WIN32
     AND NOT APPLE)
    # FXIME third_party cpp redis do not support build with clang-cl
    target_include_directories(load_and_run_depends_shared
                               PRIVATE ${CPP_REDIS_INCLUDES})
  endif()

  install(
    TARGETS load_and_run_depends_shared
    EXPORT ${MGE_EXPORT_TARGETS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
