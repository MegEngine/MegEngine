add_executable(lite_c_examples ./main.c)
if(LITE_BUILD_WITH_RKNPU)
  # rknn sdk1.0.0 depend on libc++_shared, use gold to remove NEEDED so symbol check
  target_link_options(lite_c_examples PRIVATE "-fuse-ld=gold")
endif()
target_link_libraries(lite_c_examples lite_static)
if(LITE_BUILD_WITH_MGE AND MGE_WITH_ROCM)
  # FIXME: hip obj can not find cpp obj only through lite_static
  target_link_libraries(lite_c_examples megdnn)
endif()

if(UNIX)
  if(APPLE OR ANDROID)
    target_link_libraries(lite_c_examples dl)
  else()
    target_link_libraries(lite_c_examples dl rt)
  endif()
endif()

install(
  TARGETS lite_c_examples
  EXPORT ${LITE_EXPORT_TARGETS}
  RUNTIME DESTINATION lite/bin)

# add lite_examples_depends_shared for CI check symbol export valid
add_executable(lite_c_examples_depends_shared ./main.c)

if(LITE_BUILD_WITH_RKNPU)
  # rknn sdk1.0.0 depend on libc++_shared, use gold to remove NEEDED so symbol check
  target_link_options(lite_c_examples_depends_shared PRIVATE "-fuse-ld=gold")
endif()

target_link_libraries(lite_c_examples_depends_shared lite_shared)

if(UNIX)
  if(APPLE OR ANDROID)
    target_link_libraries(lite_c_examples_depends_shared dl)
  else()
    target_link_libraries(lite_c_examples_depends_shared dl rt)
  endif()
endif()

install(
  TARGETS lite_c_examples_depends_shared
  EXPORT ${LITE_EXPORT_TARGETS}
  RUNTIME DESTINATION lite/bin)
