add_library(atlas-stub STATIC src/libatlas-wrap.cpp)
target_include_directories(atlas-stub PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
install(TARGETS atlas-stub EXPORT ${MGE_EXPORT_TARGETS})

add_library(acl-cblas STATIC src/libacl_cblas-wrap.cpp)
target_include_directories(acl-cblas PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
