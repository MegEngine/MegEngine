//! this file always for bazel

#ifndef _HEADER_LITE_BUILD_CONFIG
#define _HEADER_LITE_BUILD_CONFIG

#ifndef LITE_ENABLE_LOGGING
#define LITE_ENABLE_LOGGING 1
#endif

#ifndef LITE_ENABLE_EXCEPTION
#if __cpp_exceptions || __EXCEPTIONS || (defined(_MSC_VER) && defined(_CPPUNWIND))
#define LITE_ENABLE_EXCEPTION 1
#else
#define LITE_ENABLE_EXCEPTION 0
#endif
#endif

#ifndef LITE_WITH_CUDA
#define LITE_WITH_CUDA 0
#endif

#ifndef LITE_ASSERT_LOC
#define LITE_ASSERT_LOC 1
#endif

#ifndef LITE_BUILD_WITH_MGE
#define LITE_BUILD_WITH_MGE 1
#endif

#ifndef LITE_BUILD_WITH_RKNPU
#define LITE_BUILD_WITH_RKNPU 0
#endif
#endif  // _HEADER_LITE_BUILD_CONFIG
