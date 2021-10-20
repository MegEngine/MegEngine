/**
 * \file lite/build_config/lite_build_config.h
 *
 * This file is part of MegEngine, a deep learning framework developed by
 * Megvii.
 *
 * \copyright Copyright (c) 2020-2021 Megvii Inc. All rights reserved.
 */
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
#endif  // _HEADER_LITE_BUILD_CONFIG
