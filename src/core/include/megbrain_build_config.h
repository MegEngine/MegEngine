/**
 * \file src/core/include/megbrain_build_config.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef _HEADER_MGB_BUILD_CONFIG
#define _HEADER_MGB_BUILD_CONFIG

// whether cuda is available
#ifndef MGB_CUDA
#define MGB_CUDA    1
#endif


// whether to include file/line location for assert message
#ifndef MGB_ASSERT_LOC
#define MGB_ASSERT_LOC          1
#endif

// whether to enable utils/debug.h and other debug methods
#ifndef MGB_ENABLE_DEBUG_UTIL
#define MGB_ENABLE_DEBUG_UTIL   1
#endif

// whether to enable logging
#ifndef MGB_ENABLE_LOGGING
#define MGB_ENABLE_LOGGING      1
#endif

// whether to enable registering opr grad functions
#ifndef MGB_ENABLE_GRAD
#define MGB_ENABLE_GRAD         1
#endif

// whether to include actual class name in mgb::Typeinfo object; if this is
// disabled, mgb::serialization::OprRegistry::find_opr_by_name would not work.
#ifndef MGB_VERBOSE_TYPEINFO_NAME
#define MGB_VERBOSE_TYPEINFO_NAME   1
#endif

// whether to enbale configuing megbrain internals through env vars
#ifndef MGB_ENABLE_GETENV
#define MGB_ENABLE_GETENV       1
#endif

// whether to remove unnecessary features when used for serving
#ifndef MGB_BUILD_SLIM_SERVING
#define MGB_BUILD_SLIM_SERVING  0
#endif

// whether to enable exception
#ifndef MGB_ENABLE_EXCEPTION
#if __EXCEPTIONS
#define MGB_ENABLE_EXCEPTION    1
#else
#define MGB_ENABLE_EXCEPTION    0
#endif
#endif

// whether <thread> is available and usable
#ifndef MGB_HAVE_THREAD
#define MGB_HAVE_THREAD         1
#endif

// whether to trade thread safety for memory usage
#ifndef MGB_THREAD_SAFE
#define MGB_THREAD_SAFE MGB_HAVE_THREAD
#endif

// whether to enable JIT
#ifndef MGB_JIT
#define MGB_JIT     1
#endif
#ifndef MGB_JIT_HALIDE
#define MGB_JIT_HALIDE 0
#endif


// whether to enable TensorRT support
#ifndef MGB_ENABLE_TENSOR_RT
#define MGB_ENABLE_TENSOR_RT    MGB_CUDA
#endif

// whether to enable fastrun profile
#ifndef MGB_ENABLE_FASTRUN
#define MGB_ENABLE_FASTRUN 1
#endif


/* ================= following are more finegrind controls ================= */

// whether to enable json dumper
#ifndef MGB_ENABLE_JSON
#define MGB_ENABLE_JSON !MGB_BUILD_SLIM_SERVING
#endif

#endif  // _HEADER_MGB_BUILD_CONFIG
