/**
 * \file dnn/include/megdnn/config/config.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if !defined(__CUDACC__)

// Try to detect if no architecture flags defined.
#if !defined(MEGDNN_NAIVE) && !defined(MEGDNN_X86) &&         \
        !defined(MEGDNN_X86_64) && !defined(MEGDNN_X86_32) && \
        !defined(MEGDNN_64_BIT) && !defined(MEGDNN_MIPS) &&   \
        !defined(MEGDNN_ARMV7) && !defined(MEGDNN_AARCH64)
#if defined(__x86_64__) || defined(_M_X64)
#define MEGDNN_X86 1
#define MEGDNN_X86_64 1
#define MEGDNN_64_BIT 1
#elif defined(__i386) || defined(_M_IX86)
#define MEGDNN_X86 1
#define MEGDNN_X86_32 1
#endif
#endif

#endif  // !defined(__CUDACC__)

// vim: syntax=cpp.doxygen
