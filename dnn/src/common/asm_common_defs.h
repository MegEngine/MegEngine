/**
 * \file dnn/src/common/asm_common_defs.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#if defined(__WIN32__) || defined(__APPLE__)
# define cdecl(s) _##s
#else
# define cdecl(s) s
#endif

#if !defined(__APPLE__)
#define hidden_sym(s) .hidden cdecl(s)
#else
#define hidden_sym(s) .private_extern cdecl(s)
#endif

#if defined(__linux__) && defined(__ELF__) && (defined(__arm__) || defined(__aarch64__))
.pushsection .note.GNU-stack,"",%progbits
.popsection
#endif

