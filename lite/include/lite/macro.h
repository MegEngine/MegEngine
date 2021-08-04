/**
 * \file include/lite/macro.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#ifndef LITE_MACRO_H_
#define LITE_MACRO_H_

#if defined(_WIN32)
#define LITE_API __declspec(dllexport)
#else
#define LITE_API __attribute__((visibility("default")))
#endif
#endif

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
