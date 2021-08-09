/**
 * \file dnn/include/megdnn/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain_build_config.h"

#if MGB_ENABLE_GETENV
#define MGB_GETENV  ::std::getenv
#else
#define MGB_GETENV(_name)  static_cast<char*>(nullptr)
#endif

#ifdef WIN32
#define unsetenv(_name) _putenv_s(_name, "");
#define setenv(name,value,overwrite) _putenv_s(name,value)
#endif


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
