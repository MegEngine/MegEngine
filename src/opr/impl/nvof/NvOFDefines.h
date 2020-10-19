/*
* Copyright 2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/**
 * \file src/opr/impl/nvof/NvOFDefines.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megbrain_build_config.h"

#if MGB_CUDA
#pragma once
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
//FIXME: mgb code redefine CALLBACK, some win32 API will be disable
#undef CALLBACK
#undef CONST
#define DIR_SEP "\\"
#else
#define HMODULE void *
#define _stricmp strcasecmp
#define DIR_SEP "/"
#endif
#include <memory>

class NvOF;
class NvOFBuffer;

/**
* @brief A managed pointer wrapper for NvOF class objects
*/
using NvOFObj = std::unique_ptr<NvOF>;

/**
* @brief A managed pointer wrapper for NvOFBuffer class objects
*/
using NvOFBufferObj = std::unique_ptr<NvOFBuffer>;

#endif
