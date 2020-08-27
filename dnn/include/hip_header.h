/**
 * \file dnn/include/hip_header.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

/**
 * \remarks The files in the subdirectory include/hip are copied from HIP
 * headers provided by ROCm-Developer-Tools/HIP, which can be found from
 * https://github.com/ROCm-Developer-Tools/HIP. These files are included to make
 * the MegDNN can be compiled with both CUDA and ROCm backends, and the both
 * backends share the same code.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#pragma GCC diagnostic pop

#if !defined(__HIP_PLATFORM_HCC__)
#error "platform macro __HIP_PLATFORM_HCC__ must be defined"
#endif

// vim: syntax=cpp.doxygen
