/**
 * \file dnn/src/x86/conv_bias/f32/do_conv_stride2.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
// clang-format off
#include "src/x86/simd_macro/sse_helper.h"
#include "src/fallback/convolution/do_conv_stride2_decl.inl"
#include "src/x86/simd_macro/sse_helper_epilogue.h"
// clang-format on
