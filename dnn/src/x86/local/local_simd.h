/**
 * \file dnn/src/x86/local/local_simd.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/simd_macro/sse_helper.h"
#include "src/common/local/local_decl.inl"
#include "src/x86/simd_macro/sse_helper_epilogue.h"

#include "src/x86/simd_macro/avx_helper.h"
#include "src/common/local/local_decl.inl"
#include "src/x86/simd_macro/avx_helper_epilogue.h"

#include "src/x86/simd_macro/fma_helper.h"
#include "src/common/local/local_decl.inl"
#include "src/x86/simd_macro/fma_helper_epilogue.h"
