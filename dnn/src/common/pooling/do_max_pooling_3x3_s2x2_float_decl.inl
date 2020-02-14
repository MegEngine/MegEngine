/**
 * \file dnn/src/common/pooling/do_max_pooling_3x3_s2x2_float_decl.inl
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
// simd_macro/*_helper.h should be included before including this file.
//
// The following function would be declared in this file:
//
// void do_max_pooling_3x3_s2x2_float_MEGDNN_SIMD_NAME(const float *src,
//      const float *filter, float *dst,
//      size_t IH, size_t IW, size_t OH, size_t OW,
//      size_t FH, size_t FW, size_t PH, size_t PW)
#include "src/common/macro_helper.h"
#include "src/common/utils.h"

#include "megdnn/arch.h"

namespace megdnn {

#define FUNC_NAME CONCAT_STR(do_max_pooling_3x3_s2x2_float_, MEGDNN_SIMD_NAME)

void FUNC_NAME(const float *src, float *dst,
        size_t IH_, size_t IW_, size_t OH_, size_t OW_, size_t PH_, size_t PW_,
        const WorkspaceBundle& ws)
MEGDNN_SIMD_ATTRIBUTE_TARGET;

#undef FUNC_NAME

}

#include "src/common/macro_helper_epilogue.h"

