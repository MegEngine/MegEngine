/**
 * \file dnn/src/cuda/dct/dct_channel_select.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the
 "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express
 or
 * implied.
 */
#pragma once
#include <stdint.h>
#include <cstdio>
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace dct {

using DctLayoutFormat = megdnn::param_enumv::DctChannelSelect::Format;

template <int dct_block, uint32_t format, typename DstDtype>
void call_kern_dct(const uint8_t* d_src, DstDtype* d_dst, const int n,
                   const int c, const int h, const int w, const int oc,
                   bool fix_32_mask, const int* mask_offset,
                   const int* mask_val, cudaStream_t stream,
                   megcore::AsyncErrorInfo* error_info, void* error_tracker,
                   float scale = 1.f);

}  // namespace dct
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen