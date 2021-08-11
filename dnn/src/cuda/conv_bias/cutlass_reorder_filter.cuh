/**
 * \file dnn/src/cuda/conv_bias/cutlass_reorder_filter.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace cutlass_wrapper {

template <uint32_t size_bits, uint32_t interleaved>
void reorder_ncxhwx_imma_filter(int8_t* dst_filter, const int8_t* src_filter,
                                uint32_t OC, uint32_t IC, uint32_t FH,
                                uint32_t FW, bool trans_oc,
                                cudaStream_t stream);

template <uint32_t size_bits>
void reorder_nhwc_imma_filter(int8_t* dst_filter, const int8_t* src_filter,
                              uint32_t OC, uint32_t IC, uint32_t FH,
                              uint32_t FW, bool trans_oc, uint32_t alignbits,
                              uint32_t interleaved, cudaStream_t stream);
}  // namespace cutlass_wrapper
}  // namespace cuda
}  // namespace megdnn
