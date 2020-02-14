/**
 * \file dnn/src/cuda/mask_conv/mask_conv.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

namespace megdnn {
namespace cuda {
namespace mask_conv {

template <typename ctype>
void set_zero_by_mask_proxy(float* dst, const ctype* mask, size_t N, size_t OC,
                            size_t OH, size_t OW, cudaStream_t stream);

template <typename ctype>
void mask_propagate_exec_proxy(const ctype* src, ctype* dst, size_t IH,
                               size_t IW, size_t OH, size_t OW, size_t FH,
                               size_t FW, size_t SH, size_t SW, size_t PH,
                               size_t PW, size_t DH, size_t DW,
                               cudaStream_t stream);

}  // namespace mask_conv

}  // namespace cuda
}  // namespace megdnn
