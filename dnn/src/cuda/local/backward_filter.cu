/**
 * \file dnn/src/cuda/local/backward_filter.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/local/local.cuh"

#include "src/cuda/utils.cuh"
#include "src/cuda/local/cuda-convnet2/nvmatrix.cuh"
#include "src/cuda/local/cuda-convnet2/cudaconv2.cuh"

namespace megdnn {
namespace cuda {
namespace local {

bool can_backward_filter_proxy_convnet(size_t N,
        size_t IC, size_t /* IH */, size_t /* IW */,
        size_t /*OC*/, size_t /* OH */, size_t /* OW */,
        size_t FH, size_t FW,
        size_t /* INs */, size_t /* ONs */,
        size_t PH, size_t PW,
        size_t SH, size_t SW)
{
    bool flag = true;
    // check pad
    flag &= (PH == PW);
    // check stride
    flag &= (SH == SW);
    // megdnn_assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 16 == 0)));
    flag &= (IC <= 3 || IC % 8 == 0);
    // megdnn_assert(numFilters % (16 * numGroups) == 0);
    //flag &= (OC % 16 == 0);
    // megdnn_assert(filterSize * filterSize == filterPixels);
    flag &= (FH == FW);
    flag &= (SH <= FH);
    flag &= (N % 32 == 0);
    return flag;
}

size_t get_workspace_in_floats_backward_filter_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t /* FH */, size_t /* FW */,
        size_t /* INs */, size_t /* ONs */,
        size_t /* PH */, size_t /* PW */,
        size_t /* SH */, size_t /* SW */)
{
    return N*IC*IH*IW + N*OC*OH*OW;
}

void backward_filter_proxy_convnet(const float *src,
        const float *diff,
        float *grad,
        float *workspace,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t /* PW */,
        size_t SH, size_t /* SW */,
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        float *one, float *zero)
{
    MemorySegment mimage_n(const_cast<float *>(src)),
                  mhid_n(const_cast<float *>(diff)),
                  mimage_t(workspace),
                  mhid_t(workspace+N*IC*IH*IW),
                  mtarget(grad);
    NVMatrix nvimage_n(&mimage_n, N, IC*IH*IW, INs),
             nvhid_n(&mhid_n, N, OC*OH*OW, ONs),
             nvimage_t(&mimage_t, IC*IH*IW, N),
             nvhid_t(&mhid_t, OC*OH*OW, N),
             nvtarget(&mtarget, OH*OW*IC*FH*FW, OC);

    nvhid_n.transpose(nvhid_t, cublas_handle, one, zero);
    nvimage_n.transpose(nvimage_t, cublas_handle, one, zero);

    localWeightActs(stream, nvimage_t, nvhid_t, nvtarget,
            IH, OH, OW, FH, -static_cast<int>(PH), SH, IC, 1);
    after_kernel_launch();
}

} // namespace local
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
