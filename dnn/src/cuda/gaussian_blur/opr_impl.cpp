/**
 * \file dnn/src/cuda/gaussian_blur/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./gaussian_blur.cuh"

#include "src/cuda/handle.h"
#include "src/cuda/utils.h"
#include "src/common/cv/common.h"
#include "src/common/cv/enums.h"
#include "src/common/cv/filter.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace cuda {

namespace intl {

template <typename ctype>
void gaussian_blur_exec(const ctype* src, ctype* dst, size_t N, size_t IH,
                        size_t IW, size_t IC, size_t stride0, size_t stride1,
                        size_t stride2, size_t stride3,
                        uint8_t* kernel_ptr, size_t kernel_height,
                        size_t kernel_width, double sigma_x, double sigma_y,
                        param::GaussianBlur::BorderMode bmode,
                        cudaStream_t stream) {
    megdnn_assert(IC == 1_z || IC == 3_z);
#define INIT_KERN(bmode)                                                   \
    if (IC == 1) {                                                         \
        gaussian_blur::gaussian_blur<ctype, 1, bmode>(                     \
                src, dst, N, IH, IW, stride0, stride1, stride2, stride3,   \
                kernel_ptr, kernel_height, kernel_width, sigma_x, sigma_y, \
                stream);                                                   \
    } else {                                                               \
        gaussian_blur::gaussian_blur<ctype, 3, bmode>(                     \
                src, dst, N, IH, IW, stride0, stride1, stride2, stride3,   \
                kernel_ptr, kernel_height, kernel_width, sigma_x, sigma_y, \
                stream);                                                   \
    }

    switch (bmode) {
        case param::GaussianBlur::BorderMode::BORDER_REPLICATE:
            INIT_KERN(BORDER_REPLICATE);
            break;
        case param::GaussianBlur::BorderMode::BORDER_REFLECT:
            INIT_KERN(::BorderMode::BORDER_REFLECT);
            break;
        case param::GaussianBlur::BorderMode::BORDER_REFLECT_101:
            INIT_KERN(::BorderMode::BORDER_REFLECT_101);
            break;
        case param::GaussianBlur::BorderMode::BORDER_CONSTANT:
            INIT_KERN(::BorderMode::BORDER_CONSTANT);
            break;
        default:
            MegCVException("Unsupport Bordermode in GaussianBlur\n");
    }
}

}  // namespace intl

void GaussianBlurImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                            _megdnn_workspace workspace) {
    megdnn_assert(src.layout.dtype == dtype::Uint8() ||
                  src.layout.dtype == dtype::Float32());
    check_exec(src.layout, dst.layout, workspace.size);

    auto stream = cuda_stream(handle());
    //! src layout is the same as dst layout
    size_t N = src.layout.shape[0];
    size_t batch_size = 0;
#define cb(DType)                                                              \
    if (src.layout.dtype == DType()) {                                         \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        ctype* src_ptr = src.ptr<ctype>() + curr_batch * src.layout.stride[0]; \
        ctype* dst_ptr = dst.ptr<ctype>() + curr_batch * src.layout.stride[0]; \
        batch_size = std::min<size_t>(N - curr_batch, max_batch_x_channel);    \
        intl::gaussian_blur_exec<ctype>(                                       \
                src_ptr, dst_ptr, batch_size, src.layout.shape[1],             \
                src.layout.shape[2], src.layout.shape[3],                      \
                src.layout.stride[0], src.layout.stride[1],                    \
                src.layout.stride[2], src.layout.stride[3],                    \
                workspace.ptr<uint8_t>(), m_kernel_height, m_kernel_width,     \
                m_sigma_x, m_sigma_y, param().border_mode, stream);            \
    }

    size_t max_batch_x_channel = max_batch_x_channel_size();
    size_t curr_batch = 0;
    if (N <= max_batch_x_channel) {
        cb(dtype::Uint8);
        cb(dtype::Float32);
    } else {
        while (curr_batch < N) {
            cb(dtype::Uint8);
            cb(dtype::Float32);

            curr_batch += max_batch_x_channel;
        }
    }
#undef cb
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
