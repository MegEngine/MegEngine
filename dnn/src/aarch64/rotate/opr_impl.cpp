/**
 * \file dnn/src/aarch64/rotate/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <cstring>

#include "src/aarch64/rotate/opr_impl.h"
#include "src/aarch64/handle.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"


namespace megdnn {
namespace megcv {

void rotate_8uc1_clockwise_16x16(const uchar *src,
        uchar *dst,
        size_t src_step, size_t dst_step)
{
    asm volatile ("\n"
            "ld1 {v0.16b}, [%[src]], %[src_step] \n"
            "ld1 {v1.16b}, [%[src]], %[src_step] \n"
            "ld1 {v2.16b}, [%[src]], %[src_step] \n"
            "ld1 {v3.16b}, [%[src]], %[src_step] \n"
            "ld1 {v4.16b}, [%[src]], %[src_step] \n"
            "ld1 {v5.16b}, [%[src]], %[src_step] \n"
            "ld1 {v6.16b}, [%[src]], %[src_step] \n"
            "ld1 {v7.16b}, [%[src]], %[src_step] \n"
            "ld1 {v8.16b}, [%[src]], %[src_step] \n"
            "ld1 {v9.16b}, [%[src]], %[src_step] \n"
            "ld1 {v10.16b}, [%[src]], %[src_step] \n"
            "ld1 {v11.16b}, [%[src]], %[src_step] \n"
            "ld1 {v12.16b}, [%[src]], %[src_step] \n"
            "ld1 {v13.16b}, [%[src]], %[src_step] \n"
            "ld1 {v14.16b}, [%[src]], %[src_step] \n"
            "ld1 {v15.16b}, [%[src]], %[src_step] \n"

            "trn1 v16.16b, v0.16b, v1.16b \n"
            "trn2 v17.16b, v0.16b, v1.16b \n"
            "trn1 v18.16b, v2.16b, v3.16b \n"
            "trn2 v19.16b, v2.16b, v3.16b \n"
            "trn1 v20.16b, v4.16b, v5.16b \n"
            "trn2 v21.16b, v4.16b, v5.16b \n"
            "trn1 v22.16b, v6.16b, v7.16b \n"
            "trn2 v23.16b, v6.16b, v7.16b \n"
            "trn1 v24.16b, v8.16b, v9.16b \n"
            "trn2 v25.16b, v8.16b, v9.16b \n"
            "trn1 v26.16b, v10.16b, v11.16b \n"
            "trn2 v27.16b, v10.16b, v11.16b \n"
            "trn1 v28.16b, v12.16b, v13.16b \n"
            "trn2 v29.16b, v12.16b, v13.16b \n"
            "trn1 v30.16b, v14.16b, v15.16b \n"
            "trn2 v31.16b, v14.16b, v15.16b \n"

            "trn1 v0.8h, v16.8h, v18.8h \n"
            "trn2 v2.8h, v16.8h, v18.8h \n"
            "trn1 v4.8h, v20.8h, v22.8h \n"
            "trn2 v6.8h, v20.8h, v22.8h \n"
            "trn1 v8.8h, v24.8h, v26.8h \n"
            "trn2 v10.8h, v24.8h, v26.8h \n"
            "trn1 v12.8h, v28.8h, v30.8h \n"
            "trn2 v14.8h, v28.8h, v30.8h \n"
            "trn1 v1.8h, v17.8h, v19.8h \n"
            "trn2 v3.8h, v17.8h, v19.8h \n"
            "trn1 v5.8h, v21.8h, v23.8h \n"
            "trn2 v7.8h, v21.8h, v23.8h \n"
            "trn1 v9.8h, v25.8h, v27.8h \n"
            "trn2 v11.8h, v25.8h, v27.8h \n"
            "trn1 v13.8h, v29.8h, v31.8h \n"
            "trn2 v15.8h, v29.8h, v31.8h \n"

            "trn1 v16.4s, v0.4s, v4.4s \n"
            "trn2 v20.4s, v0.4s, v4.4s \n"
            "trn1 v24.4s, v8.4s, v12.4s \n"
            "trn2 v28.4s, v8.4s, v12.4s \n"
            "trn1 v17.4s, v1.4s, v5.4s \n"
            "trn2 v21.4s, v1.4s, v5.4s \n"
            "trn1 v25.4s, v9.4s, v13.4s \n"
            "trn2 v29.4s, v9.4s, v13.4s \n"
            "trn1 v18.4s, v2.4s, v6.4s \n"
            "trn2 v22.4s, v2.4s, v6.4s \n"
            "trn1 v26.4s, v10.4s, v14.4s \n"
            "trn2 v30.4s, v10.4s, v14.4s \n"
            "trn1 v19.4s, v3.4s, v7.4s \n"
            "trn2 v23.4s, v3.4s, v7.4s \n"
            "trn1 v27.4s, v11.4s, v15.4s \n"
            "trn2 v31.4s, v11.4s, v15.4s \n"

            "trn1 v0.2d, v16.2d, v24.2d \n"
            "trn2 v8.2d, v16.2d, v24.2d \n"
            "trn1 v1.2d, v17.2d, v25.2d \n"
            "trn2 v9.2d, v17.2d, v25.2d \n"
            "trn1 v2.2d, v18.2d, v26.2d \n"
            "trn2 v10.2d, v18.2d, v26.2d \n"
            "trn1 v3.2d, v19.2d, v27.2d \n"
            "trn2 v11.2d, v19.2d, v27.2d \n"
            "trn1 v4.2d, v20.2d, v28.2d \n"
            "trn2 v12.2d, v20.2d, v28.2d \n"
            "trn1 v5.2d, v21.2d, v29.2d \n"
            "trn2 v13.2d, v21.2d, v29.2d \n"
            "trn1 v6.2d, v22.2d, v30.2d \n"
            "trn2 v14.2d, v22.2d, v30.2d \n"
            "trn1 v7.2d, v23.2d, v31.2d \n"
            "trn2 v15.2d, v23.2d, v31.2d \n"
// There is no rev128 instruction, so we use rev64 and ext to simulate it.
            "rev64 v0.16b, v0.16b \n"
            "rev64 v1.16b, v1.16b \n"
            "rev64 v2.16b, v2.16b \n"
            "rev64 v3.16b, v3.16b \n"
            "rev64 v4.16b, v4.16b \n"
            "rev64 v5.16b, v5.16b \n"
            "rev64 v6.16b, v6.16b \n"
            "rev64 v7.16b, v7.16b \n"
            "rev64 v8.16b, v8.16b \n"
            "rev64 v9.16b, v9.16b \n"
            "rev64 v10.16b, v10.16b \n"
            "rev64 v11.16b, v11.16b \n"
            "rev64 v12.16b, v12.16b \n"
            "rev64 v13.16b, v13.16b \n"
            "rev64 v14.16b, v14.16b \n"
            "rev64 v15.16b, v15.16b \n"
            "ext v0.16b, v0.16b, v0.16b, #8 \n"
            "ext v1.16b, v1.16b, v1.16b, #8 \n"
            "ext v2.16b, v2.16b, v2.16b, #8 \n"
            "ext v3.16b, v3.16b, v3.16b, #8 \n"
            "ext v4.16b, v4.16b, v4.16b, #8 \n"
            "ext v5.16b, v5.16b, v5.16b, #8 \n"
            "ext v6.16b, v6.16b, v6.16b, #8 \n"
            "ext v7.16b, v7.16b, v7.16b, #8 \n"
            "ext v8.16b, v8.16b, v8.16b, #8 \n"
            "ext v9.16b, v9.16b, v9.16b, #8 \n"
            "ext v10.16b, v10.16b, v10.16b, #8 \n"
            "ext v11.16b, v11.16b, v11.16b, #8 \n"
            "ext v12.16b, v12.16b, v12.16b, #8 \n"
            "ext v13.16b, v13.16b, v13.16b, #8 \n"
            "ext v14.16b, v14.16b, v14.16b, #8 \n"
            "ext v15.16b, v15.16b, v15.16b, #8 \n"

            "st1 {v0.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v1.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v2.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v3.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v4.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v5.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v6.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v7.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v8.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v9.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v10.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v11.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v12.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v13.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v14.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v15.16b}, [%[dst]], %[dst_step] \n"
            :
            [src] "+r" (src),
            [dst] "+r" (dst)
            :
            [src_step] "r" (src_step),
            [dst_step] "r" (dst_step)
            :
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            );
}

void rotate_8uc1_counterclockwise_16x16(const uchar *src,
        uchar *dst,
        size_t src_step, size_t dst_step)
{
    asm volatile ("\n"
            "ld1 {v0.16b}, [%[src]], %[src_step] \n"
            "ld1 {v1.16b}, [%[src]], %[src_step] \n"
            "ld1 {v2.16b}, [%[src]], %[src_step] \n"
            "ld1 {v3.16b}, [%[src]], %[src_step] \n"
            "ld1 {v4.16b}, [%[src]], %[src_step] \n"
            "ld1 {v5.16b}, [%[src]], %[src_step] \n"
            "ld1 {v6.16b}, [%[src]], %[src_step] \n"
            "ld1 {v7.16b}, [%[src]], %[src_step] \n"
            "ld1 {v8.16b}, [%[src]], %[src_step] \n"
            "ld1 {v9.16b}, [%[src]], %[src_step] \n"
            "ld1 {v10.16b}, [%[src]], %[src_step] \n"
            "ld1 {v11.16b}, [%[src]], %[src_step] \n"
            "ld1 {v12.16b}, [%[src]], %[src_step] \n"
            "ld1 {v13.16b}, [%[src]], %[src_step] \n"
            "ld1 {v14.16b}, [%[src]], %[src_step] \n"
            "ld1 {v15.16b}, [%[src]], %[src_step] \n"

            "trn1 v16.16b, v0.16b, v1.16b \n"
            "trn2 v17.16b, v0.16b, v1.16b \n"
            "trn1 v18.16b, v2.16b, v3.16b \n"
            "trn2 v19.16b, v2.16b, v3.16b \n"
            "trn1 v20.16b, v4.16b, v5.16b \n"
            "trn2 v21.16b, v4.16b, v5.16b \n"
            "trn1 v22.16b, v6.16b, v7.16b \n"
            "trn2 v23.16b, v6.16b, v7.16b \n"
            "trn1 v24.16b, v8.16b, v9.16b \n"
            "trn2 v25.16b, v8.16b, v9.16b \n"
            "trn1 v26.16b, v10.16b, v11.16b \n"
            "trn2 v27.16b, v10.16b, v11.16b \n"
            "trn1 v28.16b, v12.16b, v13.16b \n"
            "trn2 v29.16b, v12.16b, v13.16b \n"
            "trn1 v30.16b, v14.16b, v15.16b \n"
            "trn2 v31.16b, v14.16b, v15.16b \n"

            "trn1 v0.8h, v16.8h, v18.8h \n"
            "trn2 v2.8h, v16.8h, v18.8h \n"
            "trn1 v4.8h, v20.8h, v22.8h \n"
            "trn2 v6.8h, v20.8h, v22.8h \n"
            "trn1 v8.8h, v24.8h, v26.8h \n"
            "trn2 v10.8h, v24.8h, v26.8h \n"
            "trn1 v12.8h, v28.8h, v30.8h \n"
            "trn2 v14.8h, v28.8h, v30.8h \n"
            "trn1 v1.8h, v17.8h, v19.8h \n"
            "trn2 v3.8h, v17.8h, v19.8h \n"
            "trn1 v5.8h, v21.8h, v23.8h \n"
            "trn2 v7.8h, v21.8h, v23.8h \n"
            "trn1 v9.8h, v25.8h, v27.8h \n"
            "trn2 v11.8h, v25.8h, v27.8h \n"
            "trn1 v13.8h, v29.8h, v31.8h \n"
            "trn2 v15.8h, v29.8h, v31.8h \n"

            "trn1 v16.4s, v0.4s, v4.4s \n"
            "trn2 v20.4s, v0.4s, v4.4s \n"
            "trn1 v24.4s, v8.4s, v12.4s \n"
            "trn2 v28.4s, v8.4s, v12.4s \n"
            "trn1 v17.4s, v1.4s, v5.4s \n"
            "trn2 v21.4s, v1.4s, v5.4s \n"
            "trn1 v25.4s, v9.4s, v13.4s \n"
            "trn2 v29.4s, v9.4s, v13.4s \n"
            "trn1 v18.4s, v2.4s, v6.4s \n"
            "trn2 v22.4s, v2.4s, v6.4s \n"
            "trn1 v26.4s, v10.4s, v14.4s \n"
            "trn2 v30.4s, v10.4s, v14.4s \n"
            "trn1 v19.4s, v3.4s, v7.4s \n"
            "trn2 v23.4s, v3.4s, v7.4s \n"
            "trn1 v27.4s, v11.4s, v15.4s \n"
            "trn2 v31.4s, v11.4s, v15.4s \n"

            "trn1 v0.2d, v16.2d, v24.2d \n"
            "trn2 v8.2d, v16.2d, v24.2d \n"
            "trn1 v1.2d, v17.2d, v25.2d \n"
            "trn2 v9.2d, v17.2d, v25.2d \n"
            "trn1 v2.2d, v18.2d, v26.2d \n"
            "trn2 v10.2d, v18.2d, v26.2d \n"
            "trn1 v3.2d, v19.2d, v27.2d \n"
            "trn2 v11.2d, v19.2d, v27.2d \n"
            "trn1 v4.2d, v20.2d, v28.2d \n"
            "trn2 v12.2d, v20.2d, v28.2d \n"
            "trn1 v5.2d, v21.2d, v29.2d \n"
            "trn2 v13.2d, v21.2d, v29.2d \n"
            "trn1 v6.2d, v22.2d, v30.2d \n"
            "trn2 v14.2d, v22.2d, v30.2d \n"
            "trn1 v7.2d, v23.2d, v31.2d \n"
            "trn2 v15.2d, v23.2d, v31.2d \n"

            "st1 {v15.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v14.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v13.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v12.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v11.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v10.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v9.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v8.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v7.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v6.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v5.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v4.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v3.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v2.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v1.16b}, [%[dst]], %[dst_step] \n"
            "st1 {v0.16b}, [%[dst]], %[dst_step] \n"
            :
            [src] "+r" (src),
            [dst] "+r" (dst)
            :
            [src_step] "r" (src_step),
            [dst_step] "r" (dst_step)
            :
            "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7",
            "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15"
            );
}

void rotate_8uc1_clockwise(const uchar* src, uchar* dst, const size_t rows,
                           const size_t cols, const size_t src_step,
                           const size_t dst_step) {
    const size_t block = 16;
    (void)block;
    size_t i = 0;

    for (; i + block <= rows; i += block) {
        size_t j = 0;
        for (; j + block <= cols; j += block) {
            rotate_8uc1_clockwise_16x16(
                    src + i * src_step + j,
                    dst + j * dst_step + (rows - (i + block)), src_step,
                    dst_step);
        }
        for (; j < cols; ++j) {
            for (size_t k = 0; k < block; ++k) {
                dst[j * dst_step + (rows - 1 - (i + k))] =
                        src[(i + k) * src_step + j];
            }
        }
    }

    for (; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[j * dst_step + (rows - 1 - i)] = src[i * src_step + j];
        }
    }
}

void rotate_8uc1_counterclockwise(const uchar* src, uchar* dst,
                                  const size_t rows, const size_t cols,
                                  const size_t src_step,
                                  const size_t dst_step) {
    const size_t block = 16;
    (void)block;
    size_t i = 0;

    for (; i + block <= rows; i += block) {
        size_t j = 0;
        for (; j + block <= cols; j += block) {
            rotate_8uc1_counterclockwise_16x16(
                    src + i * src_step + j,
                    dst + (cols - (j + block)) * dst_step + i, src_step,
                    dst_step);
        }
        for (; j < cols; ++j) {
            for (size_t k = 0; k < block; ++k) {
                dst[(cols - 1 - j) * dst_step + (i + k)] =
                        src[(i + k) * src_step + j];
            }
        }
    }

    for (; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[(cols - 1 - j) * dst_step + i] = src[i * src_step + j];
        }
    }
}

void rotate(const Mat<uchar>& src, Mat<uchar>& dst, bool clockwise) {
    megdnn_assert(src.rows() == dst.cols());
    megdnn_assert(src.cols() == dst.rows());
    megdnn_assert(src.channels() == dst.channels());
    megdnn_assert(src.channels() == 1_z);
    if (clockwise) {
        rotate_8uc1_clockwise(src.ptr(), dst.ptr(), src.rows(), src.cols(),
                              src.step(), dst.step());
    } else {
        rotate_8uc1_counterclockwise(src.ptr(), dst.ptr(), src.rows(),
                                     src.cols(), src.step(), dst.step());
    }
}

}  // namespace megcv

namespace aarch64 {

void RotateImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);

    //! rotate only support data type is uchar and the channel size is 1
    if (dst.layout.dtype != dtype::Uint8() || src.layout.shape[3] != 1) {
        return fallback::RotateImpl::exec(src, dst, workspace);
    }

    MEGDNN_DISPATCH_CPU_KERN_OPR({
        for (size_t i = 0; i < src.layout.shape[0]; ++i) {
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);
            rotate(src_mat, dst_mat, param().clockwise);
        }
    });
}

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
