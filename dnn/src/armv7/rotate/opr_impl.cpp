/**
 * \file dnn/src/armv7/rotate/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstring>

#include "src/armv7/rotate/opr_impl.h"
#include "src/armv7/handle.h"
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
            "vld1.8 {d0, d1}, [%[src]], %[src_step] \n"
            "vld1.8 {d2, d3}, [%[src]], %[src_step] \n"
            "vld1.8 {d4, d5}, [%[src]], %[src_step] \n"
            "vld1.8 {d6, d7}, [%[src]], %[src_step] \n"
            "vld1.8 {d8, d9}, [%[src]], %[src_step] \n"
            "vld1.8 {d10, d11}, [%[src]], %[src_step] \n"
            "vld1.8 {d12, d13}, [%[src]], %[src_step] \n"
            "vld1.8 {d14, d15}, [%[src]], %[src_step] \n"
            "vld1.8 {d16, d17}, [%[src]], %[src_step] \n"
            "vld1.8 {d18, d19}, [%[src]], %[src_step] \n"
            "vld1.8 {d20, d21}, [%[src]], %[src_step] \n"
            "vld1.8 {d22, d23}, [%[src]], %[src_step] \n"
            "vld1.8 {d24, d25}, [%[src]], %[src_step] \n"
            "vld1.8 {d26, d27}, [%[src]], %[src_step] \n"
            "vld1.8 {d28, d29}, [%[src]], %[src_step] \n"
            "vld1.8 {d30, d31}, [%[src]], %[src_step] \n"
            "vtrn.8 q0, q1 \n"
            "vtrn.8 q2, q3 \n"
            "vtrn.8 q4, q5 \n"
            "vtrn.8 q6, q7 \n"
            "vtrn.8 q8, q9 \n"
            "vtrn.8 q10, q11 \n"
            "vtrn.8 q12, q13 \n"
            "vtrn.8 q14, q15 \n"
            "vtrn.16 q0, q2 \n"
            "vtrn.16 q1, q3 \n"
            "vtrn.16 q4, q6 \n"
            "vtrn.16 q5, q7 \n"
            "vtrn.16 q8, q10 \n"
            "vtrn.16 q9, q11 \n"
            "vtrn.16 q12, q14 \n"
            "vtrn.16 q13, q15 \n"
            "vtrn.32 q0, q4 \n"
            "vtrn.32 q1, q5 \n"
            "vtrn.32 q2, q6 \n"
            "vtrn.32 q3, q7 \n"
            "vtrn.32 q8, q12 \n"
            "vtrn.32 q9, q13 \n"
            "vtrn.32 q10, q14 \n"
            "vtrn.32 q11, q15 \n"
            "vswp d1, d16 \n"
            "vswp d3, d18 \n"
            "vswp d5, d20 \n"
            "vswp d7, d22 \n"
            "vswp d9, d24 \n"
            "vswp d11, d26 \n"
            "vswp d13, d28 \n"
            "vswp d15, d30 \n"
            "vswp d0, d1 \n"
            "vswp d2, d3 \n"
            "vswp d4, d5 \n"
            "vswp d6, d7 \n"
            "vswp d8, d9 \n"
            "vswp d10, d11 \n"
            "vswp d12, d13 \n"
            "vswp d14, d15 \n"
            "vswp d16, d17 \n"
            "vswp d18, d19 \n"
            "vswp d20, d21 \n"
            "vswp d22, d23 \n"
            "vswp d24, d25 \n"
            "vswp d26, d27 \n"
            "vswp d28, d29 \n"
            "vswp d30, d31 \n"
            "vrev64.8 q0, q0\n"
            "vrev64.8 q1, q1\n"
            "vrev64.8 q2, q2\n"
            "vrev64.8 q3, q3\n"
            "vrev64.8 q4, q4\n"
            "vrev64.8 q5, q5\n"
            "vrev64.8 q6, q6\n"
            "vrev64.8 q7, q7\n"
            "vrev64.8 q8, q8\n"
            "vrev64.8 q9, q9\n"
            "vrev64.8 q10, q10\n"
            "vrev64.8 q11, q11\n"
            "vrev64.8 q12, q12\n"
            "vrev64.8 q13, q13\n"
            "vrev64.8 q14, q14\n"
            "vrev64.8 q15, q15\n"
            "vst1.8 {d0, d1}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d2, d3}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d4, d5}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d6, d7}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d8, d9}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d10, d11}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d12, d13}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d14, d15}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d16, d17}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d18, d19}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d20, d21}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d22, d23}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d24, d25}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d26, d27}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d28, d29}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d30, d31}, [%[dst]], %[dst_step] \n"
            :
            [src] "+r" (src),
            [dst] "+r" (dst)
            :
            [src_step] "r" (src_step),
            [dst_step] "r" (dst_step)
            :
            "r0", "r1", "r2", "r3",
            "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
            "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
            "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
            "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31"
        );

}

void rotate_8uc1_counterclockwise_16x16(const uchar *src,
        uchar *dst,
        size_t src_step, size_t dst_step)
{
    asm volatile ("\n"
            "vld1.8 {d0, d1}, [%[src]], %[src_step] \n"
            "vld1.8 {d2, d3}, [%[src]], %[src_step] \n"
            "vld1.8 {d4, d5}, [%[src]], %[src_step] \n"
            "vld1.8 {d6, d7}, [%[src]], %[src_step] \n"
            "vld1.8 {d8, d9}, [%[src]], %[src_step] \n"
            "vld1.8 {d10, d11}, [%[src]], %[src_step] \n"
            "vld1.8 {d12, d13}, [%[src]], %[src_step] \n"
            "vld1.8 {d14, d15}, [%[src]], %[src_step] \n"
            "vld1.8 {d16, d17}, [%[src]], %[src_step] \n"
            "vld1.8 {d18, d19}, [%[src]], %[src_step] \n"
            "vld1.8 {d20, d21}, [%[src]], %[src_step] \n"
            "vld1.8 {d22, d23}, [%[src]], %[src_step] \n"
            "vld1.8 {d24, d25}, [%[src]], %[src_step] \n"
            "vld1.8 {d26, d27}, [%[src]], %[src_step] \n"
            "vld1.8 {d28, d29}, [%[src]], %[src_step] \n"
            "vld1.8 {d30, d31}, [%[src]], %[src_step] \n"
            "vtrn.8 q0, q1 \n"
            "vtrn.8 q2, q3 \n"
            "vtrn.8 q4, q5 \n"
            "vtrn.8 q6, q7 \n"
            "vtrn.8 q8, q9 \n"
            "vtrn.8 q10, q11 \n"
            "vtrn.8 q12, q13 \n"
            "vtrn.8 q14, q15 \n"
            "vtrn.16 q0, q2 \n"
            "vtrn.16 q1, q3 \n"
            "vtrn.16 q4, q6 \n"
            "vtrn.16 q5, q7 \n"
            "vtrn.16 q8, q10 \n"
            "vtrn.16 q9, q11 \n"
            "vtrn.16 q12, q14 \n"
            "vtrn.16 q13, q15 \n"
            "vtrn.32 q0, q4 \n"
            "vtrn.32 q1, q5 \n"
            "vtrn.32 q2, q6 \n"
            "vtrn.32 q3, q7 \n"
            "vtrn.32 q8, q12 \n"
            "vtrn.32 q9, q13 \n"
            "vtrn.32 q10, q14 \n"
            "vtrn.32 q11, q15 \n"
            "vswp d1, d16 \n"
            "vswp d3, d18 \n"
            "vswp d5, d20 \n"
            "vswp d7, d22 \n"
            "vswp d9, d24 \n"
            "vswp d11, d26 \n"
            "vswp d13, d28 \n"
            "vswp d15, d30 \n"
            "vst1.8 {d30, d31}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d28, d29}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d26, d27}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d24, d25}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d22, d23}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d20, d21}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d18, d19}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d16, d17}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d14, d15}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d12, d13}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d10, d11}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d8, d9}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d6, d7}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d4, d5}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d2, d3}, [%[dst]], %[dst_step] \n"
            "vst1.8 {d0, d1}, [%[dst]], %[dst_step] \n"
            :
            [src] "+r" (src),
            [dst] "+r" (dst)
            :
            [src_step] "r" (src_step),
            [dst_step] "r" (dst_step)
            :
            "r0", "r1", "r2", "r3",
            "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7",
            "d8", "d9", "d10", "d11", "d12", "d13", "d14", "d15",
            "d16", "d17", "d18", "d19", "d20", "d21", "d22", "d23",
            "d24", "d25", "d26", "d27", "d28", "d29", "d30", "d31"
        );
}

void rotate_8uc1_clockwise(const uchar *src, uchar *dst,
        const size_t rows, const size_t cols,
        const size_t src_step, const size_t dst_step)
{
    const size_t block = 16;
    (void)block;
    size_t i = 0;

    for (; i + block <= rows; i += block) {
        size_t j = 0;
        for (; j + block <= cols; j += block) {
            rotate_8uc1_clockwise_16x16(src + i*src_step + j,
                    dst + j*dst_step + (rows-(i+block)),
                    src_step, dst_step);
        }
        for (; j < cols; ++j) {
            for (size_t k = 0; k < block; ++k) {
                dst[j*dst_step + (rows-1-(i+k))] = src[(i+k)*src_step + j];
            }
        }
    }

    for (; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[j*dst_step + (rows-1-i)] = src[i*src_step + j];
        }
    }
}

void rotate_8uc1_counterclockwise(const uchar *src, uchar *dst,
        const size_t rows, const size_t cols,
        const size_t src_step, const size_t dst_step)
{
    const size_t block = 16;
    (void)block;
    size_t i = 0;

    for (; i + block <= rows; i += block) {
        size_t j = 0;
        for (; j + block <= cols; j += block) {
            rotate_8uc1_counterclockwise_16x16(src + i*src_step + j,
                    dst + (cols-(j+block))*dst_step + i,
                    src_step, dst_step);
        }
        for (; j < cols; ++j) {
            for (size_t k = 0; k < block; ++k) {
                dst[(cols-1-j)*dst_step + (i+k)] = src[(i+k)*src_step + j];
            }
        }
    }

    for (; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[(cols-1-j)*dst_step + i] = src[i*src_step + j];
        }
    }
}

void rotate(const Mat<uchar> &src, Mat<uchar> &dst,
        bool clockwise)
{
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

namespace armv7 {

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

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
