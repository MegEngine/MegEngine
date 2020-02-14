/**
 * \file dnn/src/x86/convolution/convolution_direct_special_cases.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cstddef>
#include "megdnn/arch.h"

namespace megdnn {
namespace x86 {
namespace detail {

void convolution_xcorr_fh1_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh2_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh3_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh4_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh5_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh6_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh7_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_xcorr_fh1_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh2_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh3_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh4_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh5_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh6_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh7_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_xcorr_fh1_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_xcorr_fh2_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_xcorr_fh3_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_xcorr_fh4_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_xcorr_fh5_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_xcorr_fh6_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_xcorr_fh7_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh1_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh2_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh3_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh4_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh5_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh6_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh7_sse(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("sse");

void convolution_conv_fh1_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh2_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh3_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh4_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh5_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh6_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh7_avx(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("avx");

void convolution_conv_fh1_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh2_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh3_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh4_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh5_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh6_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");

void convolution_conv_fh7_fma(const float *src, const float *filter, float *dst,
        const size_t src_h, const size_t src_w, const size_t dst_h, const size_t dst_w,
        const size_t flt_w) MEGDNN_ATTRIBUTE_TARGET("fma");
} // namespace detail
} // namespace x86
} // namespace megdnn
