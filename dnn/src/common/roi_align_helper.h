/**
 * \file dnn/src/common/roi_align_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/dtype.h"

#if MEGDNN_CC_CUDA
#include "src/cuda/utils.cuh"
#endif

namespace megdnn {
namespace roi_align {

template <typename T>
MEGDNN_HOST MEGDNN_DEVICE T bilinear_interp(const T* data, const float h,
                                            const float w, const int height,
                                            const int width) {
    int h0 = floorf(h), w0 = floorf(w), h1 = h0 + 1, w1 = w0 + 1;
    T top_left = (h0 >= 0 && h0 < height && w0 >= 0 && w0 < width)
                         ? data[h0 * width + w0]
                         : T(0.f);
    T top_right = (h0 >= 0 && h0 < height && w1 >= 0 && w1 < width)
                          ? data[h0 * width + w1]
                          : T(0.f);
    T bottom_left = (h1 >= 0 && h1 < height && w0 >= 0 && w0 < width)
                            ? data[h1 * width + w0]
                            : T(0.f);
    T bottom_right = (h1 >= 0 && h1 < height && w1 >= 0 && w1 < width)
                             ? data[h1 * width + w1]
                             : T(0.f);
    T top = top_left + (top_right - top_left) * static_cast<T>(w - w0);
    T bottom =
            bottom_left + (bottom_right - bottom_left) * static_cast<T>(w - w0);
    T res = top + (bottom - top) * static_cast<T>(h - h0);
    return res;
}

template <typename T>
MEGDNN_HOST MEGDNN_DEVICE void distribute_diff(T* diff, const T top_diff,
                                               const float h, const float w,
                                               const int height,
                                               const int width) {
#if MEGDNN_CC_CUDA
    using namespace ::megdnn::cuda;
#endif
    int h0 = floorf(h), w0 = floorf(w), h1 = h0 + 1, w1 = w0 + 1;
    if (h0 >= 0 && h0 < height) {
        if (w0 >= 0 && w0 < width) {
            T val = top_diff * static_cast<T>((h1 - h) * (w1 - w));
#if MEGDNN_CC_CUDA
            atomic_add(&diff[h0 * width + w0], val);
#else
            diff[h0 * width + w0] += val;
#endif
        }
        if (w1 >= 0 && w1 < width) {
            T val = top_diff * static_cast<T>((h1 - h) * (w - w0));
#if MEGDNN_CC_CUDA
            atomic_add(&diff[h0 * width + w1], val);
#else
            diff[h0 * width + w1] += val;
#endif
        }
    }
    if (h1 >= 0 && h1 < height) {
        if (w0 >= 0 && w0 < width) {
            T val = top_diff * static_cast<T>((h - h0) * (w1 - w));
#if MEGDNN_CC_CUDA
            atomic_add(&diff[h1 * width + w0], val);
#else
            diff[h1 * width + w0] += val;
#endif
        }
        if (w1 >= 0 && w1 < width) {
            T val = top_diff * static_cast<T>((h - h0) * (w - w0));
#if MEGDNN_CC_CUDA
            atomic_add(&diff[h1 * width + w1], val);
#else
            diff[h1 * width + w1] += val;
#endif
        }
    }
}

template <typename T>
struct MaxPooler {
    T maxval;
    int maxidx;
    size_t cnt;
    MEGDNN_HOST MEGDNN_DEVICE MaxPooler()
            : maxval(DTypeTrait<T>::min()), maxidx(-1), cnt(0) {}
    MEGDNN_HOST MEGDNN_DEVICE void feed(T val, int idx) {
        ++cnt;
        if (val > maxval) {
            maxval = val;
            maxidx = idx;
        }
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_val(T& val) {
        val = cnt > 0 ? maxval : 0;
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_idx(int& idx) { idx = maxidx; }
};

template <typename T>
struct AveragePooler {
    T sum;
    size_t cnt;
    MEGDNN_HOST MEGDNN_DEVICE AveragePooler() : sum(T(0)), cnt(0) {}
    MEGDNN_HOST MEGDNN_DEVICE void feed(T val, int) {
        sum += val;
        ++cnt;
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_val(T& val) {
        val = cnt > 0 ? sum / T(cnt) : 0;
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_idx(int&) {}
};

template <typename T>
struct BwdPooler {
    int ph, pw;
    int sample_height, sample_width;
    int height, width;
    float roi_start_h, roi_start_w, bin_size_h, bin_size_w;
    float sample_h_rate, sample_w_rate;
    MEGDNN_HOST MEGDNN_DEVICE BwdPooler(int ph, int pw, int sample_height,
                                        int sample_width, int height, int width,
                                        float roi_start_h, float roi_start_w,
                                        float bin_size_h, float bin_size_w)
            : ph{ph},
              pw{pw},
              sample_height{sample_height},
              sample_width{sample_width},
              height{height},
              width{width},
              roi_start_h{roi_start_h},
              roi_start_w{roi_start_w},
              bin_size_h{bin_size_h},
              bin_size_w{bin_size_w} {
        sample_h_rate = 1.0f / ((float)(sample_height));
        sample_w_rate = 1.0f / ((float)(sample_width));
    }
};

template <typename T>
struct BwdMaxPooler : public BwdPooler<T> {
    using Super = BwdPooler<T>;
    MEGDNN_HOST MEGDNN_DEVICE BwdMaxPooler(int ph, int pw, int sample_height,
                                           int sample_width, int height,
                                           int width, float roi_start_h,
                                           float roi_start_w, float bin_size_h,
                                           float bin_size_w)
            : BwdPooler<T>{ph,         pw,        sample_height, sample_width,
                           height,     width,     roi_start_h,   roi_start_w,
                           bin_size_h, bin_size_w} {}
    MEGDNN_HOST MEGDNN_DEVICE void update(int index, const T* diff,
                                          const int* argmax, T* grad) {
        int h_iter = argmax[index] / Super::sample_width;
        int w_iter = argmax[index] - Super::sample_width * h_iter;
        float hcenter =
                Super::roi_start_h +
                Super::bin_size_h *
                        (Super::ph + Super::sample_h_rate * (h_iter + 0.5f));
        float wcenter =
                Super::roi_start_w +
                Super::bin_size_w *
                        (Super::pw + Super::sample_w_rate * (w_iter + 0.5f));
        distribute_diff(grad, diff[index], hcenter, wcenter, Super::height,
                        Super::width);
    }
};

template <typename T>
struct BwdAveragePooler : public BwdPooler<T> {
    using Super = BwdPooler<T>;
    MEGDNN_HOST MEGDNN_DEVICE
    BwdAveragePooler(int ph, int pw, int sample_height, int sample_width,
                     int height, int width, float roi_start_h,
                     float roi_start_w, float bin_size_h, float bin_size_w)
            : BwdPooler<T>{ph,         pw,        sample_height, sample_width,
                           height,     width,     roi_start_h,   roi_start_w,
                           bin_size_h, bin_size_w} {}
    MEGDNN_HOST MEGDNN_DEVICE void update(int index, const T* diff,
                                          const int* /* argmax */, T* grad) {
        int cnt = Super::sample_height * Super::sample_width;
        for (int h_iter = 0; h_iter < Super::sample_height; ++h_iter) {
            for (int w_iter = 0; w_iter < Super::sample_width; ++w_iter) {
                float hcenter = Super::roi_start_h +
                                Super::bin_size_h *
                                        (Super::ph + Super::sample_h_rate *
                                                             (h_iter + 0.5f));
                float wcenter = Super::roi_start_w +
                                Super::bin_size_w *
                                        (Super::pw + Super::sample_w_rate *
                                                             (w_iter + 0.5f));
                T val = diff[index] / static_cast<T>(cnt);
                distribute_diff(grad, val, hcenter, wcenter, Super::height,
                                Super::width);
            }
        }
    }
};

}  // namespace roi_align
}  // namespace megdnn

// vim: syntax=cpp.doxygen
