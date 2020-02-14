/**
 * \file dnn/src/common/roi_pooling_helper.h
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

namespace megdnn {
namespace roi_pooling {

template <typename T> struct MaxPooler {
    T maxval;
    int maxidx;
    size_t cnt;
    MEGDNN_HOST MEGDNN_DEVICE MaxPooler():
        maxval(DTypeTrait<T>::min()),
        maxidx(-1),
        cnt(0)
    {}
    MEGDNN_HOST MEGDNN_DEVICE void feed(T val, int idx)
    {
        ++cnt;
        if (val > maxval) {
            maxval = val;
            maxidx = idx;
        }
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_val(T &val)
    {
        val = cnt > 0 ? maxval : 0;
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_idx(int &idx)
    {
        idx = maxidx;
    }
};
template <typename T> struct AveragePooler {
    T sum;
    size_t cnt;
    MEGDNN_HOST MEGDNN_DEVICE AveragePooler():
        sum(T(0)), cnt(0)
    {}
    MEGDNN_HOST MEGDNN_DEVICE void feed(T val, int)
    {
        sum += val;
        ++cnt;
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_val(T &val)
    {
        val = cnt > 0 ? sum / T(cnt) : 0;
    }
    MEGDNN_HOST MEGDNN_DEVICE void writeback_idx(int &)
    {
    }
};

template <typename T> struct BwdMaxPooler {
    MEGDNN_HOST MEGDNN_DEVICE void update(
            int ph, int pw, int h, int w,
            float /* bin_size_h */, float /* bin_size_w */,
            int /* roi_start_h */, int /* roi_start_w */,
            size_t /* pooled_height */, size_t pooled_width,
            size_t /* height */, size_t width,
            const T *offset_src_diff,
            const int *offset_fp_idx,
            T &gradient)
    {
        if (offset_fp_idx[ph * pooled_width + pw] ==
                (int)(h * width + w)) {
            gradient += offset_src_diff[ph  * pooled_width + pw];
        }
    }
};

template <typename T> struct BwdAveragePooler
{
    MEGDNN_HOST MEGDNN_DEVICE void update(
            int ph, int pw, int h, int w, float bin_size_h, float bin_size_w,
            int roi_start_h, int roi_start_w,
            size_t /* pooled_height */, size_t pooled_width,
            size_t height, size_t width,
            const T *offset_src_diff,
            const int * /* offset_fp_idx */,
            T &gradient)
    {
#if MEGDNN_CC_HOST
        using std::min;
        using std::max;
#endif
        int hstart = static_cast<int>(floor(static_cast<float>(ph)
                    * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)
                    * bin_size_w));
        int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                    * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                    * bin_size_w));
        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart + roi_start_h, 0), (int)height);
        hend = min(max(hend + roi_start_h, 0), (int)height);
        wstart = min(max(wstart + roi_start_w, 0), (int)width);
        wend = min(max(wend + roi_start_w, 0), (int)width);
        int size = (hend - hstart) * (wend - wstart);
        float inv_size = 1.0f / size;
        if (h >= hstart && h < hend && w >= wstart && w < wend) {
            gradient += offset_src_diff[ph  * pooled_width + pw] * inv_size;
        }
    }
};

} // namespace roi_pooling
} // namespace megdnn

// vim: syntax=cpp.doxygen
