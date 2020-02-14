/**
 * \file dnn/src/common/cv/cvt_color.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#define GENERATE_CVT_OPR_DECL(_opr) \
    template <typename T>           \
    void _opr(const megcv::Mat<T>& src, megcv::Mat<T>& dst)

#define GENERATE_CVT_OPR_DECL_FOREACH(_cb) \
    _cb(cvt_rgb2gray);                     \
    _cb(cvt_rgb2yuv);                      \
    _cb(cvt_yuv2rgb);                      \
    _cb(cvt_gray2rgb);                     \
    _cb(cvt_rgba2rgb);                     \
    _cb(cvt_rgba2bgr);                     \
    _cb(cvt_rgba2gray);                    \
    _cb(cvt_rgb2bgr);                      \
    _cb(cvt_bgr2gray);                     \
    _cb(cvt_bgr2rgb);                      \
    _cb(cvt_yuv2gray_nv21);                \
    _cb(cvt_yuv2rgb_nv21);                 \
    _cb(cvt_yuv2bgr_nv21);                 \
    _cb(cvt_yuv2gray_nv12);                \
    _cb(cvt_yuv2rgb_nv12);                 \
    _cb(cvt_yuv2bgr_nv12);                 \
    _cb(cvt_yuv2gray_yv12);                \
    _cb(cvt_yuv2rgb_yv12);                 \
    _cb(cvt_yuv2bgr_yv12);                 \
    _cb(cvt_yuv2gray_yu12);                \
    _cb(cvt_yuv2rgb_yu12);                 \
    _cb(cvt_yuv2bgr_yu12);

#define descale(x, n) (((x) + (1 << ((n)-1))) >> (n))

#define GENERATE_UNSUPPORT_CVT_OPR_FOR_FLOAT(_cb) \
    _cb(cvt_rgba2rgb, float) \
    _cb(cvt_rgba2bgr, float) \
    _cb(cvt_rgba2gray, float) \
    _cb(cvt_rgb2bgr, float) \
    _cb(cvt_bgr2gray, float) \
    _cb(cvt_bgr2rgb, float) \
    _cb(cvt_yuv2gray_nv21, float) \
    _cb(cvt_yuv2rgb_nv21, float) \
    _cb(cvt_yuv2bgr_nv21, float) \
    _cb(cvt_yuv2gray_nv12, float) \
    _cb(cvt_yuv2rgb_nv12, float) \
    _cb(cvt_yuv2bgr_nv12, float) \
    _cb(cvt_yuv2gray_yv12, float) \
    _cb(cvt_yuv2rgb_yv12, float) \
    _cb(cvt_yuv2bgr_yv12, float) \
    _cb(cvt_yuv2gray_yu12, float) \
    _cb(cvt_yuv2rgb_yu12, float) \
    _cb(cvt_yuv2bgr_yu12, float)

#define GENERATE_UNSUPPORT_CVT_OPR(_opr, _type)                      \
    template <>                                                      \
    void _opr<_type>(const megcv::Mat<_type>&, megcv::Mat<_type>&) { \
        MegCVException("There is not a cvt_opr " #_opr               \
                       " to deal with " #_type);                     \
    }

// vim: syntax=cpp.doxygen
