/**
 * \file dnn/src/arm_common/convolution/img2col_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <cstddef>
#include "src/common/utils.h"

namespace {

template <bool is_xcorr, typename dtype>
void img2col_stride(const dtype* __restrict src,
                    dtype* __restrict dst, const int OC, const int OH,
                    const int OW, const int IC, const int IH, const int IW,
                    const int FH, const int FW, const int SH, const int SW) {
    (void)OC;
    size_t i = 0;
    rep(ic, IC) {
        rep(fh, FH) {
            rep(fw, FW) {
                rep(oh, OH) {
                    rep(ow, OW) {
                        int fh2, fw2;
                        if (is_xcorr) {
                            fh2 = fh;
                            fw2 = fw;
                        } else {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
                        dst[i++] = src[ic * IH * IW + (oh * SH + fh2) * IW +
                                       (ow * SW + fw2)];
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col(const dtype* src, dtype* dst, size_t /* OC */, size_t OH,
             size_t OW, size_t IC, size_t IH, size_t IW, size_t FH, size_t FW) {
    size_t offset = (4 - OW % 4) % 4;
    size_t i = 0;
    rep(ic, IC) {
        rep(fh, FH) {
            rep(fw, FW) {
                rep(oh, OH) {
                    size_t ow = 0;
                    for (; ow < OW; ow += 4) {
                        size_t fh2, fw2;
                        if (is_xcorr) {
                            fh2 = fh;
                            fw2 = fw;
                        } else {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW +
                                       (ow + fw2) + 0];
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW +
                                       (ow + fw2) + 1];
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW +
                                       (ow + fw2) + 2];
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW +
                                       (ow + fw2) + 3];
                    }
                    i -= offset;
                }
            }
        }
    }
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen
