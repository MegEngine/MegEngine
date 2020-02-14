/**
 * \file dnn/src/fallback/convolution/img2col_helper.h
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
void img2col_stride(const dtype* __restrict src, dtype* __restrict dst,
                    const int OC, const int OH, const int OW, const int IC,
                    const int IH, const int IW, const int FH, const int FW,
                    const int SH, const int SW) {
    megdnn_ignore(OC);
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

//! add for im2col matmul multithread
template <bool is_xcorr, typename dtype>
void img2col_stride(const dtype* __restrict src, dtype* __restrict dst,
                    const int OC, const int OH, const int OW, const int IC,
                    const int IH, const int IW, const int FH, const int FW,
                    const int SH, const int SW, const int cur_index,
                    const int block_size) {
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    int start_h = cur_index / OW;
    int cur_remain_w = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_remain_w = (cur_index + block_size) % OW;

    bool same_line = false;
    if (start_h == end_h) {
        same_line = true;
    }

    size_t i = 0;
    if (same_line) {
        rep(ic, IC) {
            rep(fh, FH) {
                rep(fw, FW) {
                    int fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }

                    for (int w = cur_remain_w; w < end_remain_w; w++) {
                        dst[i++] =
                                src[ic * IH * IW + (start_h * SH + fh2) * IW +
                                    (w * SW + fw2)];
                    }
                }
            }
        }
    } else {
        rep(ic, IC) {
            rep(fh, FH) {
                rep(fw, FW) {
                    int fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }

                    for (int w = cur_remain_w; w < OW; w++) {
                        dst[i++] =
                                src[ic * IH * IW + (start_h * SH + fh2) * IW +
                                    (w * SW + fw2)];
                    }

                    for (int h = start_h + 1; h < end_h; h++) {
                        rep(ow, OW) {
                            dst[i++] = src[ic * IH * IW + (h * SH + fh2) * IW +
                                           (ow * SW + fw2)];
                        }
                    }

                    for (int w = 0; w < end_remain_w; w++) {
                        dst[i++] = src[ic * IH * IW + (end_h * SH + fh2) * IW +
                                       (w * SW + fw2)];
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col(const dtype* __restrict src, dtype* __restrict dst, const int OC,
             const int OH, const int OW, const int IC, const int IH,
             const int IW, const int FH, const int FW, const int cur_index,
             const int block_size) {
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    int start_h = cur_index / OW;
    int cur_remain_w = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_remain_w = (cur_index + block_size) % OW;

    bool same_line = false;
    if (start_h == end_h) {
        same_line = true;
    }
    int i = 0;
    if (same_line) {
        rep(ic, IC) {
            rep(fh, FH) {
                rep(fw, FW) {
                    int fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }
                    for (int w = cur_remain_w; w < end_remain_w; w++) {
                        dst[i++] = src[ic * IH * IW + (start_h + fh2) * IW +
                                       (w + fw2)];
                    }
                }
            }
        }
    } else {
        rep(ic, IC) {
            rep(fh, FH) {
                rep(fw, FW) {
                    int fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }
                    for (int w = cur_remain_w; w < OW; w++) {
                        dst[i++] = src[ic * IH * IW + (start_h + fh2) * IW +
                                       (w + fw2)];
                    }

                    for (int h = start_h + 1; h < end_h; h++) {
                        rep(ow, OW) {
                            dst[i++] = src[ic * IH * IW + (h + fh2) * IW +
                                           (ow + fw2)];
                        }
                    }

                    for (int w = 0; w < end_remain_w; w++) {
                        dst[i++] = src[ic * IH * IW + (end_h + fh2) * IW +
                                       (w + fw2)];
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
