#include <cstddef>
#include "src/common/utils.h"
#include "src/fallback/general_intrinsic/gi_float.h"

namespace {

template <bool is_xcorr, typename dtype>
void col2img_stride_padding(
        const dtype* __restrict src, dtype* __restrict dst, const int OH, const int OW,
        const int IC, const int IH, const int IW, const int FH, const int FW,
        const int SH, const int SW, int PH, int PW) {
    size_t i = 0;
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
                rep(ih, IH) {
                    int h = ih * SH - PH + fh2;
                    rep(iw, IW) {
                        int w = iw * SW - PW + fw2;
                        if (h >= 0 && h < OH && w >= 0 && w < OW) {
                            dst[ic * OH * OW + h * OW + w] += src[i];
                        }
                        i++;
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void col2img(
        const dtype* __restrict src, dtype* __restrict dst, const int OH, const int OW,
        const int IC, const int IH, const int IW, const int FH, const int FW) {
    size_t i = 0;
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
                rep(ih, IH) {
                    rep(iw, IW) {
                        dst[ic * OH * OW + (ih + fh2) * OW + iw + fw2] += src[i++];
                    }
                }
            }
        }
    }
}

template <bool is_xcorr>
void col2img_stride_padding_nchw44(
        const float* __restrict src, float* __restrict dst, const int OH, const int OW,
        const int IC, const int IH, const int IW, const int FH, const int FW,
        const int SH, const int SW, int PH, int PW) {
    size_t i = 0;
    rep(ic, IC / 4) {
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
                rep(ih, IH) {
                    int h = ih * SH - PH + fh2;
                    rep(iw, IW) {
                        int w = iw * SW - PW + fw2;
                        if (h >= 0 && h < OH && w >= 0 && w < OW) {
                            float* dst_ptr = dst + (ic * OH * OW + h * OW + w) * 4;
                            GI_FLOAT32_t dst_data = GiLoadFloat32(dst_ptr);
                            GI_FLOAT32_t src_data = GiLoadFloat32(src + i);
                            GiStoreFloat32(dst_ptr, GiAddFloat32(dst_data, src_data));
                        }
                        i += 4;
                    }
                }
            }
        }
    }
}

template <bool is_xcorr>
void col2img_nchw44(
        const float* __restrict src, float* __restrict dst, const int OH, const int OW,
        const int IC, const int IH, const int IW, const int FH, const int FW) {
    size_t i = 0;
    rep(ic, IC / 4) {
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
                rep(ih, IH) {
                    rep(iw, IW) {
                        float* dst_ptr = dst + ic * OH * OW * 4 + (ih + fh2) * OW * 4 +
                                         iw * 4 + fw2 * 4;
                        GI_FLOAT32_t dst_data = GiLoadFloat32(dst_ptr);
                        GI_FLOAT32_t src_data = GiLoadFloat32(src + i);
                        GiStoreFloat32(dst_ptr, GiAddFloat32(dst_data, src_data));
                        i += 4;
                    }
                }
            }
        }
    }
}

}  // anonymous namespace

// vim: syntax=cpp.doxygen
