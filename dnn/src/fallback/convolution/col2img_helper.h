#include <cstddef>
#include "src/common/utils.h"

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

}  // anonymous namespace

// vim: syntax=cpp.doxygen
