#include "src/common/utils.h"

namespace {

template <bool is_xcorr, typename dtype>
void img2col_stride(
        const dtype* __restrict src, dtype* __restrict dst, const int OC, const int OH,
        const int OW, const int IC, const int IH, const int IW, const int FH,
        const int FW, const int SH, const int SW) {
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
                        dst[i++] =
                                src[ic * IH * IW + (oh * SH + fh2) * IW +
                                    (ow * SW + fw2)];
                    }
                }
            }
        }
    }
}

//! add for im2col matmul multithread
//
template <bool is_xcorr, typename dtype>
void img2col_stride_nchw4(
        const dtype* __restrict src, dtype* __restrict dst, const int OC, const int OH,
        const int OW, const int IC, const int IH, const int IW, const int FH,
        const int FW, const int SH, const int SW, const int cur_index,
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

    size_t newIC = IC / 4;
    size_t i = 0;
    if (sizeof(dtype) != 1) {
        if (same_line) {
            rep(ic, newIC) {
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
                            size_t index =
                                    4 * (ic * IH * IW + (start_h * SH + fh2) * IW +
                                         (w * SW + fw2));
                            dst[i++] = src[index];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }
                    }
                }
            }
        } else {
            rep(ic, newIC) {
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
                            size_t index =
                                    4 * (ic * IH * IW + (start_h * SH + fh2) * IW +
                                         (w * SW + fw2));
                            dst[i++] = src[index + 0];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                size_t index = 4 * (ic * IH * IW + (h * SH + fh2) * IW +
                                                    (ow * SW + fw2));
                                dst[i++] = src[index + 0];
                                dst[i++] = src[index + 1];
                                dst[i++] = src[index + 2];
                                dst[i++] = src[index + 3];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            size_t index = 4 * (ic * IH * IW + (end_h * SH + fh2) * IW +
                                                (w * SW + fw2));
                            dst[i++] = src[index + 0];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }
                    }
                }
            }
        }
    } else {
        uint32_t* output = nullptr;
        const uint32_t* uint32_src =
                static_cast<const uint32_t*>(static_cast<const void*>(src));
        output = static_cast<uint32_t*>(static_cast<void*>(dst));
        if (same_line) {
            rep(ic, newIC) {
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

                        size_t index =
                                (ic * IH * IW + (start_h * SH + fh2) * IW +
                                 (cur_remain_w * SW + fw2));
                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            output[i++] = uint32_src[index];
                            index += SW;
                        }
                    }
                }
            }
        } else {
            rep(ic, newIC) {
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

                        size_t index = ic * IH * IW + (start_h * SH + fh2) * IW +
                                       cur_remain_w * SW + fw2;
                        for (int w = cur_remain_w; w < OW; w++) {
                            output[i++] = uint32_src[index];
                            index += SW;
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            index = ic * IH * IW + (h * SH + fh2) * IW + fw2;
                            rep(ow, OW) {
                                output[i++] = uint32_src[index];
                                index += SW;
                            }
                        }

                        index = ic * IH * IW + (end_h * SH + fh2) * IW + fw2;
                        for (int w = 0; w < end_remain_w; w++) {
                            output[i++] = uint32_src[index];
                            index += SW;
                        }
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col_nchw4(
        const dtype* __restrict src, dtype* __restrict dst, const int OC, const int OH,
        const int OW, const int IC, const int IH, const int IW, const int FH,
        const int FW, const int SH, const int SW, const int cur_index,
        const int block_size) {
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    int start_h = cur_index / OW;
    int cur_remain_w = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_remain_w = (cur_index + block_size) % OW;
    bool same_line = false;
    if (start_h == end_h) {
        same_line = true;
    }
    size_t newIC = IC / 4;
    size_t i = 0;
    if (sizeof(dtype) != 1) {
        if (same_line) {
            rep(ic, newIC) {
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
                            size_t index = 4 * (ic * IH * IW + (start_h + fh2) * IW +
                                                (w + fw2));
                            dst[i++] = src[index];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }
                    }
                }
            }
        } else {
            rep(ic, newIC) {
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
                            size_t index = 4 * (ic * IH * IW + (start_h + fh2) * IW +
                                                (w + fw2));
                            dst[i++] = src[index];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                size_t index = 4 * (ic * IH * IW + (h + fh2) * IW +
                                                    (ow + fw2));
                                dst[i++] = src[index + 0];
                                dst[i++] = src[index + 1];
                                dst[i++] = src[index + 2];
                                dst[i++] = src[index + 3];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            size_t index =
                                    4 * (ic * IH * IW + (end_h + fh2) * IW + (w + fw2));
                            dst[i++] = src[index + 0];
                            dst[i++] = src[index + 1];
                            dst[i++] = src[index + 2];
                            dst[i++] = src[index + 3];
                        }
                    }
                }
            }
        }
    } else {
        uint32_t* output = nullptr;
        const uint32_t* uint32_src =
                static_cast<const uint32_t*>(static_cast<const void*>(src));
        output = static_cast<uint32_t*>(static_cast<void*>(dst));
        if (same_line) {
            rep(ic, newIC) {
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
                            size_t index =
                                    (ic * IH * IW + (start_h + fh2) * IW + (w + fw2));
                            output[i++] = uint32_src[index];
                        }
                    }
                }
            }
        } else {
            rep(ic, newIC) {
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
                            size_t index =
                                    ic * IH * IW + (start_h + fh2) * IW + (w + fw2);
                            output[i++] = uint32_src[index];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                size_t index =
                                        (ic * IH * IW + (h + fh2) * IW + (ow + fw2));
                                output[i++] = uint32_src[index];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            size_t index =
                                    (ic * IH * IW + (end_h + fh2) * IW + (w + fw2));
                            output[i++] = uint32_src[index];
                        }
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col_nchw8(
        const dtype* __restrict src, dtype* __restrict dst, const int OW, const int IC,
        const int IH, const int IW, const int FH, const int FW, const int cur_index,
        const int block_size) {
    int start_h = cur_index / OW;
    int cur_n_remain = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_n_remain = (cur_index + block_size) % OW;
    bool same_line = (start_h == end_h);

    int IC_div_8 = IC / 8;

    if (sizeof(dtype) == 2) {
        if (same_line) {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        //! TODO: Substitute GI for arm intrinsic when GI supports FP16
                        //! data type.
                        int src_idx = 8 * (ic * IH * IW + (start_h + fh2) * IW +
                                           cur_n_remain + fw2);
                        for (int w = cur_n_remain; w < end_n_remain; ++w) {
                            vst1q_f16(
                                    reinterpret_cast<__fp16*>(dst) + dst_idx,
                                    vld1q_f16(
                                            reinterpret_cast<const __fp16*>(src) +
                                            src_idx));
                            dst_idx += 8;
                            src_idx += 8;
                        }
#else
                        int src_idx = 2 * (ic * IH * IW + (start_h + fh2) * IW +
                                           cur_n_remain + fw2);
                        uint64_t* u64_src = reinterpret_cast<uint64_t*>(src);
                        uint64_t* u64_dst = reinterpret_cast<uint64_t*>(dst);
                        for (int w = cur_n_remain; w < end_n_remain; w++) {
                            u64_dst[dst_idx] = u64_src[src_idx];
                            u64_dst[dst_idx + 1] = u64_src[src_idx + 1];
                            dst_idx += 2;
                            src_idx += 2;
                        }
#endif
                    }
                }
            }
        } else {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        int src_idx = 8 * (ic * IH * IW + (fh2 + start_h) * IW + fw2 +
                                           cur_n_remain);
                        for (int w = cur_n_remain; w < OW; ++w) {
                            vst1q_f16(
                                    reinterpret_cast<__fp16*>(dst) + dst_idx,
                                    vld1q_f16(
                                            reinterpret_cast<const __fp16*>(src) +
                                            src_idx));
                            dst_idx += 8;
                            src_idx += 8;
                        }
                        src_idx = 8 * (ic * IH * IW + (fh2 + start_h + 1) * IW + fw2);
                        for (int h = start_h + 1; h < end_h; ++h) {
                            int _src_idx = src_idx;
                            rep(w, OW) {
                                vst1q_f16(
                                        reinterpret_cast<__fp16*>(dst) + dst_idx,
                                        vld1q_f16(
                                                reinterpret_cast<const __fp16*>(src) +
                                                _src_idx));
                                dst_idx += 8;
                                _src_idx += 8;
                            }
                            src_idx += IW * 8;
                        }
                        src_idx = 8 * (ic * IH * IW + (fh2 + end_h) * IW + fw2);
                        rep(w, end_n_remain) {
                            vst1q_f16(
                                    reinterpret_cast<__fp16*>(dst) + dst_idx,
                                    vld1q_f16(
                                            reinterpret_cast<const __fp16*>(src) +
                                            src_idx));
                            dst_idx += 8;
                            src_idx += 8;
                        }
#else
                        uint64_t* u64_src = reinterpret_cast<uint64_t*>(src);
                        uint64_t* u64_dst = reinterpret_cast<uint64_t*>(dst);
                        int src_idx = 2 * (ic * IH * IW + (fh2 + start_h) * IW + fw2 +
                                           cur_n_remain);
                        for (int w = cur_n_remain; w < OW; ++w) {
                            u64_dst[dst_idx] = u64_src[src_idx];
                            u64_dst[dst_idx + 1] = u64_src[src_idx + 1];
                            dst_idx += 2;
                            src_idx += 2;
                        }
                        src_idx = 2 * (ic * IH * IW + (fh2 + start_h + 1) * IW + fw2);
                        for (int h = start_h + 1; h < end_h; ++h) {
                            int _src_idx = src_idx;
                            rep(w, OW) {
                                u64_dst[dst_idx] = u64_src[_src_idx];
                                u64_dst[dst_idx + 1] = u64_src[_src_idx + 1];
                                dst_idx += 2;
                                _src_idx += 2;
                            }
                            src_idx += IW * 2;
                        }
                        src_idx = 2 * (ic * IH * IW + (fh2 + end_h) * IW + fw2);
                        rep(w, end_n_remain) {
                            u64_dst[dst_idx] = u64_src[src_idx];
                            u64_dst[dst_idx + 1] = u64_src[src_idx + 1];
                            dst_idx += 2;
                            src_idx += 2;
                        }
#endif
                    }
                }
            }
        }
    } else {
        if (same_line) {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
                        int src_idx = 8 * (ic * IH * IW + (start_h + fh2) * IW + fw2 +
                                           cur_n_remain);
                        for (int w = cur_n_remain; w < end_n_remain; ++w) {
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                        }
                    }
                }
            }
        } else {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }

                        int src_idx = 8 * (ic * IH * IW + (start_h + fh2) * IW + fw2 +
                                           cur_n_remain);
                        for (int w = cur_n_remain; w < OW; ++w) {
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                        }

                        src_idx = 8 * (ic * IH * IW + (start_h + 1 + fh2) * IW + fw2);
                        for (int h = start_h + 1; h < end_h; ++h) {
                            rep(w, OW) {
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                                dst[dst_idx++] = src[src_idx++];
                            }
                        }

                        src_idx = 8 * (ic * IH * IW + (end_h + fh2) * IW + fw2);
                        rep(w, end_n_remain) {
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                            dst[dst_idx++] = src[src_idx++];
                        }
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col_stride_nchw8(
        const dtype* __restrict src, dtype* __restrict dst, const int OW, const int IC,
        const int IH, const int IW, const int FH, const int FW, const int SH,
        const int SW, const int cur_index, const int block_size) {
    int start_h = cur_index / OW;
    int cur_n_remain = cur_index % OW;
    int end_h = (cur_index + block_size) / OW;
    int end_n_remain = (cur_index + block_size) % OW;
    bool same_line = (start_h == end_h);

    int IC_div_8 = IC / 8;

    if (sizeof(dtype) == 2) {
        if (same_line) {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        int src_idx = 8 * (ic * IH * IW + (start_h * SH + fh2) * IW +
                                           cur_n_remain * SW + fw2);
                        for (int w = cur_n_remain; w < end_n_remain; ++w) {
                            vst1q_f16(
                                    reinterpret_cast<__fp16*>(dst) + dst_idx,
                                    vld1q_f16(
                                            reinterpret_cast<const __fp16*>(src) +
                                            src_idx));
                            dst_idx += 8;
                            src_idx += 8 * SW;
                        }
#else
                        int src_idx = 2 * (ic * IH * IW + (start_h * SH + fh2) * IW +
                                           cur_n_remain * SW + fw2);
                        uint64_t* u64_src = reinterpret_cast<uint64_t*>(src);
                        uint64_t* u64_dst = reinterpret_cast<uint64_t*>(dst);
                        for (int w = cur_n_remain; w < end_n_remain; w++) {
                            u64_dst[dst_idx] = u64_src[src_idx];
                            u64_dst[dst_idx + 1] = u64_src[src_idx + 1];
                            dst_idx += 2;
                            src_idx += 2 * SW;
                        }
#endif
                    }
                }
            }
        } else {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                        int src_idx = 8 * (ic * IH * IW + (fh2 + start_h * SH) * IW +
                                           fw2 + cur_n_remain * SW);
                        for (int w = cur_n_remain; w < OW; ++w) {
                            vst1q_f16(
                                    reinterpret_cast<__fp16*>(dst) + dst_idx,
                                    vld1q_f16(
                                            reinterpret_cast<const __fp16*>(src) +
                                            src_idx));
                            dst_idx += 8;
                            src_idx += 8 * SW;
                        }
                        src_idx = 8 * (ic * IH * IW + (fh2 + (start_h + 1) * SH) * IW +
                                       fw2);
                        for (int h = start_h + 1; h < end_h; ++h) {
                            int _src_idx = src_idx;
                            rep(w, OW) {
                                vst1q_f16(
                                        reinterpret_cast<__fp16*>(dst) + dst_idx,
                                        vld1q_f16(
                                                reinterpret_cast<const __fp16*>(src) +
                                                _src_idx));
                                dst_idx += 8;
                                _src_idx += 8 * SW;
                            }
                            src_idx += IW * 8 * SH;
                        }
                        src_idx = 8 * (ic * IH * IW + (fh2 + end_h * SH) * IW + fw2);
                        rep(w, end_n_remain) {
                            vst1q_f16(
                                    reinterpret_cast<__fp16*>(dst) + dst_idx,
                                    vld1q_f16(
                                            reinterpret_cast<const __fp16*>(src) +
                                            src_idx));
                            dst_idx += 8;
                            src_idx += 8 * SW;
                        }
#else
                        uint64_t* u64_src = reinterpret_cast<uint64_t*>(src);
                        uint64_t* u64_dst = reinterpret_cast<uint64_t*>(dst);
                        int src_idx = 2 * (ic * IH * IW + (fh2 + start_h * SH) * IW +
                                           fw2 + cur_n_remain * SW);
                        for (int w = cur_n_remain; w < OW; ++w) {
                            u64_dst[dst_idx] = u64_src[src_idx];
                            u64_dst[dst_idx + 1] = u64_src[src_idx + 1];
                            dst_idx += 2;
                            src_idx += 2 * SW;
                        }
                        src_idx = 2 * (ic * IH * IW + (fh2 + (start_h + 1) * SH) * IW +
                                       fw2);
                        for (int h = start_h + 1; h < end_h; ++h) {
                            int _src_idx = src_idx;
                            rep(w, OW) {
                                u64_dst[dst_idx] = u64_src[_src_idx];
                                u64_dst[dst_idx + 1] = u64_src[_src_idx + 1];
                                dst_idx += 2;
                                _src_idx += 2 * SW;
                            }
                            src_idx += IW * 2 * SH;
                        }
                        src_idx = 2 * (ic * IH * IW + (fh2 + end_h * SH) * IW + fw2);
                        rep(w, end_n_remain) {
                            u64_dst[dst_idx] = u64_src[src_idx];
                            u64_dst[dst_idx + 1] = u64_src[src_idx + 1];
                            dst_idx += 2;
                            src_idx += 2 * SW;
                        }
#endif
                    }
                }
            }
        }
    } else {
        if (same_line) {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }
                        int src_idx = 8 * (ic * IH * IW + (start_h * SH + fh2) * IW +
                                           fw2 + cur_n_remain * SW);
                        for (int w = cur_n_remain; w < end_n_remain; ++w) {
                            dst[dst_idx++] = src[src_idx];
                            dst[dst_idx++] = src[src_idx + 1];
                            dst[dst_idx++] = src[src_idx + 2];
                            dst[dst_idx++] = src[src_idx + 3];
                            dst[dst_idx++] = src[src_idx + 4];
                            dst[dst_idx++] = src[src_idx + 5];
                            dst[dst_idx++] = src[src_idx + 6];
                            dst[dst_idx++] = src[src_idx + 7];
                            src_idx += 8 * SW;
                        }
                    }
                }
            }
        } else {
            int dst_idx = 0;
            rep(ic, IC_div_8) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        int fh2 = fh, fw2 = fw;
                        if (!is_xcorr) {
                            fh2 = FH - fh - 1;
                            fw2 = FW - fw - 1;
                        }

                        int src_idx = 8 * (ic * IH * IW + (start_h * SH + fh2) * IW +
                                           fw2 + cur_n_remain * SW);
                        for (int w = cur_n_remain; w < OW; ++w) {
                            dst[dst_idx++] = src[src_idx];
                            dst[dst_idx++] = src[src_idx + 1];
                            dst[dst_idx++] = src[src_idx + 2];
                            dst[dst_idx++] = src[src_idx + 3];
                            dst[dst_idx++] = src[src_idx + 4];
                            dst[dst_idx++] = src[src_idx + 5];
                            dst[dst_idx++] = src[src_idx + 6];
                            dst[dst_idx++] = src[src_idx + 7];
                            src_idx += 8 * SW;
                        }

                        src_idx = 8 * (ic * IH * IW + ((start_h + 1) * SH + fh2) * IW +
                                       fw2);
                        for (int h = start_h + 1; h < end_h; ++h) {
                            rep(w, OW) {
                                dst[dst_idx++] = src[src_idx];
                                dst[dst_idx++] = src[src_idx + 1];
                                dst[dst_idx++] = src[src_idx + 2];
                                dst[dst_idx++] = src[src_idx + 3];
                                dst[dst_idx++] = src[src_idx + 4];
                                dst[dst_idx++] = src[src_idx + 5];
                                dst[dst_idx++] = src[src_idx + 6];
                                dst[dst_idx++] = src[src_idx + 7];
                                src_idx += 8 * SW;
                            }
                        }

                        src_idx = 8 * (ic * IH * IW + (end_h * SH + fh2) * IW + fw2);
                        rep(w, end_n_remain) {
                            dst[dst_idx++] = src[src_idx];
                            dst[dst_idx++] = src[src_idx + 1];
                            dst[dst_idx++] = src[src_idx + 2];
                            dst[dst_idx++] = src[src_idx + 3];
                            dst[dst_idx++] = src[src_idx + 4];
                            dst[dst_idx++] = src[src_idx + 5];
                            dst[dst_idx++] = src[src_idx + 6];
                            dst[dst_idx++] = src[src_idx + 7];
                            src_idx += 8 * SW;
                        }
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col_stride(
        const dtype* __restrict src, dtype* __restrict dst, const int OC, const int OH,
        const int OW, const int IC, const int IH, const int IW, const int FH,
        const int FW, const int SH, const int SW, const int cur_index,
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
                            dst[i++] =
                                    src[ic * IH * IW + (h * SH + fh2) * IW +
                                        (ow * SW + fw2)];
                        }
                    }

                    for (int w = 0; w < end_remain_w; w++) {
                        dst[i++] =
                                src[ic * IH * IW + (end_h * SH + fh2) * IW +
                                    (w * SW + fw2)];
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col(
        const dtype* __restrict src, dtype* __restrict dst, const int OC, const int OH,
        const int OW, const int IC, const int IH, const int IW, const int FH,
        const int FW, const int cur_index, const int block_size) {
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    int64_t start_h = cur_index / OW;
    int64_t cur_remain_w = cur_index % OW;
    int64_t end_h = (cur_index + block_size) / OW;
    int64_t end_remain_w = (cur_index + block_size) % OW;

    bool same_line = false;
    if (start_h == end_h) {
        same_line = true;
    }
    int64_t i = 0;
    if (same_line) {
        rep(ic, IC) {
            rep(fh, FH) {
                rep(fw, FW) {
                    int64_t fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }
                    for (int64_t w = cur_remain_w; w < end_remain_w; w++) {
                        dst[i++] = src[ic * IH * IW + (start_h + fh2) * IW + (w + fw2)];
                    }
                }
            }
        }
    } else {
        rep(ic, IC) {
            rep(fh, FH) {
                rep(fw, FW) {
                    int64_t fh2, fw2;
                    if (is_xcorr) {
                        fh2 = fh;
                        fw2 = fw;
                    } else {
                        fh2 = FH - fh - 1;
                        fw2 = FW - fw - 1;
                    }
                    for (int64_t w = cur_remain_w; w < OW; w++) {
                        dst[i++] = src[ic * IH * IW + (start_h + fh2) * IW + (w + fw2)];
                    }

                    for (int64_t h = start_h + 1; h < end_h; h++) {
                        rep(ow, OW) {
                            dst[i++] = src[ic * IH * IW + (h + fh2) * IW + (ow + fw2)];
                        }
                    }

                    for (int64_t w = 0; w < end_remain_w; w++) {
                        dst[i++] = src[ic * IH * IW + (end_h + fh2) * IW + (w + fw2)];
                    }
                }
            }
        }
    }
}

template <bool is_xcorr, typename dtype>
void img2col(
        const dtype* src, dtype* dst, size_t /* OC */, size_t OH, size_t OW, size_t IC,
        size_t IH, size_t IW, size_t FH, size_t FW) {
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
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW + (ow + fw2) + 0];
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW + (ow + fw2) + 1];
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW + (ow + fw2) + 2];
                        dst[i++] = src[ic * IH * IW + (oh + fh2) * IW + (ow + fw2) + 3];
                    }
                    i -= offset;
                }
            }
        }
    }
}
}  // anonymous namespace

// vim: syntax=cpp.doxygen
