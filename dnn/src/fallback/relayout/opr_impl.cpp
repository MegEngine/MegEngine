#include "src/fallback/relayout/opr_impl.h"
#include "src/common/relayout_helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

using namespace megdnn;
using namespace fallback;

namespace megdnn {
namespace relayout {
namespace transpose_fallback {
template <>
struct transpose_traits<dt_qint4> {
    static constexpr size_t block_size = BLOCK_LINE_SIZE_BYTES;
};

template <>
void transpose_block_fallback<dt_qint4>(
        const dt_qint4* src, dt_qint4* dst, const size_t src_stride,
        const size_t dst_stride, size_t block_h, size_t block_w) {
    constexpr size_t block_size = transpose_traits<dt_qint4>::block_size;
    uint8_t block[block_size][block_size];
    uint8_t* src_ptr = (uint8_t*)src;
    uint8_t* dst_ptr = (uint8_t*)dst;
    for (size_t i = 0; i < block_h; ++i) {
        size_t src_offset_base = i * src_stride;
        for (size_t j = 0; j < block_w; ++j) {
            size_t src_offset = src_offset_base + j;
            size_t src_byte_offset = src_offset >> 1;
            if (src_offset % 2 == 0) {
                block[j][i] = src_ptr[src_byte_offset] & 0xf;
            } else {
                block[j][i] = ((src_ptr[src_byte_offset] & 0xf0) >> 4) & 0xf;
            }
        }
    }
    for (size_t i = 0; i < block_w; ++i) {
        size_t dst_offset_base = i * dst_stride;
        for (size_t j = 0; j < block_h; ++j) {
            size_t dst_offset = dst_offset_base + j;
            size_t dst_byte_offset = dst_offset >> 1;
            uint8_t dst_temp = dst_ptr[dst_byte_offset];
            uint8_t src_temp = block[i][j];
            if (dst_offset % 2 == 0) {
                dst_temp = (dst_temp & 0xf0) | src_temp;
            } else {
                dst_temp = (dst_temp & 0xf) | (src_temp << 4);
            }
            dst_ptr[dst_byte_offset] = dst_temp;
        }
    }
}

template <>
void transpose<dt_qint4>(
        size_t batch, size_t m, size_t n, dt_qint4* src, dt_qint4* dst,
        size_t stride_m) {
    if (stride_m == 0) {
        stride_m = n;
    }
    uint8_t* batch_src = (uint8_t*)(src);
    uint8_t* batch_dst = (uint8_t*)(dst);
    constexpr size_t B = transpose_traits<dt_qint4>::block_size;

    auto work_block = [m, stride_m, &batch_src, &batch_dst](
                              const size_t i, const size_t j, const size_t h,
                              const size_t w) {
        size_t src_offset = i * stride_m + j;
        size_t dst_offset = j * m + i;
        megdnn_assert(src_offset % 2 == 0 && dst_offset % 2 == 0);
        auto src = batch_src + (src_offset >> 1);
        auto dst = batch_dst + (dst_offset >> 1);
        MIDOUT_BEGIN(transpose_fallback, midout_iv(0)) {
            if (h == B && w == B) {
                transpose_block((dt_qint4*)src, (dt_qint4*)dst, stride_m, m);
            } else {
                transpose_block((dt_qint4*)src, (dt_qint4*)dst, stride_m, m, h, w);
            }
        }
        MIDOUT_END();
    };
    auto work_row = [&work_block, n](size_t i, size_t h) {
        size_t j = 0;
        for (; j + B <= n; j += B) {
            work_block(i, j, h, B);
        }
        if (j < n) {
            work_block(i, j, h, n - j);
        }
    };

    for (size_t b = 0; b < batch; ++b) {
        size_t i = 0;
        for (; i + B <= m; i += B) {
            work_row(i, B);
        }
        if (i < m) {
            work_row(i, m - i);
        }
        size_t src_offset = m * stride_m;
        size_t dst_offset = m * n;
        megdnn_assert(src_offset % 2 == 0 && dst_offset % 2 == 0);
        batch_src += (src_offset >> 1);
        batch_dst += (dst_offset >> 1);
    }
}

}  // namespace transpose_fallback
}  // namespace relayout
}  // namespace megdnn

namespace {

bool is_lastdim_contig(const TensorLayout& layout) {
    return layout.ndim <= 3 && layout.stride[layout.ndim - 1] == 1;
}

bool is_int4(const TensorLayout& layout) {
    return layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
           layout.dtype.enumv() == DTypeEnum::Quantized4Asymm;
}

inline bool check_dtype_support_transparam(
        bool trans, bool is_bit4, const relayout::TransposeParam& param) {
    if (trans && is_bit4) {
        auto c = param.c;
        return c == 1 || c == 2 || c == 4 || c == 8;
    }
    return trans;
}

template <size_t sz, typename T0 = char>
struct equiv_ctype_storage {
    T0 _[sz];
};

template <typename dtype>
struct equiv_ctype {
    using type = std::aligned_storage_t<
            sizeof(typename DTypeTrait<dtype>::ctype),
            alignof(typename DTypeTrait<dtype>::ctype)>;
};

typedef void (*memcpy_policy_t)(
        void* cont, void* non_cont, size_t src_offset, size_t dst_offset, size_t size);

void memcpy_cont2noncont(void* cont, void* non_cont, size_t, size_t, size_t size) {
    memcpy(non_cont, cont, size);
}

void memcpy_noncont2cont(void* cont, void* non_cont, size_t, size_t, size_t size) {
    memcpy(cont, non_cont, size);
}

void memcpy_4bit(
        void* cont, void* nocont, size_t cont_offset, size_t nocont_offset,
        size_t size) {
    if (size == 0)
        return;
    uint8_t* cont_u8 = (uint8_t*)cont;
    uint8_t* nocont_u8 = (uint8_t*)nocont;
    size_t cont_bytes = cont_offset >> 1;
    size_t nocont_bytes = nocont_offset >> 1;
    size_t size_byte = size >> 1;
    void* cont_ptr = cont_u8 + cont_bytes;
    void* nocont_ptr = nocont_u8 + nocont_bytes;
    bool size_align = size % 2 == 0;
    bool cont_align = cont_offset % 2 == 0;
    bool nocont_align = nocont_offset % 2 == 0;
    if (cont_align && nocont_align) {
        memcpy(cont_ptr, nocont_ptr, size_byte);
        if (!size_align) {
            uint8_t* dst_ptr = (uint8_t*)cont_ptr + size_byte;
            uint8_t* src_ptr = (uint8_t*)nocont_ptr + size_byte;
            *dst_ptr = (*src_ptr) & 0xf;
        }
    } else if (!cont_align && nocont_align) {
        uint8_t* dst_ptr = (uint8_t*)cont_ptr;
        uint8_t* src_ptr = (uint8_t*)nocont_ptr;
        for (size_t i = 0; i < size_byte; ++i) {
            uint8_t dst_low = *dst_ptr;
            uint8_t src_all = *src_ptr;
            uint8_t last = (dst_low & 0xf) | (src_all & 0xf) << 4;
            uint8_t now = ((src_all & 0xf0) >> 4) & 0xf;
            *dst_ptr = last;
            ++dst_ptr;
            *dst_ptr = now;
            ++src_ptr;
        }
        if (!size_align) {
            uint8_t dst_low = *dst_ptr;
            uint8_t src_all = *src_ptr;
            uint8_t last = (dst_low & 0xf) | (src_all & 0xf) << 4;
            *dst_ptr = last;
        }
    } else if (cont_align && !nocont_align) {
        uint8_t* dst_ptr = (uint8_t*)cont_ptr;
        uint8_t* src_ptr = (uint8_t*)nocont_ptr;
        for (size_t i = 0; i < size_byte; ++i) {
            uint8_t src_last_high = *src_ptr;
            ++src_ptr;
            uint8_t src_low = *src_ptr;
            uint8_t rst = (src_low & 0xf) << 4 | ((src_last_high >> 4) & 0xf);
            *dst_ptr = rst;
            ++dst_ptr;
        }
        if (!size_align) {
            uint8_t src_last_high = *src_ptr;
            *dst_ptr = ((src_last_high >> 4) & 0xf);
        }
    } else {
        uint8_t* dst_ptr = (uint8_t*)cont_ptr;
        uint8_t* src_ptr = (uint8_t*)nocont_ptr;
        {
            uint8_t src_last_high = *src_ptr;
            uint8_t dst_last_low = *dst_ptr;
            uint8_t rst = (dst_last_low & 0xf) | (src_last_high & 0xf0);
            *dst_ptr = rst;
            ++dst_ptr;
            ++src_ptr;
        }
        if (!size_align) {
            memcpy(dst_ptr, src_ptr, size_byte);
        } else {
            if (size_byte > 1) {
                size_t align_size = size_byte - 1;
                memcpy(dst_ptr, src_ptr, align_size);
                dst_ptr += align_size;
                src_ptr += align_size;
            }
            uint8_t src_last_low = *src_ptr;
            *dst_ptr = src_last_low & 0xf;
        }
    }
}

void memcpy_cont2noncont_4bit(
        void* cont, void* non_cont, size_t cont_offset, size_t nocont_offset,
        size_t size) {
    memcpy_4bit(non_cont, cont, nocont_offset, cont_offset, size);
}

void memcpy_noncont2cont_4bit(
        void* cont, void* non_cont, size_t cont_offset, size_t nocont_offset,
        size_t size) {
    memcpy_4bit(cont, non_cont, cont_offset, nocont_offset, size);
}

template <typename T>
void call_transpose(
        size_t batch, size_t m, size_t n, size_t ch, void* src, void* dst,
        size_t stride_m) {
    megdnn_assert(ch == 1);
    relayout::transpose_fallback::transpose<T>(
            batch, m, n, static_cast<T*>(src), static_cast<T*>(dst), stride_m);
}

//! one operand contiguous, and the other non-contiguous
template <int bits>
void dispatch_on_dtype_cont(
        Handle* handle, const TensorND& cont, const TensorND& nonc,
        memcpy_policy_t mcp_pol) {
    thin_function<void()> kern;
    switch (nonc.layout.ndim) {
        case 2: {
            auto shp0 = nonc.layout.shape[0], shp1 = nonc.layout.shape[1];
            auto strd0_n = nonc.layout.stride[0] * bits / 8;
            auto strd0_c = shp1 * bits / 8;
            kern = [=]() {
                auto cur_ctptr = static_cast<uint8_t*>(cont.raw_ptr());
                auto cur_ncptr = static_cast<uint8_t*>(nonc.raw_ptr());
                for (size_t i = 0; i < shp0; ++i) {
                    mcp_pol(cur_ctptr, cur_ncptr, 0, 0, strd0_c);
                    cur_ctptr += strd0_c;
                    cur_ncptr += strd0_n;
                }
            };
            break;
        }
        case 3: {
            auto shp0 = nonc.layout.shape[0], shp1 = nonc.layout.shape[1],
                 shp2 = nonc.layout.shape[2];
            auto strd0_n = nonc.layout.stride[0] * bits / 8,
                 strd1_n = nonc.layout.stride[1] * bits / 8;
            auto strd1_c = shp2 * bits / 8;
            kern = [=]() {
                auto cur_ctptr = static_cast<uint8_t*>(cont.raw_ptr());
                auto ncptr_row = static_cast<uint8_t*>(nonc.raw_ptr());
                for (size_t i = 0; i < shp0; ++i) {
                    auto cur_ncptr = ncptr_row;
                    for (size_t j = 0; j < shp1; ++j) {
                        mcp_pol(cur_ctptr, cur_ncptr, 0, 0, strd1_c);
                        cur_ctptr += strd1_c;
                        cur_ncptr += strd1_n;
                    }
                    ncptr_row += strd0_n;
                }
            };
            break;
        }
        default:
            megdnn_assert(0);
    }

    static_cast<naive::HandleImpl*>(handle)->dispatch_kern(std::move(kern));
}

template <>
void dispatch_on_dtype_cont<4>(
        Handle* handle, const TensorND& cont, const TensorND& nonc,
        memcpy_policy_t mcp_pol) {
    thin_function<void()> kern;
    switch (nonc.layout.ndim) {
        case 2: {
            auto shp0 = nonc.layout.shape[0], shp1 = nonc.layout.shape[1];
            auto strd0_n = nonc.layout.stride[0];
            auto strd0_c = shp1;
            kern = [=]() {
                auto cur_ctptr = static_cast<uint8_t*>(cont.raw_ptr());
                auto cur_ncptr = static_cast<uint8_t*>(nonc.raw_ptr());
                size_t c_cnt = 0;
                size_t n_cnt = 0;
                for (size_t i = 0; i < shp0; ++i) {
                    mcp_pol(cur_ctptr, cur_ncptr, c_cnt, n_cnt, strd0_c);
                    c_cnt += strd0_c;
                    n_cnt += strd0_n;
                }
            };
            break;
        }
        case 3: {
            auto shp0 = nonc.layout.shape[0], shp1 = nonc.layout.shape[1],
                 shp2 = nonc.layout.shape[2];
            auto strd0_n = nonc.layout.stride[0], strd1_n = nonc.layout.stride[1];
            auto strd1_c = shp2;
            kern = [=]() {
                auto cur_ctptr = static_cast<uint8_t*>(cont.raw_ptr());
                auto ncptr_row = static_cast<uint8_t*>(nonc.raw_ptr());
                size_t c_cnt = 0;
                size_t n_cnt = 0;
                for (size_t i = 0; i < shp0; ++i) {
                    n_cnt = i * strd0_n;
                    for (size_t j = 0; j < shp1; ++j) {
                        mcp_pol(cur_ctptr, ncptr_row, c_cnt, n_cnt, strd1_c);
                        c_cnt += strd1_c;
                        n_cnt += strd1_n;
                    }
                }
            };
            break;
        }
        default:
            megdnn_assert(0);
    }

    static_cast<naive::HandleImpl*>(handle)->dispatch_kern(std::move(kern));
}

void dispatch_cont(
        Handle* handle, const TensorND& cont, const TensorND& nonc,
        memcpy_policy_t mcp_pol) {
    switch (cont.layout.dtype.enumv()) {
#define cb(_dt)                                                      \
    case DTypeTrait<dtype::_dt>::enumv:                              \
        return dispatch_on_dtype_cont<DTypeTrait<dtype::_dt>::bits>( \
                handle, cont, nonc, mcp_pol);
        MEGDNN_FOREACH_DTYPE_NAME(cb)
        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
        megdnn_assert(0);
    }
}

const size_t BLOCK_SIZE = 16;
const size_t TRANSPOSE_CV_MAX_C = relayout::transpose_fallback::BLOCK_LINE_SIZE_BYTES;

/*!
 * \tparam ctype The type of the data
 */
template <typename ctype>
void transpose_cv_block(
        size_t m, size_t n, size_t ch, size_t i, size_t j, size_t h, size_t w,
        void* src, void* dst) {
    auto batch_src = static_cast<const ctype*>(src);
    auto batch_dst = static_cast<ctype*>(dst);

#define SET_VAL(dst, src)                      \
    switch (ch) {                              \
        case 3:                                \
            dst[2] = src[2];                   \
            MEGDNN_FALLTHRU                    \
        case 2:                                \
            dst[1] = src[1];                   \
            MEGDNN_FALLTHRU                    \
        case 1:                                \
            dst[0] = src[0];                   \
            break;                             \
        default:                               \
            for (size_t _c = 0; _c < ch; ++_c) \
                dst[_c] = src[_c];             \
            break;                             \
    }

    constexpr size_t B = BLOCK_SIZE;
    static_assert(TRANSPOSE_CV_MAX_C % sizeof(ctype) == 0, "bad ctype");
    ctype tmp[B][B][TRANSPOSE_CV_MAX_C / sizeof(ctype)];
    auto sptr = batch_src + i * n * ch + j * ch;
    for (size_t x = 0; x < h; ++x) {
        for (size_t y = 0; y < w; ++y) {
            SET_VAL(tmp[y][x], (sptr + y * ch))
        }
        sptr += n * ch;
    }

    auto dptr = batch_dst + j * m * ch + i * ch;
    for (size_t x = 0; x < w; ++x) {
        for (size_t y = 0; y < h; ++y) {
            SET_VAL((dptr + y * ch), tmp[x][y])
        }
        dptr += m * ch;
    }
#undef SET_VAL
}

template <typename ctype>
void transpose_cv_row(
        size_t m, size_t n, size_t ch, size_t i, size_t h, void* src, void* dst) {
    constexpr size_t B = BLOCK_SIZE;
    size_t j = 0;
    for (; j + B <= n; j += B) {
        transpose_cv_block<ctype>(m, n, ch, i, j, h, B, src, dst);
    }
    if (j < n) {
        transpose_cv_block<ctype>(m, n, ch, i, j, h, n - j, src, dst);
    }
}

template <typename ctype>
void transpose_cv(
        size_t batch, size_t m, size_t n, size_t ch, void* src, void* dst,
        size_t stride_m) {
    megdnn_assert(stride_m == 0);
    constexpr size_t B = BLOCK_SIZE;
    auto batch_src = static_cast<ctype*>(src);
    auto batch_dst = static_cast<ctype*>(dst);
    for (size_t b = 0; b < batch; ++b) {
        size_t i = 0;
        for (; i + B <= m; i += B) {
            transpose_cv_row<ctype>(m, n, ch, i, B, batch_src, batch_dst);
        }
        if (i < m) {
            transpose_cv_row<ctype>(m, n, ch, i, m - i, batch_src, batch_dst);
        }
        batch_src += m * n * ch;
        batch_dst += m * n * ch;
    }
}

}  // anonymous namespace

void RelayoutForwardImpl::exec(
        _megdnn_tensor_in src0, _megdnn_tensor_out dst0, Handle* src_handle) {
    check_cpu_handle(src_handle);
    TensorND src = src0, dst = dst0;
    check_layout_and_canonize(src.layout, dst.layout);

    bool has_neg_stride = false;
    for (size_t i = 0; i < src.layout.ndim; ++i) {
        if (src.layout.stride[i] < 0) {
            has_neg_stride = true;
            break;
        }
    }
    for (size_t i = 0; i < dst.layout.ndim; ++i) {
        if (dst.layout.stride[i] < 0) {
            has_neg_stride = true;
            break;
        }
    }
    if (has_neg_stride) {
        NaiveRelayoutForwardImpl::do_exec(src, dst);
        return;
    }

    bool is_bit4 = is_int4(src.layout);
    bool allow_nocontig = !is_bit4;
    relayout::TransposeParam trans_param;
    bool trans =
            relayout::is_transpose(src.layout, dst.layout, trans_param, allow_nocontig);
    trans = check_dtype_support_transparam(trans, is_bit4, trans_param);
    exec_after_preprocess(src, dst, trans ? &trans_param : nullptr);
}

void RelayoutForwardImpl::exec_after_preprocess(
        const TensorND& src, const TensorND& dst, relayout::TransposeParam* transpose) {
    if (transpose) {
        bool is_bit4 = is_int4(src.layout);
        auto kernel = [tparam = *transpose, src, dst, is_bit4]() {
            auto t = tparam;
            void (*kptr)(size_t, size_t, size_t, size_t, void*, void*, size_t) =
                    nullptr;
            auto src_addr = reinterpret_cast<uintptr_t>(src.raw_ptr()),
                 dst_addr = reinterpret_cast<uintptr_t>(dst.raw_ptr());
            size_t dsize = 0;
            if (is_bit4) {
                dsize = t.c >> 1;
            } else {
                dsize = src.layout.dtype.size() * t.c;
            }
            if (is_bit4 && dsize == 0) {
                kptr = call_transpose<dt_qint4>;
            } else {
                if (dsize == 1) {
                    megdnn_assert(t.c == 1);
                    kptr = call_transpose<uint8_t>;
                } else if (dsize == 2) {
                    t.c = 1;
                    if (!((src_addr | dst_addr) & (alignof(uint16_t) - 1))) {
                        kptr = call_transpose<uint16_t>;
                    } else {
                        kptr = call_transpose<equiv_ctype_storage<2>>;
                        megdnn_log_error("unaligned addr in relayout");
                    }
                } else if (dsize == 3) {
                    t.c = 1;
                    kptr = call_transpose<equiv_ctype_storage<3>>;
                } else if (dsize == 4) {
                    t.c = 1;
                    if (!((src_addr | dst_addr) & (alignof(uint32_t) - 1))) {
                        kptr = call_transpose<uint32_t>;
                    } else {
                        kptr = call_transpose<equiv_ctype_storage<4>>;
                        megdnn_log_error("unaligned addr in relayout");
                    }
                } else if (dsize == 12) {
                    t.c = 1;
                    if (!((src_addr | dst_addr) & (alignof(uint32_t) - 1))) {
                        kptr = call_transpose<equiv_ctype_storage<3, uint32_t>>;
                    } else {
                        kptr = call_transpose<equiv_ctype_storage<12>>;
                        megdnn_log_error("unaligned addr in relayout");
                    }
                } else if (dsize <= TRANSPOSE_CV_MAX_C) {
                    switch (dst.layout.dtype.enumv()) {
#define cb(_dt)                                             \
    case DTypeTrait<dtype::_dt>::enumv:                     \
        kptr = transpose_cv<equiv_ctype<dtype::_dt>::type>; \
        break;
                        MEGDNN_FOREACH_DTYPE_NAME(cb)
                        MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)
#undef cb
                    }
                    megdnn_assert(kptr);
                }
            }

            if (kptr) {
                auto sptr = src.raw_ptr();
                auto dptr = dst.raw_ptr();
                kptr(t.batch, t.m, t.n, t.c, sptr, dptr, t.stride_m);
                return;
            } else {
                megdnn_assert(t.c != 1, "unsupported dtype size");
            }
        };
        MEGDNN_DISPATCH_CPU_KERN_OPR(kernel());
    }

    using relayout::is_contig;

    if (is_contig(dst.layout) && is_contig(src.layout)) {
        auto sz = src.layout.span().dist_byte();
        MEGDNN_DISPATCH_CPU_KERN_OPR(memcpy(dst.raw_ptr(), src.raw_ptr(), sz));
        return;
    }
    memcpy_policy_t cpy_noncont2cont = memcpy_noncont2cont;
    memcpy_policy_t cpy_cont2noncont = memcpy_cont2noncont;
    bool is_bit4 = src.layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
                   src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm;
    if (is_bit4) {
        cpy_noncont2cont = memcpy_noncont2cont_4bit;
        cpy_cont2noncont = memcpy_cont2noncont_4bit;
    }
    if (is_contig(dst.layout) && is_lastdim_contig(src.layout)) {
        return dispatch_cont(handle(), dst, src, cpy_noncont2cont);
    }

    if (is_contig(src.layout) && is_lastdim_contig(dst.layout)) {
        return dispatch_cont(handle(), src, dst, cpy_cont2noncont);
    }
    NaiveRelayoutForwardImpl::do_exec(src, dst);
}

// vim: syntax=cpp.doxygen
