#include "src/naive/relayout_format/opr_impl.h"
#include "src/naive/handle.h"

#include "megdnn/tensor_iter.h"

#include "midout.h"

MIDOUT_DECL(megdnn_naive_relayout_format)

using namespace megdnn;
using namespace naive;

namespace {

template <typename ctype>
void recursive_cp(
        const TensorND& dst, const TensorND& src, size_t idx = 0, size_t src_offset = 0,
        size_t dst_offset = 0) {
    if (idx < (src.layout.ndim - 1)) {
        for (size_t i = 0; i < src.layout[idx]; ++i) {
            recursive_cp<ctype>(
                    dst, src, idx + 1, src_offset + i * src.layout.stride[idx],
                    dst_offset + i * dst.layout.stride[idx]);
        }
    } else {
        auto src_ptr = src.ptr<ctype>();
        auto dst_ptr = dst.ptr<ctype>();
        for (size_t i = 0; i < src.layout[idx]; ++i) {
            dst_ptr[dst_offset + i * dst.layout.stride[idx]] =
                    src_ptr[src_offset + i * src.layout.stride[idx]];
        }
    }
}

template <size_t size_nbits>
void lowbit_recursive_cp(
        const TensorND& dst, const TensorND& src, size_t idx = 0, size_t src_offset = 0,
        size_t dst_offset = 0) {
    MEGDNN_STATIC_ASSERT(
            !(8_z % size_nbits),
            "size in bits of lowbit data type can only be 1, 2, 4 "
            "or 8")
    if (idx < (src.layout.ndim - 1)) {
        for (size_t i = 0; i < src.layout[idx]; ++i) {
            lowbit_recursive_cp<size_nbits>(
                    dst, src, idx + 1, src_offset + i * src.layout.stride[idx],
                    dst_offset + i * dst.layout.stride[idx]);
        }
    } else {
        megdnn_assert(src.layout.stride[idx] == 1);
        megdnn_assert(dst.layout.stride[idx] == 1);
        size_t dim_bytes = div_ceil(src.layout[idx], 8_z / size_nbits);
        // offset in elements
        uint8_t* dptr = reinterpret_cast<uint8_t*>(dst.raw_ptr()) +
                        (dst_offset * size_nbits / 8);
        uint8_t* sptr = reinterpret_cast<uint8_t*>(src.raw_ptr()) +
                        (src_offset * size_nbits / 8);
        for (size_t i = 0; i < dim_bytes; ++i) {
            *dptr = *sptr;
            dptr++;
            sptr++;
        }
    }
}

void padding_to_workspace(_megdnn_tensor_out dst, _megdnn_tensor_in src) {
    switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                \
    case (DTypeEnum::name): {          \
        recursive_cp<ctype>(dst, src); \
        break;                         \
    }

        cb(Float32, dt_float32);
        cb(Int32, dt_int32);
        cb(QuantizedS32, dt_int32);
        cb(QuantizedS8, dt_qint8);
#undef cb
#define cb(name, size_nbits)                       \
    case (DTypeEnum::name): {                      \
        lowbit_recursive_cp<size_nbits>(dst, src); \
        break;                                     \
    }
        cb(QuantizedS4, 4);
        cb(Quantized4Asymm, 4);
#undef cb
        default:
            megdnn_assert(0, "not support dtype %s", src.layout.dtype.name());
    }
}

void extract_from_workspace(
        _megdnn_tensor_out dst, _megdnn_tensor_in src, size_t group) {
    megdnn_assert(
            dst.layout.is_contiguous() && src.layout.is_contiguous(), "dst %s, src %s",
            dst.layout.to_string().c_str(), src.layout.to_string().c_str());
    const size_t n = dst.layout[0];
    const size_t n_stride_dst_in_bytes = dst.layout.dtype.size(dst.layout.stride[0]);
    const size_t n_stride_src_in_bytes = src.layout.dtype.size(src.layout.stride[0]);
    const size_t ocpg = dst.layout[1] / group;
    const size_t icpg = src.layout[1] / group;
    const size_t dst_c_stride_in_bytes = dst.layout.dtype.size(dst.layout.stride[1]);
    const size_t src_c_stride_in_bytes = src.layout.dtype.size(src.layout.stride[1]);
    megdnn_assert(dst_c_stride_in_bytes == src_c_stride_in_bytes);
    for (size_t nid = 0; nid < n; ++nid) {
        const size_t n_offset_dst = nid * n_stride_dst_in_bytes;
        const size_t n_offset_src = nid * n_stride_src_in_bytes;
        for (size_t gid = 0; gid < group; ++gid) {
            memcpy(reinterpret_cast<char*>(dst.raw_ptr()) + n_offset_dst +
                           gid * ocpg * dst_c_stride_in_bytes,
                   reinterpret_cast<char*>(src.raw_ptr()) + n_offset_src +
                           gid * icpg * src_c_stride_in_bytes,
                   ocpg * dst_c_stride_in_bytes);
        }
    }
};

template <typename dtype>
void padding_src_to_workspace(
        dtype* dptr, const dtype* sptr, size_t N, size_t IC, size_t IH, size_t IW) {
    size_t IC4 = (IC + 3) / 4 * 4;
    size_t HW = IH * IW;
    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < IC4; c++) {
            for (size_t idx = 0; idx < HW; idx++) {
                if (c < IC) {
                    *dptr = sptr[n * IC * HW + c * HW + idx];
                } else {
                    *dptr = 0;
                }
                dptr++;
            }
        }
    }
}

template <typename dtype>
void padding_nhwc_src_to_workspace(
        dtype* dptr, const dtype* sptr, size_t N, size_t IH, size_t IW, size_t IC) {
    size_t IC4 = (IC + 3) / 4 * 4;
    size_t HW = IH * IW;
    for (size_t n = 0; n < N; n++) {
        for (size_t idx = 0; idx < HW; idx++) {
            for (size_t c = 0; c < IC4; c++) {
                if (c < IC) {
                    *dptr = sptr[n * IC * HW + idx * IC + c];
                } else {
                    *dptr = 0;
                }
                dptr++;
            }
        }
    }
}

template <typename dtype>
void padding_to_workspace(
        dtype* dptr, const dtype* sptr, const TensorLayout& src_layout,
        const size_t pad_axis, const size_t align_size, const int pad_val = 0) {
    megdnn_assert(pad_axis < src_layout.ndim);
    const size_t axis_dim = src_layout[pad_axis];
    const size_t axis_dim_padded = round_up(axis_dim, align_size);
    const size_t axis_stride = src_layout.stride[pad_axis];
    const size_t repeat_number = src_layout.total_nr_elems() / (axis_dim * axis_stride);
    for (size_t outer_idx = 0; outer_idx < repeat_number; ++outer_idx) {
        const size_t dst_outer_offset = outer_idx * axis_dim_padded * axis_stride;
        const size_t src_inner_offset = outer_idx * axis_dim * axis_stride;
        for (size_t axis_idx = 0; axis_idx < axis_dim_padded; ++axis_idx) {
            for (size_t inner_idx = 0; inner_idx < axis_stride; ++inner_idx) {
                const size_t inner_idx_offset = axis_idx * axis_stride + inner_idx;
                if (axis_idx < axis_dim) {
                    dptr[dst_outer_offset + inner_idx_offset] =
                            sptr[src_inner_offset + inner_idx_offset];
                } else {
                    dptr[dst_outer_offset + inner_idx_offset] =
                            static_cast<dtype>(pad_val);
                }
            }
        }
    }
}

void padding_to_workspace(
        _megdnn_tensor_out dst, _megdnn_tensor_in src, const size_t pad_axis,
        const size_t align_size, DType exec_dst_dtype) {
    switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                                            \
    case (DTypeEnum::name): {                                                      \
        ctype* sptr = src.compatible_ptr<ctype>();                                 \
        ctype* dptr = dst.compatible_ptr<ctype>();                                 \
        padding_to_workspace<ctype>(dptr, sptr, src.layout, pad_axis, align_size); \
        break;                                                                     \
    }

        cb(Float32, dt_float32);
        cb(Int32, dt_int32);
        cb(QuantizedS32, dt_int32);
        cb(QuantizedS8, dt_qint8);

        case (DTypeEnum::Quantized8Asymm): {
            dt_quint8* sptr = src.compatible_ptr<dt_quint8>();
            dt_quint8* dptr = dst.compatible_ptr<dt_quint8>();
            padding_to_workspace<dt_quint8>(
                    dptr, sptr, src.layout, pad_axis, align_size,
                    src.layout.dtype.param<dtype::Quantized8Asymm>().zero_point);
            break;
        }
        case (DTypeEnum::Uint8): {
            uint8_t* sptr = src.compatible_ptr<uint8_t>();
            uint8_t* dptr = dst.compatible_ptr<uint8_t>();
            uint8_t zero_point =
                    exec_dst_dtype.enumv() == DTypeEnum::QuantizedS8 ? 128 : 0;
            padding_to_workspace<uint8_t>(
                    dptr, sptr, src.layout, pad_axis, align_size, zero_point);
            break;
        }
        default:
            megdnn_assert(0, "not support dtype %s", src.layout.dtype.name());
#undef cb
    }
}

template <typename dtype>
void padding_filter_to_workspace(
        dtype* dptr, const dtype* sptr, size_t OC, size_t IC, size_t FH, size_t FW) {
    size_t IC4 = (IC + 3) / 4 * 4;
    size_t HW = FH * FW;
    for (size_t oc = 0; oc < OC; ++oc) {
        for (size_t ic = 0; ic < IC4; ++ic) {
            for (size_t hw = 0; hw < HW; ++hw) {
                if (ic < IC) {
                    *dptr = sptr[oc * IC * HW + ic * HW + hw];
                } else {
                    *dptr = 0;
                }
                dptr++;
            }
        }
    }
}

void do_copy_diff_qu8_q8(const TensorND& dst, const TensorND& src) {
    auto isrc =
            tensor_iter_valonly<DTypeTrait<dtype::Quantized8Asymm>::ctype>(src).begin();
    auto idst = tensor_iter_valonly<DTypeTrait<dtype::QuantizedS8>::ctype>(dst).begin();
    auto src_dt_parm = src.layout.dtype.param<dtype::Quantized8Asymm>();
    auto dst_dt_parm = dst.layout.dtype.param<dtype::QuantizedS8>();
    for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
        *idst = dst_dt_parm.quantize(src_dt_parm.dequantize(*isrc));
        ++idst;
        ++isrc;
    }
}

void do_copy_diff_q8_q8(const TensorND& dst, const TensorND& src) {
    auto isrc = tensor_iter_valonly<DTypeTrait<dtype::QuantizedS8>::ctype>(src).begin();
    auto idst = tensor_iter_valonly<DTypeTrait<dtype::QuantizedS8>::ctype>(dst).begin();
    auto src_dt_parm = src.layout.dtype.param<dtype::QuantizedS8>();
    auto dst_dt_parm = dst.layout.dtype.param<dtype::QuantizedS8>();
    for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
        *idst = dst_dt_parm.quantize(src_dt_parm.dequantize(*isrc));
        ++idst;
        ++isrc;
    }
}

void do_copy_diff_q32_q32(const TensorND& dst, const TensorND& src) {
    auto isrc =
            tensor_iter_valonly<DTypeTrait<dtype::QuantizedS32>::ctype>(src).begin();
    auto idst =
            tensor_iter_valonly<DTypeTrait<dtype::QuantizedS32>::ctype>(dst).begin();
    auto src_dt_parm = src.layout.dtype.param<dtype::QuantizedS32>();
    auto dst_dt_parm = dst.layout.dtype.param<dtype::QuantizedS32>();
    for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
        *idst = dst_dt_parm.quantize(src_dt_parm.dequantize(*isrc));
        ++idst;
        ++isrc;
    }
}

void do_copy_diff_u8_q8(const TensorND& dst, const TensorND& src) {
    auto isrc = tensor_iter_valonly<DTypeTrait<dtype::Uint8>::ctype>(src).begin();
    auto idst = tensor_iter_valonly<DTypeTrait<dtype::QuantizedS8>::ctype>(dst).begin();
    auto dst_dt_parm = dst.layout.dtype.param<dtype::QuantizedS8>();
    for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
        *idst = dst_dt_parm.quantize((float)(*isrc) - 128.f);
        ++idst;
        ++isrc;
    }
}

void do_copy_diff_q4_q4(const TensorND& dst, const TensorND& src) {
    auto isrc = tensor_iter_valonly<DTypeTrait<dtype::QuantizedS4>::ctype>(src).begin();
    auto idst = tensor_iter_valonly<DTypeTrait<dtype::QuantizedS4>::ctype>(dst).begin();
    auto src_dt_parm = src.layout.dtype.param<dtype::QuantizedS4>();
    auto dst_dt_parm = dst.layout.dtype.param<dtype::QuantizedS4>();
    for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
        *idst = dst_dt_parm.quantize(src_dt_parm.dequantize(int8_t(*isrc)));
        ++idst;
        ++isrc;
    }
}

void do_copy_diff_qu4_qu4(const TensorND& dst, const TensorND& src) {
    auto isrc =
            tensor_iter_valonly<DTypeTrait<dtype::Quantized4Asymm>::ctype>(src).begin();
    auto idst =
            tensor_iter_valonly<DTypeTrait<dtype::Quantized4Asymm>::ctype>(dst).begin();
    auto src_dt_parm = src.layout.dtype.param<dtype::Quantized4Asymm>();
    auto dst_dt_parm = dst.layout.dtype.param<dtype::Quantized4Asymm>();
    for (size_t i = 0, it = dst.layout.total_nr_elems(); i < it; ++i) {
        *idst = dst_dt_parm.quantize(src_dt_parm.dequantize(uint8_t(*isrc)));
        ++idst;
        ++isrc;
    }
}

void check_layout_and_canonize(TensorLayout& src, TensorLayout& dst) {
    megdnn_assert(dst.is_non_overlapping_strong());
    src = src.collapse_contiguous();
    dst = dst.collapse_contiguous();
    megdnn_assert(dst.dtype.valid() && src.total_nr_elems() == dst.total_nr_elems());
}

}  // anonymous namespace

size_t RelayoutFormatImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    using Param = param::RelayoutFormat;
    switch (param().mode) {
        case Param::Mode::NCHW_NHWCD4I: {
            if (src[1] % 4 == 0)
                return 0;
            size_t IC4 = dst[2] * 4;
            size_t N = src[0];
            size_t IH = src[2];
            size_t IW = src[3];
            return N * IC4 * IH * IW * src.dtype.size();
        }
        case Param::Mode::NHWC_NHWCD4I: {
            if (src[3] % 4 == 0)
                return 0;
            size_t IC4 = dst[2] * 4;
            size_t N = src[0];
            size_t IH = src[1];
            size_t IW = src[2];
            return N * IC4 * IH * IW * src.dtype.size();
        }
        case Param::Mode::INTER_WEIGHT_DENSEI_DOT: {
            if (src[1] % 4 == 0)
                return 0;
            size_t OC = src[0];
            size_t IC4 = dst[3] * 4;
            size_t FH = src[2];
            size_t FW = src[3];
            megdnn_assert(!(OC & 0x3));
            return OC * IC4 * FH * FW;
        }
        case Param::Mode::NCHW_NCHW88: {
            if (src[1] % 8 == 0)
                return 0;
            size_t n = src[0];
            size_t c = round_up(src[1], 8_z);
            size_t h = src[2];
            size_t w = src[3];
            return n * c * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT: {
            megdnn_assert(src.ndim == 4, "src must be oihw, ndim == 5");
            megdnn_assert(
                    src[0] % 8 == 0,
                    "NCHW_NCHW88_CONV_DENSE_WEIGHT oc must align to 8");
            if (src[1] % 8 == 0)
                return 0;
            size_t oc = src[0];
            size_t ic = round_up(src[1], 8_z);
            size_t h = src[2];
            size_t w = src[3];
            return oc * ic * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT: {
            megdnn_assert(src.ndim == 5, "src must be goihw, ndim == 5");
            megdnn_assert(
                    src[1] % 8 == 0,
                    "NCHW_NCHW88_CONV_CHAN_WEIGHT oc per group must "
                    "align to 8");
            if (src[2] % 8 == 0)
                return 0;
            size_t group = src[0];
            size_t ocpg = src[1];
            size_t icpg = round_up(src[2], 8_z);
            size_t h = src[3];
            size_t w = src[4];
            return group * ocpg * icpg * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT: {
            megdnn_assert(src.ndim == 5, "src must be goihw, ndim == 5");
            if (src[0] % 8 == 0)
                return 0;
            size_t group = round_up(src[0], 8_z);
            size_t ocpg = src[1];
            size_t icpg = src[2];
            size_t h = src[3];
            size_t w = src[4];
            return group * ocpg * icpg * h * w * src.dtype.size();
        }

        case Param::Mode::NCHW_NCHW4_IC_SMALL: {
            if (src[1] % 4 == 0)
                return 0;
            size_t n = src[0];
            size_t c = round_up(src[1], 4_z);
            size_t h = src[2];
            size_t w = src[3];
            return n * c * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW4: {
            size_t group = param().group;
            size_t n = src[0];
            size_t c = group * round_up(src[1] / group, 4_z);
            size_t h = src[2];
            size_t w = src[3];
            return n * c * h * w * src.dtype.size();
        }
        case Param::Mode::NCHW4_NCHW: {
            return src.total_nr_elems() * src.dtype.size();
        }
        case Param::Mode::NCHW_NCHW4_WEIGHT: {
            if (src.ndim == 4) {
                size_t oc = round_up(src[0], 4_z);
                size_t ic = round_up(src[1], 4_z);
                size_t h = src[2];
                size_t w = src[3];
                return oc * ic * h * w * src.dtype.size();
            } else if (src.ndim == 5) {
                size_t group = src[0];
                size_t oc = round_up(src[1], 4_z);
                size_t ic = round_up(src[2], 4_z);
                size_t h = src[3];
                size_t w = src[4];
                return group * oc * ic * h * w * src.dtype.size();
            } else {
                megdnn_throw("no support nchw_nchw4_weight");
            }
        }
        case Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT: {
            megdnn_assert(src.ndim == 4, "src must be oihw, ndim == 5");
            if (src[1] % 4 == 0)
                return 0;
            size_t oc = src[0];
            size_t ic = round_up(src[1], 4_z);
            size_t h = src[2];
            size_t w = src[3];
            return oc * ic * h * w * src.dtype.size();
        }

        case Param::Mode::NCHW_NCHW64: {
            if (src[1] % 64 != 0) {
                size_t n = src[0];
                size_t c = round_up(src[1], 64_z);
                size_t h = src[2];
                size_t w = src[3];
                TensorLayout wsly({n, c, h, w}, src.dtype);
                return wsly.span().dist_byte();
            }
            return 0_z;
        }

        case Param::Mode::NCHW64_NCHW: {
            if (param().oc != 0) {
                size_t n = src[0];
                size_t c = src[1] * 64;
                size_t h = src[2];
                size_t w = src[3];
                TensorLayout wsly({n, c, h, w}, dst.dtype);
                return wsly.span().dist_byte();
            }
            return 0_z;
        }

        default:
            return 0;
    }
}

void RelayoutFormatImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    megdnn_assert(
            src.layout.dtype.category() == DTypeCategory::FLOAT ||
            src.layout.dtype.enumv() == DTypeEnum::Int32 ||
            (src.layout.dtype.enumv() == DTypeEnum::Uint8 &&
             dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) ||
            (src.layout.dtype.enumv() == DTypeEnum::Uint8 &&
             dst.layout.dtype.enumv() == DTypeEnum::Uint8) ||
            src.layout.dtype.category() == DTypeCategory::QUANTIZED);
    check_exec(src.layout, dst.layout, workspace.size);
    HandleImpl* m_handle = static_cast<HandleImpl*>(handle());
    TensorLayout exec_src_layout, exec_dst_layout, exec_workspace_layout;
    deduce_exec_layout(
            src.layout, dst.layout, exec_workspace_layout, exec_src_layout,
            exec_dst_layout);

    // clean dst
    MEGDNN_DISPATCH_CPU_KERN(
            m_handle, memset(dst.raw_ptr(), 0, dst.layout.span().dist_byte()));

    //! construct exec Tensor
    TensorND exec_src_nd{exec_src_layout, src.get_ref_ptr()};
    TensorND exec_dst_nd{exec_dst_layout, dst.get_ref_ptr()};

    // pre
    if (param().mode == Param::Mode::NCHW_NHWCD4I) {
        size_t N = src.layout[0];
        size_t IC = src.layout[1];
        size_t IH = src.layout[2];
        size_t IW = src.layout[3];
        //! ic % 4 != 0
        if ((IC & 0x3)) {
            switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                                              \
    case (DTypeEnum::name): {                                                        \
        MIDOUT_BEGIN(                                                                \
                megdnn_naive_relayout_format, ctype,                                 \
                midout_iv(Param::Mode::NCHW_NHWCD4I)) {                              \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    m_handle, padding_src_to_workspace<ctype>(                       \
                                      workspace.ptr<ctype>(),                        \
                                      src.compatible_ptr<ctype>(), N, IC, IH, IW);); \
        }                                                                            \
        MIDOUT_END();                                                                \
        break;                                                                       \
    }
                cb(Float32, dt_float32);
                DNN_INC_FLOAT16(cb(Float16, dt_float16));
                cb(Quantized8Asymm, dt_uint8);
                cb(QuantizedS8, dt_int8);
                cb(Uint8, dt_uint8);
#undef cb
                default:
                    megdnn_assert(
                            0, "NCHW_NHWCD4I not support dtype %s",
                            src.layout.dtype.name());
            }
            exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};
        }
    } else if (param().mode == Param::Mode::NHWC_NHWCD4I) {
        size_t N = src.layout[0];
        size_t IC = src.layout[3];
        size_t IH = src.layout[1];
        size_t IW = src.layout[2];
        //! ic % 4 != 0
        if ((IC & 0x3)) {
            switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                                              \
    case (DTypeEnum::name): {                                                        \
        MIDOUT_BEGIN(                                                                \
                megdnn_naive_relayout_format, ctype,                                 \
                midout_iv(Param::Mode::NHWC_NHWCD4I)) {                              \
            MEGDNN_DISPATCH_CPU_KERN(                                                \
                    m_handle, padding_nhwc_src_to_workspace<ctype>(                  \
                                      workspace.ptr<ctype>(),                        \
                                      src.compatible_ptr<ctype>(), N, IH, IW, IC);); \
        }                                                                            \
        MIDOUT_END();                                                                \
        break;                                                                       \
    }
                cb(Float32, dt_float32);
                DNN_INC_FLOAT16(cb(Float16, dt_float16));
                cb(Quantized8Asymm, dt_uint8);
                cb(QuantizedS8, dt_int8);
                cb(Uint8, dt_uint8);
#undef cb
                default:
                    megdnn_assert(
                            0, "NHWC_NHWCD4I not support dtype %s",
                            src.layout.dtype.name());
            }
            exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};
        }
    } else if (param().mode == Param::Mode::INTER_WEIGHT_DENSEI_DOT) {
        size_t OC = src.layout[0];
        size_t IC = src.layout[1];
        size_t FH = src.layout[2];
        size_t FW = src.layout[3];
        if ((IC & 0x3)) {
            switch (src.layout.dtype.enumv()) {
#define cb(name, ctype)                                                               \
    case (DTypeEnum::name): {                                                         \
        MIDOUT_BEGIN(                                                                 \
                megdnn_naive_relayout_format, ctype,                                  \
                midout_iv(Param::Mode::INTER_WEIGHT_DENSEI_DOT)) {                    \
            MEGDNN_DISPATCH_CPU_KERN(                                                 \
                    m_handle, padding_filter_to_workspace<ctype>(                     \
                                      workspace.ptr<ctype>(),                         \
                                      src.compatible_ptr<ctype>(), OC, IC, FH, FW);); \
        }                                                                             \
        MIDOUT_END();                                                                 \
        break;                                                                        \
    }
                cb(Quantized8Asymm, dt_uint8);
                cb(QuantizedS8, dt_int8);
#undef cb
                default:
                    megdnn_assert(0);
            }
            exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};
        }
    }
#define cb(_idx, _pack_size, _mode)                                             \
    MIDOUT_BEGIN(megdnn_naive_relayout_format, midout_iv(Param::Mode::_mode)) { \
        size_t val = src.layout[_idx];                                          \
        if (val % _pack_size != 0) {                                            \
            exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};      \
            MEGDNN_DISPATCH_CPU_KERN(                                           \
                    m_handle, padding_to_workspace(                             \
                                      exec_src_nd, src, _idx, _pack_size,       \
                                      exec_dst_layout.dtype));                  \
        }                                                                       \
    }                                                                           \
    MIDOUT_END();
#define cb2(_idx, _pack_size, _mode, _src_layout, _workspace_layout)               \
    MIDOUT_BEGIN(megdnn_naive_relayout_format, midout_iv(Param::Mode::_mode)) {    \
        size_t val = _src_layout[_idx];                                            \
        if (val % _pack_size != 0) {                                               \
            MEGDNN_DISPATCH_CPU_KERN(m_handle,                                     \
                                     memset(workspace.raw_ptr, 0,                  \
                                            exec_src_layout.span().dist_byte());); \
            TensorND tmp_dst{workspace.raw_ptr, _workspace_layout};                \
            TensorND tmp_src{_src_layout, src.get_ref_ptr()};                      \
            MEGDNN_DISPATCH_CPU_KERN(                                              \
                    m_handle, padding_to_workspace(tmp_dst, tmp_src));             \
            exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};         \
        }                                                                          \
    }                                                                              \
    MIDOUT_END();
    else if (param().mode == Param::Mode::NCHW_NCHW88) {
        cb(1, 8, NCHW_NCHW88);
    } else if (param().mode == Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT) {
        megdnn_assert(src.layout[0] % 8 == 0);
        cb(1, 8, NCHW_NCHW88_CONV_DENSE_WEIGHT);
    } else if (param().mode == Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT) {
        cb(0, 8, NCHW_NCHW88_CONV_CHAN_WEIGHT);
    } else if (param().mode == Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT) {
        megdnn_assert(src.layout[1] % 8 == 0);
        cb(2, 8, NCHW_NCHW88_CONV_GROUP_WEIGHT);
    } else if (param().mode == Param::Mode::NCHW_NCHW4_IC_SMALL) {
        cb(1, 4, NCHW_NCHW4_IC_SMALL);
    } else if (param().mode == Param::Mode::NCHW_NCHW4) {
        if (param().group == 1) {
            cb(1, 4, NCHW_NCHW4);
        } else {
            TensorLayout group_src_layout{
                    {src.layout[0], param().group, src.layout[1] / param().group,
                     src.layout[2], src.layout[3]},
                    src.layout.dtype,
                    src.layout.format};
            TensorLayout workspace_layout{
                    {src.layout[0], param().group,
                     div_ceil(src.layout[1] / param().group, 4_z) * 4_z, src.layout[2],
                     src.layout[3]},
                    src.layout.dtype,
                    src.layout.format};
            cb2(2, 4, NCHW_NCHW4, group_src_layout, workspace_layout);
        }
    } else if (param().mode == Param::Mode::NCHW_NCHW64) {
        MIDOUT_BEGIN(
                megdnn_naive_relayout_format, midout_iv(Param::Mode::NCHW_NCHW64)) {
            size_t c = src.layout[1];
            if (c % 64 != 0) {
                uint8_t zp = 0;
                if (src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
                    zp = src.layout.dtype.param<dtype::Quantized4Asymm>().zero_point;
                    zp = (zp & 0xf) | (zp << 4);
                }
                MEGDNN_DISPATCH_CPU_KERN(
                        m_handle, memset(workspace.raw_ptr, zp,
                                         exec_workspace_layout.span().dist_byte()));
                TensorND ws_nd(workspace.raw_ptr, exec_workspace_layout);
                MEGDNN_DISPATCH_CPU_KERN(m_handle, padding_to_workspace(ws_nd, src););
                exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};
            }
        }
        MIDOUT_END();
    } else if (param().mode == Param::Mode::NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT) {
        cb(1, 4, NCHW_NCHW4_IC_SMALL_CONV_DENSE_WEIGHT);
#undef cb
#undef cb2
    } else if (param().mode == Param::Mode::NCHW_NCHW4_WEIGHT) {
#define cb(_idx0, _idx1, _pack_size, _mode)                                            \
    MIDOUT_BEGIN(megdnn_naive_relayout_format, midout_iv(Param::Mode::_mode)) {        \
        size_t val0 = src.layout[_idx0];                                               \
        size_t val1 = src.layout[_idx1];                                               \
        if (val0 % _pack_size != 0 || val1 % _pack_size != 0) {                        \
            MEGDNN_DISPATCH_CPU_KERN(                                                  \
                    m_handle,                                                          \
                    memset(workspace.raw_ptr, 0, exec_src_layout.span().dist_byte())); \
            TensorND ws_nd(workspace.raw_ptr, exec_workspace_layout);                  \
            MEGDNN_DISPATCH_CPU_KERN(m_handle, padding_to_workspace(ws_nd, src););     \
            exec_src_nd = TensorND{workspace.raw_ptr, exec_src_nd.layout};             \
        }                                                                              \
    }                                                                                  \
    MIDOUT_END();
        if (src.layout.ndim == 4) {
            cb(0, 1, 4, NCHW_NCHW4_WEIGHT);
        } else if (src.layout.ndim == 5) {
            cb(1, 2, 4, NCHW_NCHW4_WEIGHT);
        }
#undef cb
    } else if (param().mode == Param::Mode::NCHW4_NCHW) {
        if (exec_workspace_layout.total_nr_elems() != dst.layout.total_nr_elems()) {
            exec_dst_nd = TensorND{workspace.raw_ptr, exec_workspace_layout};
        }
    } else if (param().mode == Param::Mode::NCHW64_NCHW) {
        if (exec_workspace_layout.total_nr_elems() != dst.layout.total_nr_elems()) {
            exec_dst_nd = TensorND{workspace.raw_ptr, exec_workspace_layout};
        }
    }

    // do relayout
    if (src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm &&
        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
        check_layout_and_canonize(exec_src_nd.layout, exec_dst_nd.layout);
        MEGDNN_DISPATCH_CPU_KERN(
                m_handle, do_copy_diff_qu8_q8(exec_dst_nd, exec_src_nd));
    } else if (
            src.layout.dtype.enumv() == DTypeEnum::Uint8 &&
            dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
        check_layout_and_canonize(exec_src_nd.layout, exec_dst_nd.layout);
        MEGDNN_DISPATCH_CPU_KERN(
                m_handle, do_copy_diff_u8_q8(exec_dst_nd, exec_src_nd));
    } else if (
            src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&
            dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
        check_layout_and_canonize(exec_src_nd.layout, exec_dst_nd.layout);
        MEGDNN_DISPATCH_CPU_KERN(
                m_handle, do_copy_diff_q8_q8(exec_dst_nd, exec_src_nd));
    } else if (
            src.layout.dtype.enumv() == DTypeEnum::QuantizedS32 &&
            dst.layout.dtype.enumv() == DTypeEnum::QuantizedS32) {
        check_layout_and_canonize(exec_src_nd.layout, exec_dst_nd.layout);
        MEGDNN_DISPATCH_CPU_KERN(
                m_handle, do_copy_diff_q32_q32(exec_dst_nd, exec_src_nd));
    } else if (
            src.layout.dtype.enumv() == DTypeEnum::QuantizedS4 &&
            dst.layout.dtype.enumv() == DTypeEnum::QuantizedS4) {
        check_layout_and_canonize(exec_src_nd.layout, exec_dst_nd.layout);
        MEGDNN_DISPATCH_CPU_KERN(
                m_handle, do_copy_diff_q4_q4(exec_dst_nd, exec_src_nd));
    } else if (
            src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm &&
            dst.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        check_layout_and_canonize(exec_src_nd.layout, exec_dst_nd.layout);
        MEGDNN_DISPATCH_CPU_KERN(
                m_handle, do_copy_diff_qu4_qu4(exec_dst_nd, exec_src_nd));
    } else {
        m_handle->relayout_opr()->exec(exec_src_nd, exec_dst_nd, handle());
    }

    // post
    if (param().mode == Param::Mode::NCHW4_NCHW ||
        param().mode == Param::Mode::NCHW64_NCHW) {
        if (exec_workspace_layout.total_nr_elems() != dst.layout.total_nr_elems()) {
            megdnn_assert(exec_workspace_layout.dtype == dst.layout.dtype);
            TensorND ws_nd{workspace.raw_ptr, exec_workspace_layout};
            MEGDNN_DISPATCH_CPU_KERN(
                    m_handle, extract_from_workspace(dst, ws_nd, param().group););
        }
    }
#else
    __builtin_trap();
#endif
}

// vim: syntax=cpp.doxygen
