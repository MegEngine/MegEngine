/**
 * \file dnn/src/common/tensor_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/tensor_format.h"
#include "megdnn/basic_types.h"
#include "src/common/utils.h"

#include <unordered_map>

using namespace megdnn;
using namespace megdnn::detail;

namespace {
DefaultTensorFormat* default_tensor_format_obj;
}

/* ===================== TensorFormat ===================== */

TensorFormat TensorFormat::deserialize(const std::string& bin, const Handle* handle) {
    using Type = TensorFormat::Type;
    auto type = reinterpret_cast<const Type*>(bin.data());
    switch (*type) {
        case Type::DEFAULT:
            return DefaultTensorFormat::deserialize(
                    handle, type + 1, bin.size() - sizeof(Type));
        case Type::IMAGE2D_PACK4:
            return Image2DPack4TensorFormat::deserialize(
                    handle, type + 1, bin.size() - sizeof(Type));
        case Type::LOWBITS_ALIGNED_TO_BYTE:
            return LowbitsAlignedToBytesTensorFormat::deserialize(
                    handle, type + 1, bin.size() - sizeof(Type));
        default:
            megdnn_throw("invalid tensor format type in deserialize");
    }
}

TensorFormat::Format() : m_impl{DefaultTensorFormat::make().m_impl} {}

TensorFormat::Format(DType dtype) {
    if (dtype.valid() && dtype.is_quantized_lowbit()) {  // quantized lowbit, by default
                                                         // aligned to bytes
        size_t size_nbits = dtype.low_bit();
        megdnn_assert(
                size_nbits == 1 || size_nbits == 2 || size_nbits == 4,
                "unsupported lowbits data type(%s, size in bits: %zu)", dtype.name(),
                size_nbits);
        m_impl = LowbitsAlignedToBytesTensorFormat::make(size_nbits).m_impl;
    } else {  // non parameterized lowbit, default format
        m_impl = DefaultTensorFormat::make().m_impl;
    }
}

std::string TensorFormat::to_string() const {
    return m_impl->to_string();
}

std::string TensorFormat::serialize() const {
    std::string ret;
    ret.reserve(32);
    ret.assign(sizeof(Type), '\0');
    *reinterpret_cast<Type*>(&ret[0]) = type();
    m_impl->serialize_append(ret);
    return ret;
}

void TensorFormat::on_bad_cvt(Type dst_type) const {
    MEGDNN_MARK_USED_VAR(dst_type);
    megdnn_throw(ssprintf(
            "can not convert tensor format %s to %d", impl()->to_string().c_str(),
            static_cast<int>(dst_type)));
}

bool TensorFormat::is_default() const {
    return m_impl == default_tensor_format_obj;
}

bool TensorFormat::is_lowbit_aligned() const {
    return type() == TensorFormat::Type::LOWBITS_ALIGNED_TO_BYTE;
}

/* ===================== DefaultFormat ===================== */
void DefaultTensorFormat::assert_valid(const TensorLayout& layout) const {
    megdnn_assert(
            !layout.dtype.valid() || !layout.dtype.is_quantized_lowbit(),
            "DefaultTensorFormat does not support quantized lowbit tensor(dtype:%s)",
            layout.dtype.name());
}

size_t DefaultTensorFormat::init_contiguous_stride(TensorLayout& layout) const {
    assert_valid(layout);
    if (!layout.ndim)
        return 0;
    megdnn_assert(layout.ndim <= TensorLayout::MAX_NDIM);
    size_t accum = 1;
    SafeMultiplies<size_t> mul;
    for (size_t i = layout.ndim; i; --i) {
        layout.stride[i - 1] = accum;
        accum = mul(accum, layout.shape[i - 1]);
    }
    return accum;
}

bool DefaultTensorFormat::is_contiguous_spec(const TensorLayout& layout) const {
    assert_valid(layout);
    return layout.is_physical_contiguous();
}

TensorLayout DefaultTensorFormat::collapse_contiguous_spec(
        const TensorLayout& layout) const {
    assert_valid(layout);
    megdnn_assert(layout.ndim);
    TensorLayout res{layout};

    // remove all dims with shape 1
    for (int i = static_cast<int>(res.ndim) - 1; i >= 0 && res.ndim >= 2; --i) {
        if (!res.shape[i]) {
            // empty tensor
            res.ndim = 1;
            res.shape[0] = 0;
            res.stride[0] = 1;
            return res;
        }
        if (res.shape[i] == 1)
            res.remove_axis_inplace(i);
    }

    if (res.ndim == 1) {
        if (res.shape[0] <= 1) {
            // make it the "most canonical" contiguous layout for scalars or
            // empty tensors
            res.stride[0] = 1;
        }
        return res;
    }

    megdnn_assert(res.ndim && res.shape[res.ndim - 1]);
    for (int i = static_cast<int>(res.ndim) - 2; i >= 0; --i) {
        megdnn_assert(res.shape[i]);
        if (res.stride[i] ==
            res.stride[i + 1] * static_cast<ptrdiff_t>(res.shape[i + 1])) {
            res.shape[i] *= res.shape[i + 1];
            res.stride[i] = res.stride[i + 1];
            res.remove_axis_inplace(i + 1);
        }
    }
    return res;
}

TensorLayout::Span DefaultTensorFormat::span_spec(const TensorLayout& layout) const {
    assert_valid(layout);
    if (layout.ndim == 0)
        return {0, 0, 0, 0};

    ptrdiff_t low_elem = 0;
    size_t high_elem = 0;
    for (size_t i = 0; i < layout.ndim; ++i) {
        auto shape_val = layout.shape[i];
        if (!shape_val) {
            return {0, 0, 0, 0};
        }
        auto stride_val = layout.stride[i];
        if (stride_val > 0) {
            high_elem += (shape_val - 1) * stride_val;
        } else {
            low_elem += (shape_val - 1) * stride_val;
        }
    }
    ++high_elem;
    ptrdiff_t low_byte;
    if (low_elem < 0) {
        low_byte = low_elem * layout.dtype.size();
    } else {
        low_byte = 0;
    }
    size_t high_byte = layout.dtype.size(high_elem);
    return TensorLayout::Span(low_elem, low_byte, high_elem, high_byte);
}

std::string DefaultTensorFormat::to_string() const {
    return "default{}";
}

void DefaultTensorFormat::serialize_append(std::string&) const {}

TensorFormat DefaultTensorFormat::deserialize(
        const Handle* handle, const void* buf, size_t size) {
    MEGDNN_MARK_USED_VAR(handle);
    MEGDNN_MARK_USED_VAR(buf);
    megdnn_assert(!size);
    return make();
}

TensorFormat DefaultTensorFormat::make() {
    // use static storage so the object is accessible in global destructing
    // phase
    static std::aligned_storage_t<
            sizeof(DefaultTensorFormat), alignof(DefaultTensorFormat)>
            storage;
    static DefaultTensorFormat* obj = default_tensor_format_obj =
            new (&storage) DefaultTensorFormat{};
    return impl_to_tensor_format(obj);
}

/* ===================== Image2DTensorFormatBase ===================== */

Image2DTensorFormatBase::Image2DTensorFormatBase(
        Type type, size_t align_axis, size_t align_size_in_elements)
        : ImplBase(type), m_align_axis(align_axis) {
    megdnn_assert(align_size_in_elements && align_axis);
    m_align_size_in_elements_log2 = __builtin_ctz(align_size_in_elements);
    megdnn_assert(
            (1u << m_align_size_in_elements_log2) == align_size_in_elements,
            "align size not power of 2: %zu", align_size_in_elements);
}

void Image2DTensorFormatBase::serialize_append(std::string& result) const {
    SerializePack pack;
    pack.align_axis = m_align_axis;
    megdnn_assert(pack.align_axis == m_align_axis);  // detect overflow
    result.append(reinterpret_cast<char*>(&pack), sizeof(pack));
}

size_t Image2DTensorFormatBase::image_height(const TensorLayout& layout) const {
    size_t accum = 1;
    for (int i = m_align_axis - 1; i >= 0; --i) {
        if (layout.stride[i] == 0) {
            // this dimension is broadcasted
        } else {
            accum *= layout.shape[i];
        }
    }
    return accum;
}

size_t Image2DTensorFormatBase::image_width_elems(const TensorLayout& layout) const {
    size_t high_elem = 0;
    for (size_t i = m_align_axis; i < layout.ndim; ++i) {
        high_elem += (layout.shape[i] - 1) * layout.stride[i];
    }
    return high_elem + 1;
}

std::string Image2DTensorFormatBase::to_string() const {
    return ssprintf("I2D{%zu,%d}", m_align_axis, 1 << m_align_size_in_elements_log2);
}

/* ===================== Image2DPackedTensorFormatBase ===================== */

template <size_t PIXEL_SIZE>
size_t Image2DPackedTensorFormatBase<PIXEL_SIZE>::image_width(
        const TensorLayout& layout) const {
    auto ret = image_width_elems(layout);
    megdnn_assert(ret % PIXEL_SIZE == 0);
    return ret / PIXEL_SIZE;
}

template <size_t PIXEL_SIZE>
void Image2DPackedTensorFormatBase<PIXEL_SIZE>::assert_valid(
        const TensorLayout& layout) const {
    auto m_align_axis = align_axis();
    megdnn_assert(
            !(layout.shape[layout.ndim - 1] % PIXEL_SIZE), "bad shape: %zu",
            layout.shape[layout.ndim - 1]);
    megdnn_assert(
            layout.dtype.valid() && !layout.dtype.is_quantized_lowbit() &&
            layout.ndim > m_align_axis);
    ptrdiff_t first_non_zero_stride = 0;
    for (int i = layout.ndim - 1; i >= 0; --i) {
        megdnn_assert(layout.shape[i] && layout.stride[i] >= 0);
        if (i < static_cast<int>(m_align_axis) && !first_non_zero_stride) {
            first_non_zero_stride = layout.stride[i];
        }
    }
    size_t mask = image_pitch_alignment_in_bytes(
                          align_size_in_elements(layout.dtype.size_log()), layout) -
                  1;

    megdnn_assert(
            !(first_non_zero_stride & mask), "first stride is %d, but alignment is %zu",
            static_cast<int>(first_non_zero_stride), mask + 1);
}

template <size_t PIXEL_SIZE>
size_t Image2DPackedTensorFormatBase<PIXEL_SIZE>::image_row_pitch(
        const TensorLayout& layout) const {
    for (int i = align_axis() - 1; i >= 0; --i) {
        // find a non-broadcast axis
        if (auto s = layout.stride[i]) {
            return layout.dtype.size(s);
        }
    }
    // use width for all broadcasted case
    size_t alignment_in_bytes_log2 = align_size_in_elements_log2();
    if (m_vendor_type == Handle::HandleVendorType::MALI) {
        alignment_in_bytes_log2 += __builtin_ctz(layout.dtype.size() * PIXEL_SIZE);
    }

    return get_aligned_power2<size_t>(
            layout.dtype.size(image_width_elems(layout)), 1 << alignment_in_bytes_log2);
}

template <size_t PIXEL_SIZE>
size_t Image2DPackedTensorFormatBase<PIXEL_SIZE>::image_pitch_alignment_in_bytes(
        size_t align_size_in_elements, const TensorLayout& layout) const {
    return m_vendor_type == Handle::HandleVendorType::MALI
                 ? (align_size_in_elements * layout.dtype.size() * PIXEL_SIZE)
                 : align_size_in_elements;
}

template <size_t PIXEL_SIZE>
TensorLayout::Span Image2DPackedTensorFormatBase<PIXEL_SIZE>::span_spec(
        const TensorLayout& layout) const {
    assert_valid(layout);
    size_t size = image_height(layout) * image_row_pitch(layout);
    auto mask = (1 << layout.dtype.size_log()) - 1;
    megdnn_assert(!(size & mask), "unaligned size: %zu", size);
    return {0, 0, size >> layout.dtype.size_log(), size};
}

template <size_t PIXEL_SIZE>
size_t Image2DPackedTensorFormatBase<PIXEL_SIZE>::init_contiguous_stride(
        TensorLayout& layout) const {
    auto m_align_axis = align_axis();
    if (!layout.ndim)
        return 0;
    megdnn_assert(
            layout.dtype.valid() && layout.ndim > m_align_axis,
            "dtype=%s ndim=%zu align=%zu", layout.dtype.name(), layout.ndim,
            m_align_axis);
    size_t align_size = image_pitch_alignment_in_bytes(
            align_size_in_elements(layout.dtype.size_log()), layout);

    size_t accum = 1;
    SafeMultiplies<size_t> mul;
    for (size_t i = layout.ndim; i; --i) {
        if (i == m_align_axis) {
            accum = get_aligned_power2<size_t>(accum, align_size);
        }

        layout.stride[i - 1] = accum;
        accum = mul(accum, layout.shape[i - 1]);
    }
    assert_valid(layout);
    return accum;
};

template <size_t PIXEL_SIZE>
bool Image2DPackedTensorFormatBase<PIXEL_SIZE>::is_contiguous_spec(
        const TensorLayout& layout) const {
    megdnn_assert(layout.dtype.valid());
    size_t align_size = image_pitch_alignment_in_bytes(
            align_size_in_elements(layout.dtype.size_log()), layout);

    ptrdiff_t expected = 1;
    int height_axis = static_cast<int>(align_axis() - 1);
    for (int i = layout.ndim - 1; i >= 0; --i) {
        if (i == height_axis) {
            expected = megdnn::get_aligned_power2<size_t>(expected, align_size);
        }
        if (layout.shape[i] != 1 && layout.stride[i] != expected) {
            if (i == height_axis) {
                // allow row pitch to be larger than minimal required
                auto s = layout.stride[i];
                if (!s) {
                    // broadcast is not contiguous
                    return false;
                }

                size_t mask = image_pitch_alignment_in_bytes(
                                      align_size_in_elements(layout.dtype.size_log()),
                                      layout) -
                              1;

                megdnn_assert(
                        s > expected && !(s & mask),
                        "invalid row pitch: %d; layout: %s", static_cast<int>(s),
                        layout.to_string().c_str());
                expected = s;
            } else {
                return false;
            }
        }
        expected *= layout.shape[i];
    }
    // empty tensors are not contiguous
    return expected != 0;
}

template <size_t PIXEL_SIZE>
TensorLayout Image2DPackedTensorFormatBase<PIXEL_SIZE>::collapse_contiguous_spec(
        const TensorLayout& layout) const {
    assert_valid(layout);
    TensorLayout res{layout};
    int new_axis = align_axis();
    // remove all dims with shape 1
    for (int i = static_cast<int>(res.ndim) - 1; i >= 0 && res.ndim >= 3; --i) {
        if (i == new_axis && static_cast<int>(res.ndim) == new_axis + 1) {
            // i is the only width dim
            continue;
        }
        if (i == new_axis - 1 && !i) {
            // new_xis == 1 && i == 0, i is the only height dim
            continue;
        }
        if (res.shape[i] == 1) {
            res.remove_axis_inplace(i);
            if (i < new_axis)
                new_axis -= 1;
        }
    }
    megdnn_assert(res.ndim >= 2);

    auto contig_with_next = [&](size_t i) {
        return res.stride[i] ==
               res.stride[i + 1] * static_cast<ptrdiff_t>(res.shape[i + 1]);
    };

    for (int i = static_cast<int>(res.ndim) - 2; i >= new_axis; --i) {
        megdnn_assert(res.shape[i]);
        if (contig_with_next(i)) {
            // remove next axis
            res.shape[i] *= res.shape[i + 1];
            res.stride[i] = res.stride[i + 1];
            res.remove_axis_inplace(i + 1);
        }
    }

    for (int i = new_axis - 2; i >= 0; --i) {
        megdnn_assert(res.shape[i]);
        if (contig_with_next(i)) {
            res.shape[i] *= res.shape[i + 1];
            res.stride[i] = res.stride[i + 1];
            res.remove_axis_inplace(i + 1);
            if (i <= new_axis - 2)
                new_axis -= 1;
        }
    }
    res.format = change_axis(new_axis);
    return res;
}

namespace megdnn {
namespace detail {
template class Image2DPackedTensorFormatBase<4>;
}  // namespace detail
}  // namespace megdnn

/* =============== LowbitsAlignedTensorFormatBase ============== */
LowbitsAlignedTensorFormatBase::LowbitsAlignedTensorFormatBase(
        Type type, size_t size_nbits, size_t align_size_in_bits)
        : ImplBase(type),
          m_size_nbits(size_nbits),
          m_align_size_in_bits(align_size_in_bits) {
    megdnn_assert(
            !(m_align_size_in_bits % m_size_nbits),
            "align size(%zu) must be a multiple of element size(%zu)",
            m_align_size_in_bits, m_size_nbits);
    m_align_size_in_elements = m_align_size_in_bits / m_size_nbits;
}

std::string LowbitsAlignedTensorFormatBase::to_string() const {
    return ssprintf("LOWBITS{%zu,%zu}", m_size_nbits, m_align_size_in_bits);
}

void LowbitsAlignedTensorFormatBase::assert_valid(const TensorLayout& layout) const {
    megdnn_assert(
            layout.dtype.valid() && layout.dtype.is_low_bit() &&
            layout.dtype.low_bit() == m_size_nbits);
    bool has_dim_unity_stride = false;
    bool has_dim_aligned_stride = false;
    for (int i = layout.ndim - 1; i >= 0; --i) {
        if (!has_dim_unity_stride && layout.stride[i] == 1)
            has_dim_unity_stride = true;
        megdnn_assert(
                layout.stride[i] >= 0 &&
                        (layout.stride[i] % m_align_size_in_elements == 0 ||
                         layout.stride[i] == 1),
                "bad stride:%s, %ld", layout.to_string().c_str(),
                static_cast<long>(layout.stride[i]));
        if (!has_dim_aligned_stride &&
            static_cast<size_t>(layout.stride[i]) == m_align_size_in_elements)
            has_dim_aligned_stride = true;
    }

    megdnn_assert(
            layout.ndim == 0 || has_dim_unity_stride || has_dim_aligned_stride,
            "innermost dim not contiguous");
}

void LowbitsAlignedTensorFormatBase::serialize_append(std::string& result) const {
    SerializePack pack;
    pack.size_nbits = m_size_nbits;
    pack.align_size_in_bits = m_align_size_in_bits;
    megdnn_assert(pack.align_size_in_bits == m_align_size_in_bits);  // detect overflow;
    result.append(reinterpret_cast<char*>(&pack), sizeof(pack));
}

TensorLayout::Span LowbitsAlignedTensorFormatBase::span_spec(
        const TensorLayout& layout) const {
    assert_valid(layout);
    if (layout.ndim == 0)
        return {0, 0, 0, 0};

    size_t high_elem = 0;
    for (size_t i = 0; i < layout.ndim; ++i) {
        auto shape_val = layout.shape[i];
        if (!shape_val) {
            return {0, 0, 0, 0};
        }
        auto stride_val = layout.stride[i];
        megdnn_assert(
                stride_val >= 0, "lowbit tensors shouldn't have negative strides");
        high_elem += (shape_val - 1) * stride_val;
    }
    ++high_elem;
    size_t high_byte = layout.dtype.size(high_elem);
    return TensorLayout::Span(0, 0, high_elem, high_byte);
}

size_t LowbitsAlignedTensorFormatBase::init_contiguous_stride(
        TensorLayout& layout) const {
    if (!layout.ndim)
        return 0;
    megdnn_assert(layout.ndim <= TensorLayout::MAX_NDIM);
    size_t accum = 1;
    SafeMultiplies<size_t> mul;
    for (size_t i = layout.ndim; i; --i) {
        layout.stride[i - 1] = accum;
        auto multiplier = layout.shape[i - 1];
        if (i == layout.ndim)
            multiplier = round_up(multiplier, m_align_size_in_elements);
        accum = mul(accum, multiplier);
    }
    assert_valid(layout);
    return accum;
}

bool LowbitsAlignedTensorFormatBase::is_contiguous_spec(
        const TensorLayout& layout) const {
    assert_valid(layout);
    ptrdiff_t expected = 1;
    for (int i = static_cast<int>(layout.ndim) - 1; i >= 0; --i) {
        bool is_valid_stride =
                (layout.stride[i] == expected) ||
                (expected == 1 &&
                 (int)layout.stride[i] == round_up(1, (int)m_align_size_in_elements));
        if (layout.shape[i] != 1 && !is_valid_stride)
            return false;
        auto multiplier = layout.shape[i];
        if (i == static_cast<int>(layout.ndim) - 1)
            multiplier = round_up(multiplier, m_align_size_in_elements);
        expected *= multiplier;
    }
    return expected != 0;
}

TensorLayout LowbitsAlignedTensorFormatBase::collapse_contiguous_spec(
        const TensorLayout& layout) const {
    assert_valid(layout);
    TensorLayout res{layout};
    for (int i = static_cast<int>(res.ndim) - 1; i >= 0; --i) {
        if (!res.shape[i]) {
            // empty tensor
            res.ndim = 1;
            res.shape[0] = 0;
            res.stride[0] = 1;
            return res;
        }
        if (res.shape[i] == 1) {
            res.remove_axis_inplace(i);
        }
    }

    megdnn_assert(res.ndim && res.shape[res.ndim - 1]);
    for (int i = static_cast<int>(res.ndim) - 2; i >= 0; --i) {
        megdnn_assert(res.shape[i]);
        if (res.stride[i] ==
            res.stride[i + 1] * static_cast<ptrdiff_t>(res.shape[i + 1])) {
            res.shape[i] *= res.shape[i + 1];
            res.stride[i] = res.stride[i + 1];
            res.remove_axis_inplace(i + 1);
        }
    }
    return res;
}

/* ===================== Image2DPack4TensorFormat  ===================== */
TensorFormat Image2DPack4TensorFormat::make_raw(
        size_t align_axis, size_t align_size_in_elements,
        Handle::HandleVendorType vendor_type) {
    static DNN_MUTEX mtx;
    static std::unordered_map<uint64_t, std::unique_ptr<Image2DPack4TensorFormat>>
            cache;
    megdnn_assert(
            std::max(align_axis, align_size_in_elements) <=
            std::numeric_limits<uint32_t>::max());
    MEGDNN_LOCK_GUARD(mtx);
    auto&& ptr =
            cache[(static_cast<uint64_t>(align_axis) << 32) | align_size_in_elements];
    if (!ptr) {
        ptr.reset(new Image2DPack4TensorFormat{
                align_axis, align_size_in_elements, vendor_type});
    }
    return impl_to_tensor_format(ptr.get());
}

TensorFormat Image2DPack4TensorFormat::make(size_t align_axis, const Handle* handle) {
    return make_raw(
            align_axis, handle->image2d_pitch_alignment(), handle->vendor_type());
}

TensorFormat Image2DPack4TensorFormat::deserialize(
        const Handle* handle, const void* buf, size_t size) {
    megdnn_assert(size == sizeof(SerializePack));
    auto pack = *static_cast<const SerializePack*>(buf);
    return make(pack.align_axis, handle);
}

TensorFormat Image2DPack4TensorFormat::change_axis(size_t axis) const {
    return make_raw(axis, align_size_in_elements(), vendor());
}

/* ===================== LowbitsitsAlignedToBytesTensorFormat
 * ===================== */
TensorFormat LowbitsAlignedToBytesTensorFormat::make(size_t size_nbits) {
    static DNN_MUTEX mtx;
    static std::unordered_map<
            uint64_t, std::unique_ptr<LowbitsAlignedToBytesTensorFormat>>
            cache;
    megdnn_assert(!(8 % size_nbits));
    MEGDNN_LOCK_GUARD(mtx);
    auto&& ptr = cache[static_cast<uint32_t>(size_nbits)];
    if (!ptr) {
        ptr.reset(new LowbitsAlignedToBytesTensorFormat{size_nbits});
    }
    return impl_to_tensor_format(ptr.get());
}

TensorFormat LowbitsAlignedToBytesTensorFormat::deserialize(
        const Handle*, const void* buf, size_t size) {
    megdnn_assert(size == sizeof(SerializePack));
    auto pack = *static_cast<const SerializePack*>(buf);
    return make(pack.size_nbits);
}

// vim: syntax=cpp.doxygen
