/**
 * \file dnn/include/megdnn/tensor_format.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "megdnn/handle.h"

#include "megdnn/internal/visibility_prologue.h"
namespace megdnn {

enum class TensorFormat::Type {
    DEFAULT = 0,        //!< see DefaultTensorFormat
    IMAGE2D_PACK4 = 1,  //!< see Image2DPack4TensorFormat
};

class TensorFormat::ImplBase {
public:
    using Type = TensorFormat::Type;

    virtual size_t init_contiguous_stride(TensorLayout& layout) const = 0;

    virtual bool is_contiguous_spec(const TensorLayout& layout) const = 0;

    virtual TensorLayout collapse_contiguous_spec(
            const TensorLayout& layout) const = 0;

    virtual TensorLayout::Span span_spec(const TensorLayout& layout) const = 0;

    //! a human-readable string description of this TensorFormat
    virtual std::string to_string() const = 0;

    virtual void serialize_append(std::string& result) const = 0;

    Type type() const { return m_type; }

protected:
    ImplBase(Type type) : m_type{type} {}
    virtual ~ImplBase() = default;

    static TensorFormat impl_to_tensor_format(ImplBase* impl) { return {impl}; }

private:
    Type m_type;
};

TensorFormat::Type TensorFormat::type() const {
    return m_impl->type();
}

//! default tensor format that imposes no stride constraints
class DefaultTensorFormat final : public TensorFormat::ImplBase {
public:
    static constexpr Type TYPE = Type::DEFAULT;

    DefaultTensorFormat() : ImplBase(TYPE) {}

    size_t init_contiguous_stride(TensorLayout& layout) const override;

    /*!
     * \brief A tensor is contiguous if logical offset in row-major of any
     * element always equals to its physical offset (i.e. offset considering
     * strides).
     *
     * Empty tensors are not considered to be contiguous.
     */
    bool is_contiguous_spec(const TensorLayout& layout) const override;

    TensorLayout collapse_contiguous_spec(
            const TensorLayout& layout) const override;

    TensorLayout::Span span_spec(const TensorLayout& layout) const override;

    std::string to_string() const override;
    void serialize_append(std::string& result) const override;

    static TensorFormat make();
    static TensorFormat deserialize(const Handle* handle, const void* buf,
                                    size_t size);
};

namespace detail {

/*!
 * \brief 2D image with requirement on row stride
 *
 * \p align_axis is the axis to be aligned, also the first axis of image width.
 * More precisely speaking, `stride[align_axis-1] * dtype.size()` must divide \p
 * align_size_in_elements. Axes from 0 to align_axis-1 would be considered as
 * the height of the image, and other axes are the width.
 *
 * Empty tensors and negative strides are not allowed. Only contiguous or
 * broadcasted cases are allowed.
 *
 * Note: if `stride[align_axis - 1]` is larger than minimal value, it is still
 * considered as contiguous.
 */
class Image2DTensorFormatBase : public TensorFormat::ImplBase {
    size_t m_align_axis, m_align_size_in_elements_log2;

protected:
    Image2DTensorFormatBase(Type type, size_t align_axis,
                            size_t align_size_in_elements);
    virtual ~Image2DTensorFormatBase() = default;

public:
    /*!
     * \brief get alignment requirement in elements
     * \param div_log2 the result would be divided by `(1 << div_log2)`
     */
    size_t align_size_in_elements(size_t div_log2 = 0) const {
        return 1 << (m_align_size_in_elements_log2 > div_log2
                             ? m_align_size_in_elements_log2 - div_log2
                             : 0);
    }

    size_t align_axis() const { return m_align_axis; }

    size_t align_size_in_elements_log2() const {
        return m_align_size_in_elements_log2;
    }

    std::string to_string() const override;

    //! modify the align axis and return a new TensorFormat
    virtual TensorFormat change_axis(size_t axis) const = 0;

    //! number of dtype elems in each row, considering strides
    size_t image_width_elems(const TensorLayout& layout) const;

    //! number of rows
    size_t image_height(const TensorLayout& layout) const;

    void serialize_append(std::string& result) const override;
protected:
    struct SerializePack {
        uint8_t align_axis;
    };
};

template <size_t PIXEL_SIZE>
class Image2DPackedTensorFormatBase : public Image2DTensorFormatBase {
    Handle::HandleVendorType m_vendor_type = Handle::HandleVendorType::NOT_SPEC;
    /*!
     * \brief get fix alignment requirement in bytes, consider m_vendor_type,
     * for example on MALI, CL_DEVICE_IMAGE_PITCH_ALIGNMENT means image_width
     * align COUNT, but mdl needs align size in byte, which equal to
     * (image_width algin count) * sizeof(data_type) * pixel_size
     */
    size_t image_pitch_alignment_in_bytes(size_t align_size_in_elements,
                                          const TensorLayout& layout) const;

protected:
    Image2DPackedTensorFormatBase(Type type, size_t align_axis,
                                  size_t align_size_in_elements,
                                  Handle::HandleVendorType vendor_type)
            : detail::Image2DTensorFormatBase(type, align_axis,
                                              align_size_in_elements),
              m_vendor_type(vendor_type) {}

    virtual ~Image2DPackedTensorFormatBase() = default;

    Handle::HandleVendorType vendor() const { return m_vendor_type; }

public:
    /*!
     * \brief image width in logical pixels exclude padding
     *
     * It is the number of accessible elems (in dtype) divided by PIXEL_SIZE.
     *
     * \see image_row_pitch()
     */
    size_t image_width(const TensorLayout& layout) const;

    //! raise exception if preconditions violated
    void assert_valid(const TensorLayout& layout) const;

    size_t image_row_pitch(const TensorLayout& layout) const;

    //! span for image must include the padding at the last row
    TensorLayout::Span span_spec(const TensorLayout& layout) const override;

    size_t init_contiguous_stride(TensorLayout& layout) const override;

    bool is_contiguous_spec(const TensorLayout& layout) const override;

    TensorLayout collapse_contiguous_spec(
            const TensorLayout& layout) const override;
};
using Image2DPack4TensorFormatBase = Image2DPackedTensorFormatBase<4>;
}  // namespace detail

/*!
 * \brief 2D image that requires stride of width to be aligned, and pack 4 elems
 *      into a pixel
 *
 * This is used for OpenCL.
 */
class Image2DPack4TensorFormat final
        : public detail::Image2DPack4TensorFormatBase {
public:
    static constexpr Type TYPE = Type::IMAGE2D_PACK4;

    //! for internal usage or test purposes
    static TensorFormat make_raw(size_t align_axis,
                                 size_t align_size_in_elements,
                                 Handle::HandleVendorType vendor_type =
                                         Handle::HandleVendorType::NOT_SPEC);

    static TensorFormat make(size_t align_axis, const Handle* handle);

    /*!
     * \brief deserialize on a handle
     *
     * Note that the alignment may be different if deserialized on another
     * handle
     */
    static TensorFormat deserialize(const Handle* handle, const void* buf,
                                    size_t size);

    static bool is_valid_image(const TensorLayout& layout) {
        if (layout.format.type() == TYPE) {
            layout.format.as_impl<Image2DPack4TensorFormat>().assert_valid(
                    layout);
            return true;
        }
        return false;
    }

    TensorFormat change_axis(size_t axis) const override;

private:
    Image2DPack4TensorFormat(size_t align_axis, size_t align_size_in_elements,
                             Handle::HandleVendorType vendor_type)
            : detail::Image2DPack4TensorFormatBase(
                      TYPE, align_axis, align_size_in_elements, vendor_type) {}
};

}  // namespace megdnn

#include "megdnn/internal/visibility_epilogue.h"

// vim: syntax=cpp.doxygen
