#include "lite/tensor.h"
#include "function_base.h"
#include "tensor_impl_base.h"
#if LITE_BUILD_WITH_MGE
#include "megbrain/comp_node.h"
#include "megbrain/tensor.h"
#include "mge/function_dft.h"
#include "mge/tensor_impl.h"
#endif

#include <memory>

using namespace lite;

size_t Layout::get_elem_size() const {
    size_t elesize = 1;
    switch (data_type) {
        case LiteDataType::LITE_INT64:
            elesize = 8;
            break;
        case LiteDataType::LITE_FLOAT:
        case LiteDataType::LITE_INT:
        case LiteDataType::LITE_UINT:
            elesize = 4;
            break;
        case LiteDataType::LITE_HALF:
        case LiteDataType::LITE_INT16:
        case LiteDataType::LITE_UINT16:
            elesize = 2;
            break;
        case LiteDataType::LITE_INT8:
        case LiteDataType::LITE_UINT8:
            elesize = 1;
            break;
        default:
            LITE_THROW("not support data type.");
    }
    return elesize;
}

bool Layout::operator==(const Layout& other) const {
    bool equal = true;
    equal &= (ndim == other.ndim);
    equal &= (data_type == other.data_type);
    for (size_t i = 0; i < ndim; i++) {
        equal &= (shapes[i] == other.shapes[i]);
    }
    return equal;
}

Tensor::~Tensor() = default;

Tensor::Tensor() {
    LITE_ERROR_HANDLER_BEGIN
    m_tensor_impl =
            call_func<TensorImplDft, std::shared_ptr<lite::Tensor::TensorImplBase>>(
                    "create_tensor");
    LITE_ERROR_HANDLER_END
}

Tensor::Tensor(LiteDeviceType device_type, bool is_pinned_host)
        : m_is_pinned_host(is_pinned_host), m_device_type(device_type) {
    LITE_ERROR_HANDLER_BEGIN
    m_tensor_impl =
            call_func<TensorImplDft, std::shared_ptr<lite::Tensor::TensorImplBase>>(
                    "create_tensor", device_type, is_pinned_host);
    LITE_ERROR_HANDLER_END
}

Tensor::Tensor(LiteDeviceType device_type, const Layout& layout, bool is_pinned_host)
        : m_is_pinned_host(is_pinned_host),
          m_layout(layout),
          m_device_type(device_type) {
    LITE_ERROR_HANDLER_BEGIN
    m_tensor_impl =
            call_func<TensorImplDft, std::shared_ptr<lite::Tensor::TensorImplBase>>(
                    "create_tensor", device_type, layout, is_pinned_host);
    LITE_ERROR_HANDLER_END
}

Tensor::Tensor(
        int device_id, LiteDeviceType device_type, const Layout& layout,
        bool is_pinned_host)
        : m_is_pinned_host(is_pinned_host),
          m_device_id(device_id),
          m_layout(layout),
          m_device_type(device_type) {
    LITE_ERROR_HANDLER_BEGIN
    m_tensor_impl =
            call_func<TensorImplDft, std::shared_ptr<lite::Tensor::TensorImplBase>>(
                    "create_tensor", device_id, device_type, layout, is_pinned_host);
    LITE_ERROR_HANDLER_END
}

Tensor::Tensor(
        int device_id, int stream_id, LiteDeviceType device_type, bool is_pinned_host)
        : m_is_pinned_host(is_pinned_host),
          m_device_id(device_id),
          m_device_type(device_type) {
    LITE_ERROR_HANDLER_BEGIN
    m_tensor_impl =
            call_func<TensorImplDft, std::shared_ptr<lite::Tensor::TensorImplBase>>(
                    "create_tensor", device_id, stream_id, device_type, is_pinned_host);
    LITE_ERROR_HANDLER_END
}

Tensor::Tensor(
        LiteBackend backend, LiteDeviceType device_type, int device_id,
        const Layout& layout, bool is_pinned_host) {
    if (backend == LiteBackend::LITE_DEFAULT) {
        m_tensor_impl =
                call_func<TensorImplDft, std::shared_ptr<lite::Tensor::TensorImplBase>>(
                        "create_tensor", device_id, device_type, layout,
                        is_pinned_host);
    } else {
        LITE_MARK_USED_VAR(device_type);
        LITE_MARK_USED_VAR(is_pinned_host);
        LITE_MARK_USED_VAR(layout);
        LITE_MARK_USED_VAR(device_id);
        LITE_THROW("unknow backend, enum id is : %d.");
    }
}

void Tensor::reshape(const std::vector<int>& shape) {
    LITE_ASSERT(m_layout.ndim > 0, "The tensor to be reshape is empty.");
    uint32_t length = shape.size();
    LITE_ASSERT(length < Layout::MAXDIM, "The ndim of reshape input is too large.");
    Layout new_layout = m_layout;
    new_layout.ndim = length;
    size_t total_length = get_tensor_total_size_in_byte() / m_layout.get_elem_size();
    uint32_t unfixed_number = 0;
    uint32_t unfixed_index = 0;
    for (uint32_t i = 0; i < length; i++) {
        if (shape[i] == -1) {
            unfixed_number += 1;
            unfixed_index = i;
        } else {
            LITE_ASSERT(shape[i] > 0, "The reshape inputs invalid.");
            new_layout.shapes[i] = shape[i];
        }
    }
    LITE_ASSERT(unfixed_number <= 1, "The reshape inputs invalid.");
    if (unfixed_number) {
        size_t left = total_length;
        for (uint32_t i = 0; i < length; i++) {
            if (i == unfixed_index) {
                continue;
            } else {
                LITE_ASSERT(
                        left > 0 && (left % new_layout.shapes[i] == 0),
                        "The reshape inputs invalid.");
                left = left / new_layout.shapes[i];
            }
        }
        LITE_ASSERT(left > 0, "The reshape inputs invalid.");
        new_layout.shapes[unfixed_index] = left;
    }
    size_t new_total = 1;
    for (uint32_t i = 0; i < length; i++) {
        new_total *= new_layout.shapes[i];
    }
    LITE_ASSERT(new_total == total_length, "The reshape inputs invalid.");
    m_layout = new_layout;
    m_tensor_impl->reshape(m_layout);
}

size_t Tensor::get_tensor_total_size_in_byte() const {
    LITE_ERROR_HANDLER_BEGIN
    size_t elemsize = m_layout.get_elem_size();
    size_t total = m_layout.ndim == 0 ? 0 : 1;
    for (size_t i = 0; i < m_layout.ndim; i++) {
        total *= m_layout.shapes[i];
    }
    return total * elemsize;
    LITE_ERROR_HANDLER_END
}

void* Tensor::get_memory_ptr() const {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_layout.ndim != 0, "Tensor layout is not valid when get memory ptr.");
    return m_tensor_impl->get_memory_ptr();
    LITE_ERROR_HANDLER_END
}

void* Tensor::get_memory_ptr(const std::vector<size_t>& idx) const {
    LITE_ERROR_HANDLER_BEGIN
    return m_tensor_impl->get_memory_ptr(idx);
    LITE_ERROR_HANDLER_END
}

std::shared_ptr<Tensor> Tensor::slice(
        const std::vector<size_t>& start, const std::vector<size_t>& end,
        const std::vector<size_t>& step) {
    LITE_ERROR_HANDLER_BEGIN
    auto ret = m_tensor_impl->slice(start, end, step);
    ret->update_from_implement();
    return ret;
    LITE_ERROR_HANDLER_END
}

void Tensor::fill_zero() {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(
            m_layout.ndim > 0, "fill_zero can't apply on a tensor with empty layout.");
    m_tensor_impl->fill_zero();
    LITE_ERROR_HANDLER_END
}

void Tensor::share_memory_with(const Tensor& src_tensor) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(src_tensor.m_layout.ndim > 0, "To be shared tensor with empty layout.");
    m_tensor_impl->share_memory_with(src_tensor.m_tensor_impl.get());
    update_from_implement();
    LITE_ERROR_HANDLER_END
}

void Tensor::set_layout(const Layout& layout) {
    LITE_ERROR_HANDLER_BEGIN
    m_layout = layout;
    m_tensor_impl->set_layout(layout);
    LITE_ERROR_HANDLER_END
}

void Tensor::reset(void* prepared_data, size_t data_length_in_byte) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(m_layout.ndim, "Tensor layout is empty, please reset with layout");
    LITE_ASSERT(
            data_length_in_byte >= get_tensor_total_size_in_byte(),
            "the memory reset to the tensor is too small.");
    m_tensor_impl->reset(prepared_data);
    LITE_ERROR_HANDLER_END
}

void Tensor::reset(void* prepared_data, const Layout& layout) {
    LITE_ERROR_HANDLER_BEGIN
    m_layout = layout;
    m_tensor_impl->reset(prepared_data, layout);
    LITE_ERROR_HANDLER_END
}

bool Tensor::is_continue_memory() const {
    LITE_ERROR_HANDLER_BEGIN
    return m_tensor_impl->is_continue_memory();
    LITE_ERROR_HANDLER_END
}

void Tensor::copy_from(const Tensor& src) {
    LITE_ERROR_HANDLER_BEGIN
    LITE_ASSERT(
            src.get_layout().ndim != 0,
            "when tensor copy, the src tensor layout is empty.");
    m_tensor_impl->copy_from(src.m_tensor_impl.get());
    update_from_implement();
    LITE_ERROR_HANDLER_END
}

void Tensor::update_from_implement() {
    LITE_ERROR_HANDLER_BEGIN
    m_layout = m_tensor_impl->get_layout();
    m_device_type = m_tensor_impl->get_device_type();
    m_device_id = m_tensor_impl->get_device_id();
    m_is_pinned_host = m_tensor_impl->is_pinned_host();
    LITE_ERROR_HANDLER_END
}

void LiteAny::type_missmatch(size_t expect, size_t get) const {
    LITE_THROW(ssprintf(
            "The type store in LiteAny is not match the visit type, type of "
            "storage enum is %zu, type of visit enum is %zu.",
            expect, get));
}

namespace lite {
#define GET_TYPE(ctype, ENUM)                        \
    template <>                                      \
    LiteAny::Type LiteAny::get_type<ctype>() const { \
        return ENUM;                                 \
    }
GET_TYPE(std::string, STRING)
GET_TYPE(int32_t, INT32)
GET_TYPE(uint32_t, UINT32)
GET_TYPE(int8_t, INT8)
GET_TYPE(uint8_t, UINT8)
GET_TYPE(int64_t, INT64)
GET_TYPE(uint64_t, UINT64)
GET_TYPE(float, FLOAT)
GET_TYPE(bool, BOOL)
GET_TYPE(void*, VOID_PTR)
}  // namespace lite

std::shared_ptr<Tensor> TensorUtils::concat(
        const std::vector<Tensor>& tensors, int dim, LiteDeviceType dst_device,
        int dst_device_id) {
    if (tensors.size() <= 0) {
        return std::make_shared<Tensor>();
    }
    if (dst_device == LiteDeviceType::LITE_DEVICE_DEFAULT) {
        dst_device = tensors.front().get_device_type();
    }
    if (dst_device_id == -1) {
        dst_device_id = tensors.front().get_device_id();
    }
    bool is_pinned_host = tensors.front().is_pinned_host();
    auto layout = tensors.front().get_layout();
    LITE_ASSERT(static_cast<int>(layout.ndim) > dim, "the dim in concat is error.");
    size_t sum_in_dim = layout.shapes[dim];
    for (size_t i = 1; i < tensors.size(); ++i) {
        auto other_layout = tensors[i].get_layout();
        LITE_ASSERT(
                other_layout.ndim == layout.ndim,
                "the dim size of tensors is not same!");
        LITE_ASSERT(
                other_layout.data_type == layout.data_type,
                "the dtype of tensors is not same!");
        for (size_t j = 0; j < other_layout.ndim; ++j) {
            if (dim == static_cast<int>(j)) {
                sum_in_dim += other_layout.shapes[j];
                continue;
            }
            LITE_ASSERT(
                    other_layout.shapes[j] == layout.shapes[j],
                    "the shape of tensors is not same!");
        }
    }
    layout.shapes[dim] = sum_in_dim;
    auto result =
            std::make_shared<Tensor>(dst_device_id, dst_device, layout, is_pinned_host);
    size_t index = 0;
    std::vector<size_t> start(dim + 1, 0);
    std::vector<size_t> end(dim + 1, 0);
    for (int i = 0; i < dim; i++) {
        end[i] = layout.shapes[i];
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
        auto&& tensor = tensors[i];
        auto layout = tensor.get_layout();
        if (layout.shapes[dim] == 0)
            continue;
        start[dim] = index;
        end[dim] = index + layout.shapes[dim];
        auto&& sub_dst = result->slice(start, end);
        sub_dst->copy_from(tensor);
        index += layout.shapes[dim];
    }
    return result;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
