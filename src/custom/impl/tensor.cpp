/**
 * \file src/custom/impl/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/common.h"

#if MGB_CUSTOM_OP

#include <algorithm>
#include <cctype>
#include "megbrain/comp_node.h"
#include "megbrain/custom/tensor.h"
#include "megbrain/tensor.h"

using namespace mgb;

namespace custom {

template <typename T>
SmallVector<T> to_builtin_vector(const std::vector<T>& custom_data) {
    SmallVector<T> builtin_data(custom_data.size());
    memcpy(builtin_data.data(), custom_data.data(), sizeof(T) * custom_data.size());
    return builtin_data;
}

using DeviceImpl = CompNode;
using ShapeImpl = megdnn::TensorShape;
using DTypeImpl = megdnn::DType;
using FormatImpl = megdnn::TensorLayout::Format;
using TensorImpl = DeviceTensorND;

#define DeviceImplRef(rawptr) (*reinterpret_cast<DeviceImpl*>(rawptr))
#define ShapeImplRef(rawptr)  (*reinterpret_cast<ShapeImpl*>(rawptr))
#define DTypeImplRef(rawptr)  (*reinterpret_cast<DTypeImpl*>(rawptr))
#define FormatImplRef(rawptr) (*reinterpret_cast<FormatImpl*>(rawptr))
#define TensorImplRef(rawptr) (*reinterpret_cast<TensorImpl*>(rawptr))

#define DeviceImplConstRef(rawptr) \
    static_cast<const DeviceImpl&>(*reinterpret_cast<const DeviceImpl*>(rawptr))
#define ShapeImplConstRef(rawptr) \
    static_cast<const ShapeImpl&>(*reinterpret_cast<const ShapeImpl*>(rawptr))
#define DTypeImplConstRef(rawptr) \
    static_cast<const DTypeImpl&>(*reinterpret_cast<const DTypeImpl*>(rawptr))
#define FormatImplConstRef(rawptr) \
    static_cast<const FormatImpl&>(*reinterpret_cast<const FormatImpl*>(rawptr))
#define TensorImplConstRef(rawptr) \
    static_cast<const TensorImpl&>(*reinterpret_cast<const TensorImpl*>(rawptr))

static std::unordered_map<
        DeviceImpl::DeviceType, std::string, EnumHash<DeviceImpl::DeviceType>,
        EnumCmp<DeviceImpl::DeviceType>>
        dev_benum2cstr;
static std::unordered_map<
        DeviceImpl::DeviceType, DeviceEnum, EnumHash<DeviceImpl::DeviceType>,
        EnumCmp<DeviceImpl::DeviceType>>
        dev_benum2cenum;
static std::unordered_map<std::string, std::string> dev_cstr2bstr;
static std::unordered_map<
        DeviceEnum, std::string, EnumHash<DeviceEnum>, EnumCmp<DeviceEnum>>
        dev_cenum2bstr;

#define CUSTOM_BIND_DEVICE(custom_impl, builtin_device, builtin_str)            \
    auto be2cs##custom_impl = dev_benum2cstr.emplace(                           \
            DeviceImpl::DeviceType::builtin_device, std::string(#custom_impl)); \
    auto be2ce##custom_impl = dev_benum2cenum.emplace(                          \
            DeviceImpl::DeviceType::builtin_device, DeviceEnum::custom_impl);   \
    auto cs2bs##custom_impl = dev_cstr2bstr.emplace(                            \
            std::string(#custom_impl), std::string(builtin_str));               \
    auto ce2bs##custom_impl =                                                   \
            dev_cenum2bstr.emplace(DeviceEnum::custom_impl, std::string(builtin_str));

CUSTOM_FOR_EACH_DEVICE_TYPE(CUSTOM_BIND_DEVICE)
#undef CUSTOM_BIND_DEVICE

CUSTOM_PIMPL_CLS_DEFINE(Device)

const void* Device::impl() const {
    return m_impl.get();
}

Device::Device(const void* impl) : m_impl(nullptr, impl_deleter<DeviceImpl>) {
    mgb_assert(impl != nullptr, "invalid ptr");
    if (!DeviceImplConstRef(impl).valid()) {
        m_impl.reset(new DeviceImpl());
        return;
    }

    auto builtin_device_enum = DeviceImplConstRef(impl).device_type();
    mgb_assert(
            dev_benum2cenum.find(builtin_device_enum) != dev_benum2cenum.end(),
            "unsupported compnode type: %s",
            DeviceImplConstRef(impl).to_string().c_str());
    m_impl.reset(new DeviceImpl(DeviceImplConstRef(impl)));
}

Device::Device(const std::string& device) : m_impl(nullptr, impl_deleter<DeviceImpl>) {
    mgb_assert(is_legal(device), "invalid device type: %s", device.c_str());
    std::string builtin_device = dev_cstr2bstr[device];
    m_impl.reset(new DeviceImpl(DeviceImpl::load(builtin_device)));
}

// to avoid the ambiguous from Device(const void *impl)
Device::Device(const char* device) : Device(std::string(device)) {}

Device::Device(DeviceEnum device) : m_impl(nullptr, impl_deleter<DeviceImpl>) {
    mgb_assert(is_legal(device), "invalid device type");
    std::string builtin_device = dev_cenum2bstr[device];
    m_impl.reset(new DeviceImpl(DeviceImpl::load(builtin_device)));
}

std::string Device::str(void) const {
    if (!DeviceImplRef(m_impl.get()).valid()) {
        return "invalid";
    }

    auto builtin_device_type = DeviceImplRef(m_impl.get()).device_type();
    auto iter = dev_benum2cstr.find(builtin_device_type);
    mgb_assert(
            iter != dev_benum2cstr.end(), "invalid device type %s\n",
            DeviceImplRef(m_impl.get()).to_string().c_str());
    return iter->second;
}

DeviceEnum Device::enumv(void) const {
    mgb_assert(
            DeviceImplRef(m_impl.get()).valid(),
            "cannot get the enum value of invalid device");

    auto builtin_device_type = DeviceImplRef(m_impl.get()).device_type();
    auto iter = dev_benum2cenum.find(builtin_device_type);
    mgb_assert(
            iter != dev_benum2cenum.end(), "invalid device type %s\n",
            DeviceImplRef(m_impl.get()).to_string().c_str());
    return iter->second;
}

bool Device::is_legal(const std::string& device_type) {
    return dev_cstr2bstr.find(device_type) != dev_cstr2bstr.end();
}

bool Device::is_legal(DeviceEnum device_type) {
    return dev_cenum2bstr.find(device_type) != dev_cenum2bstr.end();
}

std::vector<std::string> Device::legal_devices(void) {
    std::vector<std::string> ret;
    for (const auto& kv : dev_cstr2bstr) {
        ret.emplace_back(kv.first);
    }
    return ret;
}

bool operator==(const Device& lhs, const Device& rhs) {
    return lhs.str() == rhs.str();
}

CUSTOM_PIMPL_CLS_DEFINE(Shape)

const void* Shape::impl() const {
    return m_impl.get();
}

Shape::Shape(const void* impl) : m_impl(nullptr, impl_deleter<ShapeImpl>) {
    mgb_assert(impl != nullptr, "invalid ptr");
    m_impl.reset(new ShapeImpl(ShapeImplConstRef(impl)));
}

Shape::Shape(const std::vector<size_t>& rhs)
        : m_impl(nullptr, impl_deleter<ShapeImpl>) {
    m_impl.reset(new ShapeImpl(to_builtin_vector<size_t>(rhs)));
}

Shape::Shape(const std::initializer_list<size_t>& rhs)
        : m_impl(nullptr, impl_deleter<ShapeImpl>) {
    m_impl.reset(new ShapeImpl(rhs));
}

size_t& Shape::operator[](size_t idx) {
    mgb_assert(
            idx < ndim(), "wrong tensor dimension idx: %lu < %lu",
            static_cast<unsigned long>(idx), static_cast<unsigned long>(ndim()));
    return ShapeImplRef(m_impl.get()).operator[](idx);
}

size_t Shape::operator[](size_t idx) const {
    return const_cast<Shape*>(this)->operator[](idx);
}

void Shape::ndim(size_t dim) {
    mgb_assert(
            dim < ShapeImpl::MAX_NDIM, "dimension must <= %lu",
            static_cast<unsigned long>(ShapeImpl::MAX_NDIM));
    ShapeImplRef(m_impl.get()).ndim = dim;
}

size_t Shape::ndim(void) const {
    return ShapeImplRef(m_impl.get()).ndim;
}

bool operator==(const Shape& lhs, const Shape& rhs) {
    return ShapeImplRef(lhs.m_impl.get()).eq_shape(ShapeImplRef(rhs.m_impl.get()));
}

static std::unordered_map<std::string, megdnn::DTypeEnum> dtype_cstr2benum;
static std::unordered_map<
        DTypeEnum, megdnn::DTypeEnum, EnumHash<DTypeEnum>, EnumCmp<DTypeEnum>>
        dtype_cenum2benum;
static std::unordered_map<
        megdnn::DTypeEnum, std::string, EnumHash<megdnn::DTypeEnum>,
        EnumCmp<megdnn::DTypeEnum>>
        dtype_benum2cstr;
static std::unordered_map<
        megdnn::DTypeEnum, DTypeEnum, EnumHash<megdnn::DTypeEnum>,
        EnumCmp<megdnn::DTypeEnum>>
        dtype_benum2cenum;
static std::unordered_map<
        DTypeEnum, std::string, EnumHash<DTypeEnum>, EnumCmp<DTypeEnum>>
        dtype_cenum2cstr;

#define CUSTOM_BIND_DTYPE(custom_impl, builtin_dtype, ctype)              \
    auto cs2be##custom_impl = dtype_cstr2benum.emplace(                   \
            std::string(#custom_impl), megdnn::DTypeEnum::builtin_dtype); \
    auto ce2be##custom_impl = dtype_cenum2benum.emplace(                  \
            DTypeEnum::custom_impl, megdnn::DTypeEnum::builtin_dtype);    \
    auto be2cs##custom_impl = dtype_benum2cstr.emplace(                   \
            megdnn::DTypeEnum::builtin_dtype, std::string(#custom_impl)); \
    auto be2ce##custom_impl = dtype_benum2cenum.emplace(                  \
            megdnn::DTypeEnum::builtin_dtype, DTypeEnum::custom_impl);    \
    auto ce2cs##custom_impl = dtype_cenum2cstr.emplace(                   \
            DTypeEnum::custom_impl, std::string(#custom_impl));

CUSTOM_FOR_EACH_TENSOR_DATA_TYPE(CUSTOM_BIND_DTYPE)
#undef CUSTOM_BIND_DTYPE

CUSTOM_PIMPL_CLS_DEFINE(DType)

const void* DType::impl() const {
    return m_impl.get();
}

DType::DType(const void* impl) : m_impl(nullptr, impl_deleter<DTypeImpl>) {
    mgb_assert(impl != nullptr, "invalid ptr");
    m_impl.reset(new DTypeImpl(DTypeImplConstRef(impl)));
}

DType::DType(const std::string& dtype) : m_impl(nullptr, impl_deleter<DTypeImpl>) {
    auto iter = dtype_cstr2benum.find(dtype);
    mgb_assert(iter != dtype_cstr2benum.end(), "invalid dtype %s", dtype.c_str());
    mgb_assert(
            dtype[0] != 'q',
            "can not construct quantized dtype "
            "%s without scale and zero_point",
            dtype.c_str());
    m_impl.reset(new DTypeImpl(DTypeImpl::from_enum(iter->second)));
}

DType::DType(const char* dtype) : DType(std::string(dtype)) {}

DType::DType(const std::string& dtype, float scale, uint8_t zero_point)
        : m_impl(nullptr, impl_deleter<DTypeImpl>) {
    auto iter = dtype_cstr2benum.find(dtype);
    mgb_assert(iter != dtype_cstr2benum.end(), "invalid dtype %s", dtype.c_str());
    mgb_assert(
            dtype[0] == 'q',
            "given scale/zero_point to construct "
            "non-quantized dtype: %s is not allowed",
            dtype.c_str());
    if (dtype == "quint8") {
        m_impl.reset(new megdnn::ParameterizedDType<megdnn::DTypeEnum::Quantized8Asymm>(
                scale, zero_point));
    } else {
        mgb_assert(
                zero_point == 0, "invalid zero point %d for dtype %s", zero_point,
                dtype.c_str());
        if (dtype == "qint8") {
            m_impl.reset(new megdnn::ParameterizedDType<megdnn::DTypeEnum::QuantizedS8>(
                    scale));
        } else if (dtype == "qint16") {
            m_impl.reset(
                    new megdnn::ParameterizedDType<megdnn::DTypeEnum::QuantizedS16>(
                            scale));
        } else if (dtype == "qint32") {
            m_impl.reset(
                    new megdnn::ParameterizedDType<megdnn::DTypeEnum::QuantizedS32>(
                            scale));
        } else {
            mgb_assert(false, "invalid dtype %s", dtype.c_str());
        }
    }
}

DType::DType(const char* dtype, float scale, uint8_t zero_point)
        : DType(std::string(dtype), scale, zero_point) {}

DType::DType(DTypeEnum dtype) : m_impl(nullptr, impl_deleter<DTypeImpl>) {
    auto iter = dtype_cenum2benum.find(dtype);
    mgb_assert(iter != dtype_cenum2benum.end(), "invalid dtype");
    mgb_assert(
            dtype < DTypeEnum::quint8,
            "can not construct quantized dtype without scale and zero_point");
    m_impl.reset(new DTypeImpl(DTypeImpl::from_enum(iter->second)));
}

DType::DType(DTypeEnum dtype, float scale, uint8_t zero_point)
        : DType(dtype_cenum2cstr.find(dtype)->second, scale, zero_point) {}

std::string DType::str(void) const {
    if (!DTypeImplRef(m_impl.get()).valid())
        return "invalid";
    auto iter = dtype_benum2cstr.find(DTypeImplRef(m_impl.get()).enumv());
    if (iter == dtype_benum2cstr.end())
        return "invalid";
    return iter->second;
}

DTypeEnum DType::enumv(void) const {
    auto iter = dtype_benum2cenum.find(DTypeImplRef(m_impl.get()).enumv());
    mgb_assert(iter != dtype_benum2cenum.end(), "invalid dtype");
    return iter->second;
}

float DType::scale() const {
    if (enumv() == DTypeEnum::qint8) {
        return DTypeImplRef(m_impl.get()).param<dtype::QuantizedS8>().scale;
    } else if (enumv() == DTypeEnum::qint16) {
        return DTypeImplRef(m_impl.get()).param<dtype::QuantizedS16>().scale;
    } else if (enumv() == DTypeEnum::qint32) {
        return DTypeImplRef(m_impl.get()).param<dtype::QuantizedS32>().scale;
    } else if (enumv() == DTypeEnum::quint8) {
        return DTypeImplRef(m_impl.get()).param<dtype::Quantized8Asymm>().scale;
    } else {
        mgb_assert(false, "dtype %s has no scale", str().c_str());
        return 0.f;
    }
}

uint8_t DType::zero_point() const {
    mgb_assert(
            enumv() == DTypeEnum::quint8, "dtype %s has no zero point", str().c_str());
    return DTypeImplRef(m_impl.get()).param<dtype::Quantized8Asymm>().zero_point;
}

bool DType::is_legal(const std::string& dtype) {
    return dtype_cstr2benum.find(dtype) != dtype_cstr2benum.end();
}

bool DType::is_legal(const DTypeEnum& dtype) {
    return dtype_cenum2benum.find(dtype) != dtype_cenum2benum.end();
}

std::vector<std::string> DType::legal_dtypes(void) {
    std::vector<std::string> ret;
    for (const auto& kv : dtype_cstr2benum)
        ret.emplace_back(kv.first);
    return ret;
}

bool operator==(const DType& lhs, const DType& rhs) {
    return DTypeImplRef(lhs.m_impl.get()) == DTypeImplRef(rhs.m_impl.get());
}

bool operator==(const DType& lhs, const std::string& rhs) {
    return lhs.str() == rhs;
}

bool operator==(const DType& lhs, const char* rhs) {
    return operator==(lhs, std::string(rhs));
}

bool operator==(const std::string& lhs, const DType& rhs) {
    return operator==(rhs, lhs);
}

bool operator==(const char* lhs, const DType& rhs) {
    return operator==(rhs, std::string(lhs));
}

CUSTOM_PIMPL_CLS_DEFINE(Format)

const void* Format::impl() const {
    return m_impl.get();
}

Format::Format(const void* impl) : m_impl(nullptr, impl_deleter<FormatImpl>) {
    mgb_assert(impl != nullptr, "invalid ptr");
    mgb_assert(
            FormatImplConstRef(impl).is_default(),
            "only default format is supported now");

    m_impl.reset(new FormatImpl(FormatImplConstRef(impl)));
}

Format::Format(const std::string& format) : m_impl(nullptr, impl_deleter<FormatImpl>) {
    mgb_assert(format == "default", "only default format is supported now");
    m_impl.reset(new FormatImpl());
}

Format::Format(const char* format) : Format(std::string(format)) {}

std::string Format::str(void) const {
    return FormatImplRef(m_impl.get()).to_string();
}

bool Format::is_default(void) const {
    return FormatImplRef(m_impl.get()).is_default();
}

const void* Tensor::impl(void) const {
    return m_tensor;
}

Tensor::Tensor(const void* impl) {
    mgb_assert(impl != nullptr, "invalid ptr");
    m_tensor = const_cast<void*>(impl);
}

const size_t* Tensor::shapes_raw(void) const {
    return TensorImplRef(m_tensor).shape().shape;
}

const ptrdiff_t* Tensor::strides_raw(void) const {
    return TensorImplRef(m_tensor).layout().stride;
}

Tensor::Tensor(const Tensor& rhs) {
    mgb_assert(rhs.m_tensor != nullptr, "invalid rhs for copy constructor\n");
    m_tensor = rhs.m_tensor;
}

Tensor& Tensor::operator=(const Tensor& rhs) {
    mgb_assert(rhs.m_tensor != nullptr, "invalid rhs for assignment operator");
    if (&rhs == this || rhs.m_tensor == m_tensor)
        return *this;
    m_tensor = rhs.m_tensor;
    return *this;
}

Shape Tensor::shape(void) const {
    auto builtin = TensorImplRef(m_tensor).shape();
    return Shape(&builtin);
}

DType Tensor::dtype(void) const {
    auto builtin = TensorImplRef(m_tensor).dtype();
    return DType(&builtin);
}

Format Tensor::format(void) const {
    auto builtin = TensorImplRef(m_tensor).format();
    return Format(&builtin);
}

Device Tensor::device(void) const {
    auto builtin = TensorImplRef(m_tensor).comp_node();
    return Device(&builtin);
}

size_t Tensor::size(void) const {
    return TensorImplRef(m_tensor).shape().total_nr_elems();
}

std::vector<ptrdiff_t> Tensor::stride(void) const {
    std::vector<ptrdiff_t> ret(TensorImplRef(m_tensor).shape().ndim);
    for (size_t i = 0; i < ret.size(); i++)
        ret[i] = TensorImplRef(m_tensor).layout().stride[i];
    return ret;
}

float Tensor::scale(void) const {
    return dtype().scale();
}

uint8_t Tensor::zero_point(void) const {
    return dtype().zero_point();
}

void* Tensor::data(void) {
    return static_cast<void*>(TensorImplRef(m_tensor).raw_ptr());
}

const void* Tensor::data(void) const {
    return static_cast<const void*>(TensorImplRef(m_tensor).raw_ptr());
}

}  // namespace custom

#endif
