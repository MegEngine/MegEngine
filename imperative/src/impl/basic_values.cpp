#include "megbrain/imperative/basic_values.h"

namespace mgb {
namespace imperative {

std::string ShapeValue::to_string() const {
    return ssprintf("ValueShape%s", ValueShape::to_string().c_str());
}

std::string CompNodeValue::to_string() const {
    return CompNode::to_string();
}

std::string BoolValue::to_string() const {
    return (*this) ? "true" : "false";
}

std::string IntegerValue::to_string() const {
    return std::to_string((int)*this);
}

std::string HostStorage::to_string() const {
    return ssprintf("HostStorage{device=%s}", comp_node().to_string().c_str());
}

std::string DeviceStorage::to_string() const {
    return ssprintf("DeviceStorage{device=%s}", comp_node().to_string().c_str());
}

std::string HostValue::to_string() const {
    return ssprintf(
            "HostValue{device=%s, dtype=%s, shape=%s}", device().to_string().c_str(),
            dtype().name(), shape().to_string().c_str());
}

HostTensorND HostTensor::as_nd(bool allow_scalar) const {
    HostTensorND nd;
    TensorShape tensor_shape;
    if (m_shape.is_scalar()) {
        mgb_assert(allow_scalar);
        tensor_shape = TensorShape{1};
    } else {
        tensor_shape = m_shape.as_tensor_shape();
    }
    nd.reset(m_storage, {tensor_shape, dtype()});
    return nd;
}

std::string DeviceValue::to_string() const {
    return ssprintf(
            "DeviceValue{device=%s, dtype=%s, shape=%s}", device().to_string().c_str(),
            dtype().name(), shape().to_string().c_str());
}

DeviceTensorND DeviceTensor::as_nd(bool allow_scalar) const {
    DeviceTensorND nd;
    TensorShape tensor_shape;
    if (m_shape.is_scalar()) {
        mgb_assert(allow_scalar);
        tensor_shape = TensorShape{1};
    } else {
        tensor_shape = m_shape.as_tensor_shape();
    }
    nd.reset(m_storage, {tensor_shape, dtype()});
    return nd;
}

std::string FunctionValue::to_string() const {
    return ssprintf("FunctionValue{type=%s}", target_type().name());
}

std::string DTypeValue::to_string() const {
    return DType::name();
}

std::string StringValue::to_string() const {
    return imperative::quoted((std::string&)*this);
}

std::string ErrorValue::to_string() const {
    return ssprintf("ErrorValue{message=%s}", message().c_str());
}

}  // namespace imperative
}  // namespace mgb
