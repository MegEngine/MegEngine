#include "megbrain/imperative/basic_operators.h"

#include "megbrain/imperative/basic_values.h"

namespace mgb {
namespace imperative {

std::string ApplyOp::to_string() const {
    return m_op.to_string();
}

std::string GetAttr::to_string() const {
    std::string buffer;
    const char* attr_name = ([&] {
        switch (m_attr) {
            case None:
                return "None";
            case DType:
                return "DType";
            case Device:
                return "Device";
            case Shape:
                return "Shape";
            case Value:
                return "Value";
            case Data:
                return "Data";
            default:
                buffer = std::to_string(m_attr);
                return buffer.c_str();
        }
    })();
    return ssprintf("GetAttr{attr=%s}", attr_name);
}

CreateTensor::CreateTensor(Kind kind, CompNode device, DType dtype, ValueShape shape)
        : m_kind(kind), m_device(device), m_dtype(dtype), m_shape(shape) {}

CreateTensor::CreateTensor(Kind kind, CompNode device, TensorLayout layout)
        : m_kind(kind),
          m_device(device),
          m_dtype(layout.dtype),
          m_shape(ValueShape::from(layout)) {
    mgb_assert(
            layout.is_contiguous() || layout.is_empty(), "layout should be contiguous");
}

auto CreateTensor::parse(Span<ValueRef> inputs) const -> Args {
    Args result;
    for (auto&& input : inputs) {
        if (auto host_storage = input.as_ref<HostStorage>()) {
            mgb_assert(!result.host, "duplicated host value");
            result.host.emplace();
            result.host->reset(*host_storage, {shape().as_tensor_shape(), dtype()});
            mgb_assert(result.host->layout().ndim, "invalid shape");
        } else if (auto device_storage = input.as_ref<DeviceStorage>()) {
            mgb_assert(!result.device, "duplicated device value");
            result.device.emplace(device(), shape().as_tensor_shape(), dtype());
            result.device->reset(*device_storage, {shape().as_tensor_shape(), dtype()});
            mgb_assert(result.device->layout().ndim, "invalid shape");
        } else {
            mgb_throw(
                    MegBrainError,
                    "unknown input type, expects HostStorage or DeviceStorage, got "
                    "%s",
                    input.to_string().c_str());
        }
    }
    mgb_assert(
            result.host || result.device, "require at least one of host/device value");
    result.kind = kind();
    return result;
}

std::string CreateTensor::to_string() const {
    return ssprintf(
            "CreateTensor{kind=%d, device=%s, dtype=%s, shape=%s}", (int)m_kind,
            m_device.to_string().c_str(), m_dtype.name(), m_shape.to_string().c_str());
}

std::string DTRCommand::to_string() const {
    return ssprintf("DTRCommandValue{kind=%d}", (int)m_kind);
}

std::string GetName::to_string() const {
    return "GetName{}";
}

std::string RenameValue::to_string() const {
    return ssprintf("RenameValue{name=%s}", imperative::quoted(m_name).c_str());
}

std::string IsScalar::to_string() const {
    return "IsScalar";
}

}  // namespace imperative
}  // namespace mgb
