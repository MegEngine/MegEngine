#pragma once

#include <future>
#include <iomanip>

#include "megbrain/imperative/utils/data_format.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/utils/value_shape.h"
#include "megbrain/imperative/value.h"

namespace mgb {
namespace imperative {

class GradKey;

using GenericFunction = std::function<ValueRefList(Span<ValueRef>)>;

class ShapeValue final : public PrimitiveValue<ShapeValue, ValueShape> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class CompNodeValue final : public PrimitiveValue<CompNodeValue, CompNode> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class Boolean {
private:
    bool m_value;

public:
    Boolean() = default;
    Boolean(bool value) : m_value(value) {}

    operator bool() const { return m_value; }
};

// TODO: override factory method
class BoolValue final : public PrimitiveValue<BoolValue, Boolean> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class Integer {
private:
    int64_t m_value;

public:
    Integer() = default;
    Integer(int64_t value) : m_value(value) {}

    operator int64_t() const { return m_value; }
};

// TODO: override factory method
class IntegerValue final : public PrimitiveValue<IntegerValue, Integer> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class HostStorage final : public PrimitiveValue<HostStorage, HostTensorStorage> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class DeviceStorage final : public PrimitiveValue<DeviceStorage, DeviceTensorStorage> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class HostTensor {
private:
    DType m_dtype;
    ValueShape m_shape;
    HostTensorStorage m_storage;

public:
    HostTensor() = default;
    HostTensor(DType dtype, ValueShape shape, HostTensorStorage storage)
            : m_dtype(dtype), m_shape(shape), m_storage(storage) {}
    HostTensor(HostTensorND value)
            : HostTensor(
                      value.dtype(), ValueShape::from(value.shape()), value.storage()) {
    }

    DType dtype() const { return m_dtype; }
    const ValueShape& shape() const { return m_shape; }
    CompNode device() const { return m_storage.comp_node(); }
    const HostTensorStorage& storage() const { return m_storage; }
    DTypeScalar item() const {
        // FIXME: check scalar
        mgb_assert(m_shape.total_nr_elems());
        return DTypeScalar::make_from_raw(m_dtype, m_storage.ptr());
    }

    HostTensorND as_nd(bool allow_scalar = false) const;
};

/**
 * \brief like HostTensorND mixin, but allow scalar value
 *
 */
class HostValue final : public PrimitiveValue<HostValue, HostTensor> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class DeviceTensor {
private:
    DType m_dtype;
    ValueShape m_shape;
    DeviceTensorStorage m_storage;

public:
    DeviceTensor() = default;
    DeviceTensor(DType dtype, ValueShape shape, DeviceTensorStorage storage)
            : m_dtype(dtype), m_shape(shape), m_storage(std::move(storage)) {}
    DeviceTensor(const DeviceTensorND& value)
            : DeviceTensor(
                      value.dtype(), ValueShape::from(value.shape()), value.storage()) {
    }

    DType dtype() const { return m_dtype; }
    const ValueShape& shape() const { return m_shape; }
    CompNode device() const { return m_storage.comp_node(); }
    const DeviceTensorStorage& storage() const { return m_storage; }

    DeviceTensorND as_nd(bool allow_scalar = false) const;
};

/**
 * \brief like DeviceTensorND mixin, but allow scalar value
 *
 */
class DeviceValue final : public PrimitiveValue<DeviceValue, DeviceTensor> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class FunctionValue final : public PrimitiveValue<FunctionValue, GenericFunction> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class DTypeValue final : public PrimitiveValue<DTypeValue, DType> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class FormatValue final : public PrimitiveValue<FormatValue, Format> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override { return Format::to_string(); }
};

class StringValue final : public PrimitiveValue<StringValue, std::string> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class Error {
protected:
    std::string m_message;

public:
    Error() = default;
    Error(std::string message) : m_message(message) {}

    std::string message() const { return m_message; }
};

class ErrorValue final : public PrimitiveValue<ErrorValue, Error> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override;
};

class NodeStorage {
private:
    cg::VarNode* m_node;

public:
    NodeStorage() = default;
    NodeStorage(VarNode* node) : m_node(node) {}
    VarNode* node() const { return m_node; }
    ComputingGraph* graph() const { return m_node->owner_graph(); }
    std::string to_string() const { return m_node->name(); }
};

class NodeValue final : public PrimitiveValue<NodeValue, NodeStorage> {
public:
    using PrimitiveValue::PrimitiveValue;

    std::string to_string() const override { return NodeStorage::to_string(); }
};

}  // namespace imperative
}  // namespace mgb
