#pragma once

#include <future>
#include <iomanip>

#include "megbrain/imperative/op_def.h"
#include "megbrain/imperative/operator.h"
#include "megbrain/imperative/profiler.h"
#include "megbrain/imperative/utils/data_format.h"
#include "megbrain/imperative/utils/helper.h"
#include "megbrain/imperative/utils/value_shape.h"

namespace mgb {
namespace imperative {

class GradKey;

/**
 * \brief apply an OpDef to values
 *
 */
class ApplyOp final : public OperatorImpl<ApplyOp> {
private:
    const OpDef& m_op;

public:
    ApplyOp(const OpDef& op) : m_op(op) {}

    const OpDef& op() const { return m_op; }

    std::string to_string() const override;
    std::string raw_type() const { return "ApplyOp"; }
};

/**
 * \brief get an basic attribute from Value
 *
 */
class GetAttr final : public OperatorImpl<GetAttr, Operator::GetAttrLike> {
public:
    enum Attr {
        None,
        DType,
        Device,
        Shape,
        Value,
        Data,
    };

private:
    Attr m_attr = None;

public:
    GetAttr(Attr attr) : m_attr(attr) {
        mgb_assert(attr != None, "invalid attr value: None");
    }

    Attr attr() const { return m_attr; }
    std::string raw_type() const { return "GetAttr"; }
    std::string to_string() const;
};

/**
 * \brief create a tensor value from host value or device value
 *
 */
class CreateTensor final : public OperatorImpl<CreateTensor> {
public:
    enum Kind {
        Common,   // common mode, h2d can be cached to speed up
        Unique,   // require output value to be unqiue (donnot share memory with other
                  // values)
        Const,    // put as constant (guaranteed to be same each time)
        NoTrace,  // won't be trace in any case, would be used in make_backward_graph
                  // (looking for a better name)
    };
    struct Args {
        std::optional<HostTensorND> host;
        std::optional<DeviceTensorND> device;
        Kind kind;
    };

private:
    Kind m_kind;
    CompNode m_device;
    DType m_dtype;
    ValueShape m_shape;
    Format m_format;

public:
    CreateTensor(
            Kind kind, CompNode device, DType dtype, ValueShape shape,
            Format format = Format::Type::DEFAULT);
    CreateTensor(Kind kind, CompNode device, TensorLayout layout);

    /**
     * \brief utility function to unpack args of CreateTensor
     *
     * \param inputs contains host_storage and device_storage
     * \return Args unpacked args
     */
    Args parse(Span<ValueRef> inputs) const;

    Kind kind() const { return m_kind; }
    CompNode device() const { return m_device; }
    DType dtype() const { return m_dtype; }
    ValueShape shape() const { return m_shape; }
    Format format() const { return m_format; }
    std::string raw_type() const { return "CreateTensor"; }

    std::string to_string() const override;
};

class DTRCommand final : public OperatorImpl<DTRCommand, Operator::GetAttrLike> {
public:
    enum Kind {
        None,
        Drop,
    };

private:
    Kind m_kind = None;

public:
    DTRCommand(Kind kind) : m_kind(kind) {}

    Kind kind() const { return m_kind; }
    std::string raw_type() const { return "DTRCommand"; }

    std::string to_string() const override;

    ValueRefList fallback(Span<ValueRef> inputs) const override { return {}; }
};

// deprecated
class GetName final : public OperatorImpl<GetName, Operator::GetAttrLike> {
public:
    std::string to_string() const override;
    std::string raw_type() const { return "GetName"; }

    ValueRefList fallback(Span<ValueRef> inputs) const override { return {ValueRef()}; }
};

class GetId final : public OperatorImpl<GetId, Operator::GetAttrLike> {
public:
    std::string to_string() const override;
    std::string raw_type() const { return "GetId"; }

    ValueRefList fallback(Span<ValueRef> inputs) const override { return {ValueRef()}; }
};

/**
 * \brief return a value with new name
 *
 */
class RenameValue : public OperatorImpl<RenameValue, Operator::IdentityLike> {
private:
    std::string m_name;

public:
    RenameValue(std::string name) : m_name(name) {}

    std::string name() const { return m_name; }
    std::string raw_type() const { return "RenameValue"; }

    std::string to_string() const override;

    ValueRefList fallback(Span<ValueRef> inputs) const override {
        return {inputs.as_array<1>()[0]};
    }
};

class IsScalar final : public OperatorImpl<IsScalar, Operator::GetAttrLike> {
private:
public:
    std::string to_string() const override;
    std::string raw_type() const { return "IsScalar"; }
};

class GetFormat final : public OperatorImpl<GetFormat, Operator::GetAttrLike> {
public:
    std::string to_string() const override;
    std::string raw_type() const { return "GetFromat"; }
};

class SetFormat final : public OperatorImpl<SetFormat, Operator::IdentityLike> {
private:
    Format m_format;

public:
    SetFormat(std::string format) : m_format(format) {}

    Format format() const { return m_format; }
    std::string raw_type() const { return "SetFromat"; }

    std::string to_string() const override;
};

class AsFormat final : public OperatorImpl<AsFormat, Operator::IdentityLike> {
private:
    Format m_format;

public:
    AsFormat(std::string format) : m_format(format) {}

    Format format() const { return m_format; }
    std::string raw_type() const { return "AsFormat"; }

    std::string to_string() const override;
};

class GetVarVal final : public OperatorImpl<GetVarVal, Operator::GetAttrLike> {
public:
    std::string to_string() const override;
    std::string raw_type() const { return "GetVarVal"; }
};

class CreateNode final : public OperatorImpl<CreateNode> {
private:
    cg::VarNode* m_node;

public:
    CreateNode(cg::VarNode* node) : m_node(node) {}

    cg::VarNode* node() const { return m_node; }
    std::string raw_type() const { return "CreateNode"; }

    std::string to_string() const override;
};

class PushScope final : public OperatorImpl<PushScope> {
public:
    std::string name;
    ScopeType type;
    PushScope(std::string name, ScopeType type) : name{std::move(name)}, type{type} {};
    std::string raw_type() const { return "PushScope"; }
    std::string to_string() const override { return "PushScope"; }
};

class PopScope final : public OperatorImpl<PopScope> {
public:
    std::string name;
    ScopeType type;
    PopScope(std::string name, ScopeType type) : name{std::move(name)}, type{type} {};
    std::string raw_type() const { return "PopScope"; }
    std::string to_string() const override { return "PopScope"; }
};

class DupTensor final : public OperatorImpl<DupTensor, Operator::IdentityLike> {
public:
    std::string to_string() const override { return "DupTensor"; }
    std::string raw_type() const { return "DupTensor"; }
};

}  // namespace imperative
}  // namespace mgb
